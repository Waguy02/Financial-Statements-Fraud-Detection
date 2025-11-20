import json
import logging
import os
import random
import re
import shutil
import tempfile
import time
import uuid
from io import BytesIO
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai
import PyPDF2
import ratelimit
from backoff import expo, on_exception
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    AAER_CONTENTS_DOWNLOADED_PATH,
    GEMINI_API_KEY,
    VERTEX_JSON_KEYS_PATH,
)
from researchpkg.anomaly_detection.preprocessing.utils import load_vllm_configs

# Constants for API rate limiting
MAX_RPM_GEMINI = 15  # Maximum requests per minute
MAX_RETRY = 3  # Maximum number of retries

# Global variables for API key rotation
api_keys = []
api_key_cycle = None
request_timestamps = []


def _load_api_keys():
    """
    Load Gemini API keys from the JSON file with round-robin rotation.

    Returns:
        List[Tuple]: List of tuples with (api_key, service_account_info, project_id, location)
    """
    if not Path(VERTEX_JSON_KEYS_PATH).exists():
        raise FileNotFoundError(
            f"Gemini API Keys file not found at {VERTEX_JSON_KEYS_PATH}. "
            "Create this file with API keys to use this extractor."
        )

    with open(VERTEX_JSON_KEYS_PATH, "r") as f:
        data = json.load(f)

    # Extract valid keys (not disabled)
    keys = []
    for entry in data:
        if not entry.get("disabled", False):
            key = entry["key"]
            service_account_info = entry["service_account"]
            project_id = entry["project_id"]
            location = entry["location"]
            email = entry.get("email", "no email provided")
            keys.append((key, service_account_info, project_id, location))
            print(f"Loaded API key for email: {email}")

    if not keys:
        raise ValueError("No valid API keys found in the keys file.")

    print(f"Loaded {len(keys)} API keys for round-robin rotation")
    return keys


def _get_next_api_key():
    """
    Get the next API key in rotation, respecting rate limits.

    Returns:
        Tuple: (key, service_account_info, project_id, location)
    """
    global api_keys, api_key_cycle, request_timestamps

    # Initialize keys if not already done
    if not api_keys:
        api_keys = _load_api_keys()
        api_key_cycle = cycle(api_keys)

    # Clean up old timestamps (older than 60 seconds)
    current_time = time.time()
    request_timestamps = [t for t in request_timestamps if current_time - t < 60]

    # Check if we're approaching rate limit
    if len(request_timestamps) >= MAX_RPM_GEMINI:
        # Calculate time to wait
        oldest_timestamp = min(request_timestamps)
        wait_time = 61 - (current_time - oldest_timestamp)
        if wait_time > 0:
            print(f"Rate limit approaching. Waiting {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            # Clean timestamps again after waiting
            current_time = time.time()
            request_timestamps = [
                t for t in request_timestamps if current_time - t < 60
            ]

    # Rotate to next key
    current_api_key = next(api_key_cycle)
    request_timestamps.append(time.time())
    return current_api_key


def download_pdf_selenium(url: str):
    """
    Uses the initialized Selenium driver to open a browser and download the PDF.

    :param url: The URL of the PDF file.
    :return: Path to the downloaded PDF file.
    """

    # Selenium imports
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    user_dir = tempfile.mkdtemp()
    # Global Selenium Driver Initialization
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--user-data-dir={user_dir}")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    download_folder = "/tmp/aaer_pdf/"  # Global download folder
    if not Path(download_folder).exists():
        Path(download_folder).mkdir()

    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_folder,  # Set download directory
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,  # Auto-download PDFs
        },
    )

    service = Service(ChromeDriverManager().install())
    time.sleep(1)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    timeout = 20
    spent = 0
    while spent < timeout:
        time.sleep(1)
        # Find the newest PDF file in the download directory
        pdf_file = [
            f
            for f in os.listdir(download_folder)
            if f in url and f.lower().endswith(".pdf")
        ]
        if pdf_file:
            pdf_file = pdf_file[0]
            pdf_path = os.path.join(download_folder, pdf_file)
            # print(f"✅ PDF downloaded successfully: {pdf_path}")
            return pdf_path
        else:
            spent += 1
            continue
    driver.quit()
    return None


def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text from a PDF file using PyPDF2.

    :param pdf_path: The path to the PDF file.
    :return: Extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"

    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")

    return text


def fetch_html_content(url: str) -> str:
    """
    Downloads the content from a given HTML URL and extracts the text using the initialized Selenium driver.

    Args:
        url (str): The HTML URL to fetch content from.

    Returns:
        str: The extracted text content, or None if the download or extraction failed.
    """

    # Selenium imports
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    user_dir = tempfile.mkdtemp()

    # Global Selenium Driver Initialization
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"user-data-dir={user_dir}")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    download_folder = "/tmp/aaer_pdf/"  # Global download folder
    if not Path(download_folder).exists():
        Path(download_folder).mkdir()

    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_folder,  # Set download directory
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,  # Auto-download PDFs
        },
    )
    service = Service(ChromeDriverManager().install())
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)  # Navigate to the example url
        time.sleep(2)  # Wait 2 seconds for javascript to render

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(
            separator="\n", strip=True
        )  # Get all text with newline separator
        print(f"✅ Html content extracted from : {url}")
        return text  # Finally return all the text

    except Exception as e:
        print(f"Selenium HTML error: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def cleanup_file(file_path: str):
    """
    Deletes the specified file.

    :param file_path: Path to the file to delete.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # print(f"🗑️ Deleted file: {file_path}")
    except Exception as e:
        print(f"❌ Error deleting file: {e}")


SYSTEM_PROMPT = """
You are a specialized AI agent tasked with meticulously extracting detailed information about **earnings misstatements** (financial statement violations that directly impact the calculation of reported earnings, income, assets, or liabilities) from U.S. Securities and Exchange Commission (SEC) Accounting and Auditing Enforcement Releases (AAERs). Your goal is to deconstruct complex legal and financial text into structured, factual data.

### **Objective**
Your primary objective is to identify every quarter in which a **true earning misstatement** occurred, specify the company responsible, detail the specific types of earning misstatements based on predefined categories, and describe the fraudulent scheme that led to these misstatements. You must adhere strictly to the formats and rules defined below.

### **Key Definitions: `LIST_MISTATEMENT_TYPE`**
You must categorize all identified **earnings misstatements** using **only** the types from this predefined list. An "earnings misstatement" directly alters the reported financial performance or position (e.g., net income, assets, liabilities, equity balances). Violations related *solely* to disclosure failures that do not alter the numerical financial statements (e.g., failure to disclose related party relationships without affecting specific account balances, or control issues) should *not* be categorized here unless they clearly result in a numerical misstatement of an account listed below.

*   **Revenue**: Overstating or understating sales or income. This includes premature revenue recognition, fictitious sales, or improper income classification directly impacting the income statement.
*   **Other Expense/Shareholder Equity Account**: Manipulating expenses not directly related to cost of goods sold (e.g., operating, selling, general & administrative expenses, R&D), or directly misstating equity accounts (like retained earnings, common stock, additional paid-in capital) through improper accounting entries that affect net income or equity balances. Do not include in this category disclore fraud or governance issues that do not impact financial accounts.
*   **Assets Valuation** : Improperly recognition of assets or their values. This includes inflating asset values (e.g., property, plant & equipment, intangible assets) or failing to recognize impairments, which directly impacts the balance sheet. 
*   **Capitalized Costs as Assets**: Improperly recording expenses as long-term assets (e.g., property, plant & equipment, intangible assets) to inflate current period income by reducing expenses.
*   **Accounts Receivable**: Overstating the money owed by customers. This includes recording fictitious sales, failing to write off uncollectible receivables, or otherwise inflating the asset balance.
*   **Inventory**: Overstating the value of goods for sale. This includes counting non-existent inventory, improper valuation methods, or misclassifying costs.
*   **Cost of Goods Sold (COGS)**: Understating the direct costs of production. Often linked to inventory manipulation (e.g., overstating inventory leads to understated COGS), directly impacting gross profit and net income.
*   **Reserve Account**: Manipulating funds set aside for future contingent liabilities (e.g., warranty, litigation reserves, bad debt reserves). This includes understating reserves to boost current income or overstating them to create "cookie jar" reserves for future manipulation, directly impacting expenses or liabilities.
*   **Liabilities**: Understating company obligations. This includes concealing debt, failing to record accrued expenses (e.g., unbilled services, payroll), or misclassifying liabilities to improve financial ratios or conceal obligations.
*   **Marketable Securities**: Misstating the value of short-term investments. This includes improper valuation (e.g., failing to mark to market when required) or failing to recognize impairment losses, directly impacting asset values and potentially income.
*   **Allowance for Bad Debt**: Understating the estimated uncollectible accounts receivable to inflate net receivables and income. This is a specific type of reserve manipulation.
*   **Payables**: Understating money owed to suppliers. This includes delaying invoice recording, concealing vendor liabilities, or manipulating cut-off dates, directly impacting liabilities and potentially expenses.

---

### **Input Format**
You will receive a dictionary containing:
*   `"aaerNo"`: The unique identifier of the AAER.
*   `"content"`: The full text of the AAER.
*   `"entities"`: A list of dictionaries, each representing an entity (company or individual) involved in the AAER.

---

### **Output Format**
You must generate a JSON list of dictionaries. Each dictionary represents a single fraudulent scheme by a specific company in a specific quarter.

```json
[
    {
        "quarter": "YYYYqQ",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "A concise, factual description of the earning misstatement mechanics, focusing on how the numerical financial statements were altered. You must also justify the selection of the misstatements indicated in the 'misstatements' field. The justification should be provided only when the 'misstatements' field is not empty. The justification should be in the form of a list of sentences, each explaining why a specific misstatement type was selected for that quarter. For example: \n- Revenue because the company recorded fictitious sales transactions.\n- Accounts Receivable because the inflated sales led to an overstatement of amounts owed by customers.",
        "misstatements": ["Type1", "Type2", "Type3", ...],
        "misstating_company": {
            "name": "Company Name",
            "role": "respondent",
            "cik": "0001234567"
        }
    }
]
```

**Key-Specific Requirements:**

*   `quarter`: (String) The quarter of the violation, formatted as `YYYYqQ` (e.g., `2021q3`).
*   `is_fiscal_quarter`: (Boolean) Set to `true` if the AAER mentions "fiscal year," "fiscal quarter," "Form 10-K," or "Form 10-Q." Otherwise, set to `false`. This value should be consistent across all entries in the output.
*   `fraud_scheme_description`: (String) A factual summary of the fraud, explicitly detailing how the company's financial statements were numerically misstated. If no specific scheme directly causing an earning misstatement is described for a quarter, use "N/A".
*   `misstatements`: (List of Strings) A list of misstatement types strictly from `LIST_MISTATEMENT_TYPE`. This list **must** be empty (`[]`) if no specific earning misstatement types can be identified for the quarter, or if the violation described is solely a disclosure failure without impacting financial account balances.
*   `misstating_company`: (Dictionary) An object identifying the company that perpetrated the misstatement for that quarter. Select the entity from the input that is a "company" and is clearly described as committing the earning misstatement. If the CIK is not provided in the input, use an empty string (`""`). If no company can be identified as the perpetrator of an earning misstatement, this list should be empty (`[]`).

---

### **Detailed Instructions & Rules**

1.  **Focus Strictly on Earning Misstatements**: Prioritize violations that directly affect the numerical values of financial accounts (e.g., revenue, expenses, assets, liabilities, equity) on the income statement, balance sheet, or statement of cash flows. **Do not include violations that are purely disclosure-related (e.g., failing to disclose related party transactions if they don't involve a numerical misstatement of funds, or corporate governance issues) unless the text explicitly states these disclosure failures also led to numerical misstatements in the financial statements.**
2.  **Time Period Identification**:
    *   **Specific Quarters**: If the text mentions a specific quarter (e.g., "the second quarter of 2020," "Q2 2020," or a Form 10-Q for the period ending June 30, 2020), extract it as `2020q2`.
    *   **Full Years**: If the text implicates an entire year (e.g., "for fiscal year 2019," "throughout 2019," or a Form 10-K for 2019), you **must** create entries for all four quarters of that year (`2019q1`, `2019q2`, `2019q3`, `2019q4`).
    *   **Applying Year-Level Data**: If a fraud description or misstatement type is given for an entire year, apply that same information to all four quarterly entries for that year.

3.  **Fraud Scheme Description**:
    *   Your description must be a concise, factual summary of the fraud's mechanics for that period, explicitly linking the actions to how they altered reported financial numbers.
    *   Aim to answer: **Who** was involved (e.g., senior management)? **What specific accounting actions** were taken (e.g., backdating contracts, failing to record expenses)? **How** did this directly affect the financial statements (e.g., improperly inflating revenue, understating liabilities)?
    *   If a general scheme spans multiple quarters, repeat the description for each quarter.

4.  **Handling Multiple Companies & Schemes**:
    *   **One Entry Per Company Per Quarter**: A quarter can appear multiple times in the output **only if** different companies were perpetrating distinct fraudulent acts in that same quarter (but this is very rare).
    *   **Consolidation**: If a single company committed multiple earning misstatements in the same quarter, consolidate all information into a single entry for that company and quarter. All relevant `misstatements` should be in one list, and the `fraud_scheme_description` should cover all actions that led to earning misstatements.

5.  **Strict Adherence to `LIST_MISTATEMENT_TYPE`**:
    *   You **must not** include any misstatement type that is not on the provided list.
    *   Infer the type from the description. For example, if the text says "improperly deferred recording of operating expenses," you should map this to `Other Expense/Shareholder Equity Account` or `Liabilities` depending on the context of how the deferral was handled (e.g., if it was an unrecorded expense, it affects liabilities; if it was an improper reclassification within equity, it affects equity). If "inflated the value of unsold goods," map it to `Inventory`.

6.  **No Earning Misstatements Found**:
    *   If the AAER text does not contain enough information to identify specific periods with **earning misstatements** or specific misstatement types from the `LIST_MISTATEMENT_TYPE`, return an empty list (`[]`). Do not include entries for non-earning misstatement violations.

---

### **Step-by-Step Cognitive Process**

To ensure accuracy, follow these steps:

1.  **Full Document Scan for Financial Violations**: Read the entire AAER `content` specifically looking for descriptions of financial manipulations that directly impact the numerical values on the income statement, balance sheet, or statement of cash flows. Distinguish these from purely disclosure-related or governance violations.
2.  **Identify Time References for Earning Misstatements**: Systematically scan the text for all explicit and implicit time periods where financial manipulation occurred (e.g., "Q1 2020", "year ended December 31, 2019", "Form 10-Q for the period ending September 30, 2021"). Note whether they are "fiscal" periods.
3.  **Link Earning Misstatement Actions to Periods**: For each identified time period, locate the corresponding descriptions of fraudulent activities that specifically caused numerical misstatements in financial accounts.
4.  **Attribute to Company**: Determine which company from the `entities` list was responsible for the earning misstatements in that specific period.
5.  **Categorize Earning Misstatements**: Based on the description of the fraudulent activity, select the corresponding types **only** from the `LIST_MISTATEMENT_TYPE`. If an action doesn't fit a type, or is not a direct earning misstatement, do not include it.
6.  **Construct Fraud Description for Earning Misstatements**: Synthesize the details of the scheme (who, what specific numerical alteration, how it affected financials) into a concise description. Ensure the description clearly explains *how* the financial numbers were changed.
7.  **Build Final JSON**: Assemble the extracted information into the required JSON structure, meticulously following all formatting rules. Critically review each entry to confirm it represents a **true earning misstatement** and that the `misstatements` list only contains types from `LIST_MISTATEMENT_TYPE`. Double-check for duplicates and consolidate where necessary.

---

### **Example Output (Updated to reflect "true earning misstatement" focus)**

```json
[
    {
        "quarter": "2019q1",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "Senior management at Company A directed the sales team to backdate software license agreements to improperly recognize revenue from future periods in the first quarter of 2019. This action directly inflated the reported revenue on the income statement and accounts receivable on the balance sheet for the quarter.\n- Revenue because the company prematurely recognized revenue from future periods.\n- Accounts Receivable because the premature revenue recognition led to an overstatement of amounts owed by customers for sales that had not yet occurred.",
        "misstatements": [
            "Revenue",
            "Accounts Receivable"
        ],
        "misstating_company": {
            "name": "Company A",
            "role": "respondent",
            "cik": "0001234567"
        }
    },
    {
        "quarter": "2019q1",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "Company B, a subsidiary, improperly capitalized software development costs amounting to significant dollars that should have been expensed in the first quarter of 2019. This was done to reduce reported operating expenses and artificially boost the company's net income, while inflating assets on the balance sheet.\n- Capitalized Costs as Assets because expenses were improperly recorded as long-term assets.\n- Other Expense/Shareholder Equity Account because the improper capitalization reduced expenses, thereby inflating net income and retained earnings.",
        "misstatements": [
            "Capitalized Costs as Assets",
            "Other Expense/Shareholder Equity Account"
        ],
        "misstating_company": {
            "name": "Company B",
            "role": "involved company",
            "cik": "0007654321"
        }
    },
    {
        "quarter": "2020q1",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "Throughout fiscal year 2020, Company A's CFO instructed accounting staff to understate the company's allowance for bad debt and delay recording vendor invoices. Understating the allowance for bad debt inflated net accounts receivable and therefore income. Delaying vendor invoice recording understated accounts payable and current period expenses, artificially inflating earnings.\n- Allowance for Bad Debt because the estimated uncollectible accounts receivable were understated to inflate net receivables and income.\n- Payables because the recording of vendor invoices was delayed, understating current liabilities.",
        "misstatements": [
            "Allowance for Bad Debt",
            "Payables"
        ],
        "misstating_company": {
            "name": "Company A",
            "role": "respondent",
            "cik": "0001234567"
        }
    },
    {
        "quarter": "2020q2",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "Throughout fiscal year 2020, Company A's CFO instructed accounting staff to understate the company's allowance for bad debt and delay recording vendor invoices. Understating the allowance for bad debt inflated net accounts receivable and therefore income. Delaying vendor invoice recording understated accounts payable and current period expenses, artificially inflating earnings.\n- Allowance for Bad Debt because the estimated uncollectible accounts receivable were understated to inflate net receivables and income.\n- Payables because the recording of vendor invoices was delayed, understating current liabilities.",
        "misstatements": [
            "Allowance for Bad Debt",
            "Payables"
        ],
        "misstating_company": {
            "name": "Company A",
            "role": "respondent",
            "cik": "0001234567"
        }
    },
    {
        "quarter": "2020q3",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "Throughout fiscal year 2020, Company A's CFO instructed accounting staff to understate the company's allowance for bad debt and delay recording vendor invoices. Understating the allowance for bad debt inflated net accounts receivable and therefore income. Delaying vendor invoice recording understated accounts payable and current period expenses, artificially inflating earnings.\n- Allowance for Bad Debt because the estimated uncollectible accounts receivable were understated to inflate net receivables and income.\n- Payables because the recording of vendor invoices was delayed, understating current liabilities.",
        "misstatements": [
            "Allowance for Bad Debt",
            "Payables"
        ],
        "misstating_company": {
            "name": "Company A",
            "role": "respondent",
            "cik": "0001234567"
        }
    },
    {
        "quarter": "2020q4",
        "is_fiscal_quarter": true,
        "fraud_scheme_description": "Throughout fiscal year 2020, Company A's CFO instructed accounting staff to understate the company's allowance for bad debt and delay recording vendor invoices. Understating the allowance for bad debt inflated net accounts receivable and therefore income. Delaying vendor invoice recording understated accounts payable and current period expenses, artificially inflating earnings.\n- Allowance for Bad Debt because the estimated uncollectible accounts receivable were understated to inflate net receivables and income.\n- Payables because the recording of vendor invoices was delayed, understating current liabilities.",
        "misstatements": [
            "Allowance for Bad Debt",
            "Payables"
        ],
        "misstating_company": {
            "name": "Company A",
            "role": "respondent",
            "cik": "0001234567"
        }
    }
]
```
"""


def extract_aaer_quarters(aaer_data: List[dict]) -> List[dict]:
    """
    Process AAER data in parallel using all available VLLM clients.
    """

    def extract_single_aaer(aaer: dict, vllm_config: Dict[str, Any]) -> dict:
        """
        Extract quarter from a single AAER entry using a specific VLLM client.
        """

        vllm_config["client"] = openai.OpenAI(
            base_url=vllm_config["endpoint"], api_key="EMPTY"
        )
        # Extract quarter from the content
        urls = aaer["urls"]
        aaer_url = [e["url"] for e in urls if e["type"] == "primary"][0]

        aaer_no = aaer["aaerNo"]
        aaer_content = ""
        remaining_attempts = 2

        if aaer_url.lower().endswith(".pdf"):
            pdf_save_path = AAER_CONTENTS_DOWNLOADED_PATH / f"{aaer_no}.pdf"
            if not pdf_save_path.exists():
                print(f"Donwloading pdf : {aaer_no}.pdf")
                while remaining_attempts > 0:
                    pdf_path_temp = download_pdf_selenium(aaer_url)
                    if pdf_path_temp:
                        break
                    else:
                        print(
                            f"Failed to download PDF from {aaer_url}. Number of attempts left: {remaining_attempts}"
                        )
                        remaining_attempts -= 1
                        time.sleep(2)

                shutil.move(pdf_path_temp, pdf_save_path)

            aaer_content = extract_text_from_pdf(pdf_save_path)

        else:
            html_save_path = AAER_CONTENTS_DOWNLOADED_PATH / f"{aaer_no}.html"
            if not html_save_path.exists():
                while remaining_attempts > 0:
                    aaer_content = fetch_html_content(aaer_url)
                    if aaer_content:
                        break
                    else:
                        print(
                            f"Failed to fetch HTML content from {aaer_url}. Number of attempts left: {remaining_attempts}"
                        )
                        remaining_attempts -= 1
                        time.sleep(2)

                with open(html_save_path, "w") as f:
                    f.write(aaer_content)
            aaer_content = open(html_save_path).read()

        extracted_aaer_with_details = extract_aaer_quarter_from_content(
            aaer_data={
                "content": aaer_content,
                "entities": aaer.get("entities", []),
                "aaerNo": aaer["aaerNo"],
            },
            vllm_client=vllm_config["client"],
            vllm_model=vllm_config["model"],
        )

        original_keys_to_keep = [
            "respondents",
            "tags",
            "entities",
            "urls",
            "summary",
            "releaseNo",
            "complaints",
            "dateTime",
            "violatedSections",
        ]
        output = {
            "aaerNo": aaer_no,
            "details": [],
            **{k: v for k, v in aaer.items() if k in original_keys_to_keep},
        }

        for d in extracted_aaer_with_details:
            output["details"].append(
                {
                    "quarter": d["quarter"],
                    "is_fiscal_quarter": d.get("is_fiscal_quarter", False),
                    "misstatements": d["misstatements"],
                    "misstating_company": d["misstating_company"],
                    "fraud_scheme_description": d["fraud_scheme_description"],
                }
            )

        return output

    vllm_configs = load_vllm_configs(config_deduplication=20)

    if not vllm_configs:
        raise ValueError("No VLLM configurations available.")

    logging.info("Extracting AAER content...")
    aaer_with_content = Parallel(n_jobs=len(vllm_configs))(
        delayed(extract_single_aaer)(aaer, vllm_config)
        for aaer, vllm_config in tqdm(
            list(zip(aaer_data, cycle(vllm_configs))), desc="Extracting aaer details"
        )
    )
    logging.info("AAER content extracted...")

    return aaer_with_content


@on_exception(
    expo, ratelimit.RateLimitException, max_tries=10
)  # Exponential backoff for retry.
@ratelimit.limits(calls=1500, period=1)
def extract_aaer_quarter_from_content(
    aaer_data: dict,
    vllm_client,
    vllm_model,
) -> List[dict]:
    """
    Extract quarter from aaer
    """

    logging.info(f"Extracting quarters using {vllm_model} API...")
    prompt = f"""{json.dumps(aaer_data, indent=4)}"""

    # Initialize retry counter
    retry_count = 0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    while retry_count < MAX_RETRY:

        response = vllm_client.chat.completions.create(
            model=vllm_model,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            # top_logprobs=20,
            max_tokens=16000,
            timeout=1e5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        response_text = (
            response.choices[0]
            .message.content.split("```json")[1]
            .split("```")[0]
            .strip()
        )

        try:
            # Regex pattern to find unescaped newlines inside a JSON string
            pattern = r'("fraud_scheme_description"\s*:\s*")(.*?)(?<!\\)"'
            # This pattern greedily grabs up to the *closing quote* of the description.

            def fix_newlines(match):
                prefix = match.group(1)
                content = match.group(2)
                # Replace all raw newlines, tabs, carriage returns in the description with space
                fixed = re.sub(r"[\r\n\t]+", " ", content)
                return f'{prefix}{fixed}"'

            response_text = re.sub(
                pattern, fix_newlines, response_text, flags=re.DOTALL
            )

            formatted_output = json.loads(response_text)
            return [
                {
                    "quarter": d["quarter"],
                    "is_fiscal_quarter": d.get("is_fiscal_quarter", False),
                    "misstatements": d.get("misstatements", []),
                    "misstating_company": d.get("misstating_company", {}),
                    "fraud_scheme_description": d.get(
                        "fraud_scheme_description", False
                    ),
                    "aaerNo": aaer_data["aaerNo"],
                }
                for d in formatted_output
            ]

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Response Text: {response_text}")
            # Try a different API key on the next retry
            retry_count += 1
            if retry_count < MAX_RETRY:
                time.sleep(2**retry_count)  # Exponential backoff
            else:
                print(f"Failed after {MAX_RETRY} attempts")
                return {}  # Return empty dict if all retries fail

    return {}


def main():

    """
    Main function to demonstrate fetching AAER content.
    """

    example_url_pdf = "https://www.sec.gov/files/litigation/admin/2013/34-69945.pdf"
    example_url_html = "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-22121,securities"  # Example HTML URL

    # Test 2: Use Selenium to fetch PDF content
    pdf_path = download_pdf_selenium(example_url_pdf)
    if pdf_path:
        pdf_content = extract_text_from_pdf(pdf_path)
        cleanup_file(pdf_path)  # Delete the file after extraction
    else:
        pdf_content = ""

    # Test 3: Fetch content from HTML URL, rendered via selenium
    html_content = fetch_html_content(example_url_html)

    # Example usage with extract_aaer_fiscal_quarter (replace with your actual data)
    aaer_data = {"aaerNo": "1", "content": pdf_content if pdf_content else ""}

    with open("test_aaer_data.json", "w") as f:
        json.dump(aaer_data, f, indent=4)

    print("\n--- Extracting quarters Data ---")
    quarters_data = extract_aaer_quarter_from_content(aaer_data)
    print(json.dumps(quarters_data, indent=4))


if __name__ == "__main__":
    main()
