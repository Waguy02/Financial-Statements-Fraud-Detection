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
from typing import List, Tuple

import pdfplumber
import PyPDF2
import ratelimit
import requests
from backoff import expo, on_exception
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from google.oauth2 import service_account
from joblib import Parallel, delayed

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from researchpkg.anomaly_detection.config import (
    AAER_CONTENTS_DOWNLOADED_PATH,
    GEMINI_API_KEY,
    VERTEX_JSON_KEYS_PATH,
)

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
        print(f"✅ PDF Content extracted from {pdf_path}")
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
You are an AI agent designed to extract the fiscal or civil quarter(s) related to financial statement violations mentioned in Accounting and Auditing Enforcement Releases (AAERs).

### Task:
For each AAER entry, read the "content" and identify all the quarters (i.e., year + quarter) when financial violations (fraud, manipulation, misstatement, etc.) occurred. These quarters should be listed in the "quarter" key. Violations may span multiple quarters or even entire years.

*   **Quarter Format**: Quarters must be represented as `YYYYqQ` (e.g., "1990q1", "2021q3").
*   **Handling Full Years or Unspecified Quarters**: If the document indicates violations occurred throughout an entire year (e.g., "for the fiscal year 2020", "throughout 2019") or if a year is implicated without specific quarters, you must list all four quarters of that year (e.g., for 2020: "2020q1", "2020q2", "2020q3", "2020q4").
*   **Identifying Fiscal Quarters**: If the AAER states that the violations occurred in "fiscal periods" (e.g., "fiscal year", "fiscal quarter", or related to Forms 10-Q or Form-10k), set the `is_fiscal_quarter` boolean value to `true`. Otherwise, set it to `false`. 
*   **No Quarters Found**: If no specific quarters or years related to violations can be identified in the document, the "quarter" list must be empty.

### Input Format:
A list of AAER objects, each containing:
- "aaerNo": The unique identifier of the AAER.
- "content": The full text of the AAER.

### Output Format:
A JSON list of dicts, where each object corresponds to an input AAER entry:
- "aaerNo": The unique identifier of the AAER.
- "quarter": A list of unique extracted quarter strings (e.g., ["2018q1", "2018q2", "2019q3"]). This list should be empty (`[]`) if no quarters are found.
- "is_fiscal_quarter": A boolean value (`true` or `false`) indicating if the identified quarters were explicitly mentioned as fiscal periods in the AAER content.

### Example Output:
```json
[
    {
        "aaerNo": "1234",
        "quarter": ["2018q1", "2018q2", "2019q3"],
        "is_fiscal_quarter": true
    },
    {
        "aaerNo": "5678",
        "quarter": ["2020q4"],
        "is_fiscal_quarter": false
    },
    {
        "aaerNo": "9101",
        "quarter": [],
        "is_fiscal_quarter": false
    },
    {
        "aaerNo": "1122",
        "quarter": ["2021q1", "2021q2", "2021q3", "2021q4"],
        "is_fiscal_quarter": false
    }
]
"""


def extract_aaer_quarters(aaer_data: List[dict]) -> List[dict]:
    """
    From List of {"aaerNo":"XXX",
              "url":https://www.sec.gov/files/litigation/XXXX}
    retur  a list of  {
        "aaerNo": "1234",
        "quarters": ['2018q1', '2019q2'],
        "is_fiscal_quarter": true|false
    }
    """

    def extract_single_aaer(aaer: dict) -> dict:
        """
        Extract quarter from a single AAER entry.
        """
        # Extract quarter from the content
        urls = aaer["urls"]
        aaer_url = [e["url"] for e in urls if e["type"] == "primary"][0]

        aaer_no = aaer["aaerNo"]
        aaer_content = ""
        remaining_attempts = 2

        if aaer_url.lower().endswith(".pdf"):
            pdf_save_path = AAER_CONTENTS_DOWNLOADED_PATH / f"{aaer_no}.pdf"
            if not pdf_save_path.exists():
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

        return {"aaerNo": aaer_no, "content": aaer_content}

    # Do it with parallel processing
    print("Extracting AAER content...")
    aaer_with_content = Parallel(n_jobs=15)(
        delayed(extract_single_aaer)(aaer) for aaer in aaer_data
    )
    print("AAER content extracted...")

    aaer_with_quarters = {}
    split_size = 2
    for i in range(0, len(aaer_with_content), split_size):
        split = aaer_with_content[i : i + split_size]
        aaer_with_quarters.update(extract_aaer_quarter_from_content(split))
    print("Number of AAER :", len(aaer_with_quarters))
    print("Quarters  extracted...")

    return aaer_with_quarters


@on_exception(
    expo, ratelimit.RateLimitException, max_tries=10
)  # Exponential backoff for retry.
@ratelimit.limits(calls=1500, period=60)  # Limit of 15 Calls per minute
def extract_aaer_quarter_from_content(aaer_data: List[dict]) -> dict:
    """
    Extract quarter from aaer
    """
    # Get the next API key and service account info in rotation
    _, service_account_info, project_id, location = _get_next_api_key()

    # Create credentials from service account info
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Initialize Gemini client with credentials
    gemini_client = genai.Client(
        credentials=credentials,
        project=project_id,
        location=location,
        vertexai=True,
    )

    logging.info("Extracting quarters using Gemini API...")
    prompt = f"""
    {json.dumps(aaer_data, indent=4)}
    """

    # Initialize retry counter
    retry_count = 0
    while retry_count < MAX_RETRY:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.001,
                    max_output_tokens=8000,
                ),
            )

            try:
                formatted_output = json.loads(response.text)
                return {
                    d["aaerNo"]: (d["quarter"], d["is_fiscal_quarter"])
                    for d in formatted_output
                }
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print(f"Response Text: {response.text}")
                # Try a different API key on the next retry
                retry_count += 1
                if retry_count < MAX_RETRY:
                    print(
                        f"Retrying with a different API key. Attempt {retry_count+1}/{MAX_RETRY}"
                    )
                    _, service_account_info, project_id, location = _get_next_api_key()
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    gemini_client = genai.Client(
                        credentials=credentials,
                        project=project_id,
                        location=location,
                        vertexai=True,
                    )
                    time.sleep(2**retry_count)  # Exponential backoff
                else:
                    print(f"Failed after {MAX_RETRY} attempts")
                    return {}  # Return empty dict if all retries fail

        except Exception as e:
            print(f"Gemini API error: {e}")
            retry_count += 1
            if retry_count < MAX_RETRY:
                print(
                    f"Retrying with a different API key. Attempt {retry_count+1}/{MAX_RETRY}"
                )
                _, service_account_info, project_id, location = _get_next_api_key()
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                gemini_client = genai.Client(
                    credentials=credentials,
                    project=project_id,
                    location=location,
                    vertexai=True,
                )
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
    aaer_data = [
        {"aaerNo": "1", "content": pdf_content if pdf_content else ""},
        {"aaerNo": "2", "content": html_content if html_content else ""},
    ]
    with open("test_aaer_data.json", "w") as f:
        json.dump(aaer_data, f, indent=4)

    print("\n--- Extracting quarters Data ---")
    quarters_data = extract_aaer_quarter_from_content(aaer_data)
    print(json.dumps(quarters_data, indent=4))


if __name__ == "__main__":
    main()
