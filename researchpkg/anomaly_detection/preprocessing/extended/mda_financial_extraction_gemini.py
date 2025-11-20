"""
Script to Extract financial information as JSON from the Management Discussion and Analysis section of financial reports.
"""

import json
import logging
import multiprocessing as mp
import random
import shutil
import time
from pathlib import Path
from typing import List

import pandas as pd
import ratelimit
from backoff import expo, on_exception
from joblib import Parallel, delayed
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    GEMINI_JSON_KEYS_PATH,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.models.utils import (
    load_cross_validation_path,
    load_full_data_path,
)
from researchpkg.anomaly_detection.preprocessing.utils import clean_mda_content
from researchpkg.utils import configure_logger

MAX_RPM_GEMINI = 10
MAX_SIMULATENOUS_REQUESTS_PER_KEY = 1  # Leave this at one.
MDA_SUMMARIZE_SYSTEM_PROMPT = """
You are a highly skilled financial analyst with deep expertise in evaluating corporate disclosures. You will be provided with the *Management's Discussion and Analysis* (MD&A) section of a financial report.

Your task is to extract **financial figures, metrics, and KPIs specifically for the CURRENT QUARTER** mentioned in the MD&A and return them in a **flat, clean JSON format**.

---

**Instructions:**

**1. First, identify the current quarter/period being discussed:**
   - Look for phrases like "for the quarter ended [date]", "for the three months ended [date]", etc.
   - This will be the focal point of your extraction

**2. Extract Financial Metrics for the CURRENT QUARTER only:**

Include numerical data from the MD&A that relates to the CURRENT QUARTER ONLY, such as:

- Revenue, net sales, gross sales (total and by segment, product, region, etc.)
- Costs and expenses (COGS, R&D, SG&A, etc.)
- Profit measures (gross profit, operating income, net income, etc.)
- Margins and ratios (e.g., gross margin, operating margin, EBITDA margin, ROE, etc.)
- Quarter-over-quarter (QoQ) comparisons and sequential changes
- Balance sheet metrics as of the current quarter end date
- Cash flow figures for the current quarter
- Segment-level and region-specific financial results for the current quarter
- Current quarter KPIs like ARPU, churn rate, bookings, billings, backlog, etc.

DO NOT include year-to-date figures or prior year comparisons unless they are specifically comparing to the current quarter.

**3. Formatting Requirements:**

- Return output as a **valid flat JSON object**
- Use **double quotes** (`"`) around keys and string values
- **DO NOT** include any explanation or commentary — JSON only

**4. Key Formatting Guidelines:**

- Keys must be **descriptive, human-readable, and concise**
  - Include **metric name** and **segment or region** if applicable
  - DO NOT include the quarter or year in the keys (e.g., "Q2 2023", "Current Quarter") as these are already known
  - Avoid redundancy: no repeating the same segment unnecessarily
  - Aim to keep keys **under 30 characters** where possible without losing clarity
  - Examples:
    - "Revenue Total" instead of "Revenue Total Current Q"
    - "Gross Margin Cloud" instead of "Gross Margin Cloud Q2 2023"
    - "Operating Income EMEA"
    - "Cash Flow Operations"

**5. Value Formatting Rules:**

- Always **convert units**:
  - Convert billions/millions/thousands into actual numbers using commas (e.g., "$1.2 billion" → "$1,200,000,000")
- **Currency values** must:
  - Start with the **currency symbol** (e.g., "$", "€")
  - Include comma separators
  - Be treated as **negative** if reported in parentheses — e.g., `($200 million)` → "-$200,000,000"
- **Percentages** must:
  - Include the percent sign (`%`)
  - Reflect negative changes with a minus (e.g., -3.5%)
- If a value is **non-numeric or "N/A"** or is zero, omit it

**6. Deduplication:**

- Ensure there are **no duplicate keys or repeated values**
- Prefer the **most complete or precise form** if the same data is referenced multiple times

---

**Example Output:**

```json
{
  "Revenue Total": "$5,200,000,000",
  "Revenue Growth QoQ": "7.5%",
  "Net Income": "$920,000,000",
  "Operating Margin": "18.2%",
  "Revenue Cloud": "$2,100,000,000",
  "Revenue Software": "$1,800,000,000",
  "Revenue Hardware": "$1,300,000,000",
  "Operating Expenses": "$1,250,000,000",
  "Cash On Hand": "$10,400,000,000"
}
"""

SEC_MDA_OUTPUT_DIR = PREPROCESSED_PATH / "SEC_MDA_FINANCIAL_JSON"
SEC_MDA_QUARTERLY_OUTPUT_DIR = SEC_MDA_OUTPUT_DIR / "quarterly"
TMP_OUTPUT_DIR = Path("/tmp/mda_financial_json_temp")


SUMMARIZER_MODEL = "gemini-2.0-flash"
# SUMMARIZER_MODEL = "gemini-2.0-flash-exp"
SUMMARIZER_SURROGATES = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    # "gemini-1.5-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
]

# SUMMARIZER_MODEL_SURROGATE = "gemini-2.0-flash-exp"


# SUMMARIZER_MODEL = "gemini-1.5-flash"
# SUMMARIZER_MODEL_SURROGATE = "gemini-2.0-flash-lite"


# SUMMARIZER_MODEL = "gemini-2.0-flash-exp"
# SUMMARIZER_MODEL_SURROGATE =  "gemini-2.0-flash"


MAX_NEW_TOKENS = 8000

# Constants
MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"


def initialize_dirs():
    """
    Create the output directory if it doesn't exist.
    """
    SEC_MDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEC_MDA_QUARTERLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mda_content(mda_quarter_id, clean_small_lines=False):
    """
    Load the content of a MDA FIle.
    """
    mda_file = MDA_PATH / f"{mda_quarter_id}.txt"

    if not mda_file.exists():
        raise FileNotFoundError(f"MDA file {mda_file} does not exist.")

    with open(mda_file, "r", encoding="utf-8") as file:
        mda_content = file.read()
        # Clean the MDA content
        mda_content = clean_mda_content(
            mda_content, clean_small_lines=clean_small_lines
        )

        return mda_content


def load_and_data_index():
    """
    Load the dataset index corresponding to the V4 configuration (ie. 95%, 5% split )
    """
    full_df = pd.read_csv(load_full_data_path({"dataset_version": "company_isolated_splitting"}))

    return full_df[["mda_quarter_id"]]


def extract_quarter_info(mda_quarter_id):
    """
    Extract quarter information from the MDA quarter ID.
    Format is typically 'CIK_YYYY-QQ' e.g. '1000045_2010-Q1'

    Args:
        mda_quarter_id (str): The ID of the MDA quarter

    Returns:
        str: Quarter information in readable format
    """
    try:
        # Split by underscore to get the date part
        parts = mda_quarter_id.split("_")
        if len(parts) > 1:
            # Extract year and quarter
            date_part = parts[1]
            year, quarter = date_part.split("-")
            return f"{quarter} {year}"
        return "Unknown Quarter"
    except Exception:
        return "Unknown Quarter"


def process_chunk_with_api_key(mda_quarter_ids_chunk, api_key):
    """
    Process a chunk of MDA files using a specific API key.

    Args:
        mda_quarter_ids_chunk (List[str]): A chunk of MDA quarter IDs to process
        api_key (str): The API key to use for this chunk
    """
    # Set up process-specific logging
    process_id = mp.current_process().name
    # Extract email from process name if present
    email = process_id.split("-", 2)[2] if len(process_id.split("-")) > 2 else "unknown"

    logging.info(
        f"Process {process_id} starting with {len(mda_quarter_ids_chunk)} files using key for email: {email}"
    )

    # Create process-specific temporary directory
    process_tmp_dir = TMP_OUTPUT_DIR / f"process_{process_id}"
    process_tmp_dir.mkdir(parents=True, exist_ok=True)

    def backoff_hdlr(details):
        """Handler function to force a 10 second wait between retries"""
        print(
            f"Backing off for 10 seconds after {details['tries']} tries. Exception: {details['exception']} {email=}"
        )
        time.sleep(10)

    @on_exception(
        expo, ratelimit.RateLimitException, max_tries=1000, on_backoff=backoff_hdlr
    )
    @ratelimit.limits(calls=MAX_RPM_GEMINI, period=60)
    def call_llm_to_summarize(
        mda_content, quarter_info, temperature=0.01, model=SUMMARIZER_MODEL
    ):
        """
        Call the Gemini LLM to extract financial information as JSON from MDA content.

        Args:
            mda_content (str): The MDA content to process
            quarter_info (str): Information about which quarter the MDA relates to
            temperature (float): Temperature setting for generation
            model (str): Model identifier

        Returns:
            str: The JSON string containing financial information
        """
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=api_key)

        # Prepend quarter information to the content
        content_with_quarter = f"REPORT FOR: {quarter_info}\n\n{mda_content}"

        # Generate the response using Gemini
        response = gemini_client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=MDA_SUMMARIZE_SYSTEM_PROMPT,
                temperature=temperature,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                max_output_tokens=MAX_NEW_TOKENS - 10,
            ),
            contents=[content_with_quarter],
        )

        # Extract the response text
        json_response = response.text

        json_response = json_response.replace("```json", "").replace("```", "").strip()

        # Validate JSON
        try:
            # Parse and re-format to ensure valid JSON
            parsed_json = json.loads(json_response)
            return json.dumps(parsed_json, indent=2)
        except json.JSONDecodeError as e:
            # Try to fix the json
            fixed_json = fix_json_string(json_response)
            return fixed_json

    MAX_RETRIES = 7

    def process_single_mda_file(mda_quarter_id, num_retries=MAX_RETRIES):
        """
        Process a single MDA file and save the extracted financial information as JSON.
        """
        output_file = SEC_MDA_QUARTERLY_OUTPUT_DIR / f"{mda_quarter_id}.json"
        if output_file.exists():
            logging.info(f"Skipping {output_file.name}")
            return
        is_retrying = num_retries < MAX_RETRIES
        try:
            mda_content = load_mda_content(mda_quarter_id, clean_small_lines=False)
            quarter_info = extract_quarter_info(mda_quarter_id)

            temperature = 0.01 if not is_retrying else 0.98

            # Call the LLM to extract financial information as JSON from the MDA content
            model = (
                SUMMARIZER_MODEL
                if not is_retrying
                else SUMMARIZER_SURROGATES[MAX_RETRIES - num_retries - 1]
            )
            financial_json = call_llm_to_summarize(
                mda_content, quarter_info, temperature, model=model
            )

            # First write to a temporary file
            temp_file = process_tmp_dir / f"{mda_quarter_id}.json"
            with open(temp_file, "w", encoding="utf-8") as file:
                file.write(financial_json)

            # Then move it to the final destination
            final_file = SEC_MDA_QUARTERLY_OUTPUT_DIR / f"{mda_quarter_id}.json"
            shutil.move(str(temp_file), str(final_file))

            return True, mda_quarter_id
        except Exception as e:
            if num_retries > 0:
                print(
                    f"Retrying the extraction of {mda_quarter_id} - Attempt {MAX_RETRIES- num_retries+2} - {e}"
                )
                time.sleep(5)

                return process_single_mda_file(
                    mda_quarter_id, num_retries=num_retries - 1
                )

            logging.error(f"Error processing {mda_quarter_id}: {str(e)} - {email=}")
            return False, mda_quarter_id

    # results = Parallel(n_jobs=MAX_SIMULATENOUS_REQUESTS_PER_KEY, prefer="threads")(
    #     delayed(process_single_mda_file)(mda_id)
    #     for mda_id in tqdm(
    #         mda_quarter_ids_chunk,
    #         desc=f"Process {process_id}",
    #         unit="file",
    #         position=mp.current_process()._identity[0] % 10,
    #     )
    # )
    # not using parallel processing to avoid issues with the LLM API
    results = []
    for mda_id in tqdm(
        mda_quarter_ids_chunk,
        desc=f"Process {process_id}",
        unit="file",
        position=mp.current_process()._identity[0] % 10,
    ):
        result = process_single_mda_file(mda_id)
        results.append(result)
    # Log the results
    logging.info(f"Process {process_id} completed with {len(results)} results")

    # Count successes and failures
    successes = sum(1 for result, _ in results if result)
    failures = sum(1 for result, _ in results if not result)

    logging.info(f"Process {process_id}: {successes} successes, {failures} failures")

    # Clean up temporary directory
    try:
        shutil.rmtree(str(process_tmp_dir))
    except Exception as e:
        logging.warning(
            f"Failed to clean up temporary directory {process_tmp_dir}: {str(e)}"
        )

    # Return results for consolidated reporting
    return results


def process_mda_files(mda_quarter_ids: List[str], api_keys: List[dict]):
    """
    Distribute MDA files across multiple processes, each with its own API key.

    Args:
        mda_quarter_ids (List[str]): List of MDA quarter IDs to process
        api_keys (List[dict]): List of API key dictionaries with 'key' and 'email' fields
    """
    # Shuffle the MDA quarter IDs to ensure even distribution of work
    random.shuffle(mda_quarter_ids)

    # Calculate the number of files per API key
    num_keys = len(api_keys)
    files_per_key = len(mda_quarter_ids) // num_keys
    remainder = len(mda_quarter_ids) % num_keys

    # Split the data into chunks, one for each API key
    chunks = []
    start_idx = 0
    for i in range(num_keys):
        # Add one extra file to some chunks if we have a remainder
        chunk_size = files_per_key + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size

        chunks.append(mda_quarter_ids[start_idx:end_idx])
        start_idx = end_idx

    logging.info(f"Divided {len(mda_quarter_ids)} files into {num_keys} chunks")

    # Create and start processes
    processes = []
    for i, (chunk, api_key_dict) in enumerate(zip(chunks, api_keys)):
        if not chunk:  # Skip empty chunks
            continue

        p = mp.Process(
            target=process_chunk_with_api_key,
            args=(chunk, api_key_dict["key"]),
            name=f"APIProcess-{i}-{api_key_dict['email']}",
        )
        processes.append(p)
        p.start()
        logging.info(f"Started process {p.name} for email: {api_key_dict['email']}")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    logging.info("All processes completed")


def load_keys():
    assert (
        GEMINI_JSON_KEYS_PATH.exists()
    ), "Gemin API Keys file not found, make sure to create that file with gemini keys, to run this script."

    with open(GEMINI_JSON_KEYS_PATH, "r") as f:
        data = json.load(f)
    # Extract both keys and emails
    api_keys = []
    for entry in data:
        if not entry.get("disabled", False):
            key = entry["key"]
            email = entry.get("email", "no email provided")
            api_keys.append({"key": key, "email": email})
            logging.info(f"Loaded API key for email: {email}")

    return api_keys


def fix_json_string(json_str):
    """
    Attempts to fix JSON parsing errors, specifically focusing on incomplete JSON strings.

    Args:
        json_str (str): The potentially invalid JSON string

    Returns:
        str: A fixed JSON string or the original string if unfixable
    """
    import re

    # Remove any markdown code block markers if present
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    # Try standard parsing first
    try:
        # Check if the string is already valid JSON
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        # Continue with fixes
        pass

    # Check if the JSON is missing closing bracket
    if not json_str.endswith("}"):
        # Split by lines to handle the last incomplete entry
        lines = json_str.split("\n")

        # Find the last line that contains a complete key-value pair (has a comma)
        valid_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep lines that are complete (end with comma) or look like proper json elements
            if stripped and (
                stripped.endswith(",") or stripped.endswith("}") or "}" not in stripped
            ):
                valid_lines.append(line)
            elif '"' in stripped and ":" in stripped:
                # This might be the last line without a comma
                # Only keep it if it appears to be a properly formatted key-value pair
                if stripped.count('"') >= 2:
                    valid_lines.append(line)

        # If the last line doesn't end with a comma, it might be incomplete
        if (
            valid_lines
            and not valid_lines[-1].strip().endswith(",")
            and not valid_lines[-1].strip().endswith("}")
        ):
            # Remove the last potentially incomplete entry
            valid_lines = valid_lines[:-1]

        # Reconstruct the JSON string
        fixed_str = "\n".join(valid_lines)

        # Ensure it has opening brace
        if not fixed_str.strip().startswith("{"):
            fixed_str = "{" + fixed_str

        # Make sure we don't have a trailing comma before adding closing brace
        fixed_str = re.sub(r",\s*$", "", fixed_str)

        # Add closing brace
        fixed_str = fixed_str + "}"

        json_str = fixed_str

    # Fix issues with quotes around keys and values
    json_str = re.sub(r"([a-zA-Z0-9_\s]+):", r'"\1":', json_str)  # Unquoted keys
    json_str = re.sub(r"'([^']+)':", r'"\1":', json_str)  # Single-quoted keys
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Single-quoted values

    # Remove trailing commas before closing braces or brackets
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # Try to parse the fixed JSON
    try:
        parsed_json = json.loads(json_str)
        logging.info("Fixed JSON successfully")
        return json.dumps(parsed_json, indent=2)
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to fix JSON: {e}")
        # Last resort approach - just create a valid but possibly incomplete JSON
        try:
            # Find all key-value pairs using regex
            pattern = r'"([^"]+)"\s*:\s*"([^"]+)"|"([^"]+)"\s*:\s*(\d+(?:\.\d+)?)'
            matches = re.findall(pattern, json_str)

            result_dict = {}
            for match in matches:
                # Each match is a tuple with either (key, string_value, '', '') or ('', '', key, numeric_value)
                if match[0]:
                    result_dict[match[0]] = match[1]
                elif match[2]:
                    # Try to convert to number if it's a numeric value
                    try:
                        result_dict[match[2]] = (
                            float(match[3]) if "." in match[3] else int(match[3])
                        )
                    except ValueError:
                        result_dict[match[2]] = match[3]

            if result_dict:
                return json.dumps(result_dict, indent=2)
        except Exception as e2:
            logging.warning(f"Last resort JSON fix also failed: {e2}")

        return "{}"  # Return empty JSON as a last resort


def main():
    """
    Main function to run the MDA financial information extraction process.
    """
    # Use multiprocessing method that works with both spawn and fork
    mp.set_start_method("spawn", force=True)

    # Initialize directories
    initialize_dirs()

    # Clean up any leftover temporary files from previous runs
    if TMP_OUTPUT_DIR.exists():
        try:
            shutil.rmtree(str(TMP_OUTPUT_DIR))
            logging.info(f"Cleaned up old temporary directory: {TMP_OUTPUT_DIR}")
        except Exception as e:
            logging.warning(f"Failed to clean up old temporary directory: {str(e)}")

    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure logging
    configure_logger(
        PREPROCESSED_PATH / "extended_v4_financial_extraction.log", logging.INFO
    )

    # Load API keys
    api_keys = load_keys()
    logging.info(f"Loaded {len(api_keys)} API keys for distributed processing")

    # Load the data index
    logging.info("Loading data index")
    data_df = load_and_data_index()
    mda_quarter_ids = data_df["mda_quarter_id"].tolist()

    # Filter out already processed files
    processed_ids = [f.stem for f in SEC_MDA_QUARTERLY_OUTPUT_DIR.glob("*.json")]
    mda_quarter_ids = [id for id in mda_quarter_ids if id not in processed_ids]

    logging.info(f"Found {len(mda_quarter_ids)} unprocessed MDA files")

    # Process the MDA files using distributed API keys
    process_mda_files(mda_quarter_ids, api_keys)


if __name__ == "__main__":
    main()
