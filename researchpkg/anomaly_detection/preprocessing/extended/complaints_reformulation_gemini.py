"""
Script to reformulate complaint texts from SEC AAERs into more concise and factual forms
for downstream classification tasks.
"""

import fcntl
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import tempfile
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import dotenv
import pandas as pd
import ratelimit
from backoff import expo, on_exception
from google import genai
from google.genai import types
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

from researchpkg.anomaly_detection.config import (
    GEMINI_JSON_KEYS_PATH,
    PREPROCESSED_PATH,
    PREPROCESSED_PATH_EXTENDED,
)
from researchpkg.anomaly_detection.models.utils import load_train_test_path
from researchpkg.anomaly_detection.preprocessing.utils import clean_mda_content
from researchpkg.utils import configure_logger

MAX_RPM_GEMINI = 15
MAX_SIMULATENOUS_REQUESTS_PER_KEY = 4
MDA_SUMMARIZE_SYSTEM_PROMPT = """
You are a highly skilled financial analyst with expertise in regulatory violations and corporate misconduct.

Task: Reformulate the given SEC complaints into concise, factual bullet points suitable for classification.

Guidelines:
- Focus on the specific financial violations, accounting improprieties, or regulatory breaches
- Extract the key facts without extraneous details
- Use clear, professional financial terminology
- Each point should begin with a dash (-)
- Maintain the essential meaning while being concise
- If violations of laws sections/rules, don't mention the sections concerned, we just need the factual complaints regarding financials facts;

Example:

Input complaint:
"Emma engaged in a scheme resulting in the improper accounting for certain intercompany transactions involving foreign currency fluctuations, causing Hill to materially overstate its net income."

Output:
- Improper intercompany accounting and foreign currency misstatement
- Net income overstatement

Only output the reformulated bullet points without any additional text, explanations or comments.
"""

SUMMARIZER_MODEL = "gemini-2.0-flash-exp"

SUMMARIZER_SURROGATES = [
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
]
MAX_RETRIES = 5

MAX_NEW_TOKENS = 8000

# Constants
INDEX = PREPROCESSED_PATH_EXTENDED / "v4/all_index.csv"
INDEX_UPDATED = (
    PREPROCESSED_PATH_EXTENDED
    / "v4_unbalanced_cik_unbiased_index_with_reformulated_complaints.csv"
)

# Create temporary directory for processing
TMP_OUTPUT_DIR = Path(tempfile.gettempdir()) / "complaint_reformulation_tmp"


def initialize_dirs():
    """
    Initialize the necessary directories for processing.
    """
    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure the parent directory of the output file exists
    INDEX_UPDATED.parent.mkdir(parents=True, exist_ok=True)


def load_and_data_index():
    """
    Load the dataset index corresponding to the V4 configuration (ie. 95%, 5% split)
    """
    index_file = INDEX
    df = pd.read_csv(index_file)

    # Initialize the output file if it doesn't exist
    initialize_output_file(df)

    # Check what's already been processed
    already_processed = get_already_processed_aaers()

    # Filter the complaints dictionary to only include unprocessed ones
    aaer_complaints_dict = (
        df.dropna(subset="aaer_no").set_index("aaer_no")["complaints"].to_dict()
    )
    unprocessed_complaints = {
        aaer: complaint
        for aaer, complaint in aaer_complaints_dict.items()
        if str(aaer) not in already_processed
    }

    logging.info(
        f"Found {len(unprocessed_complaints)} unprocessed complaints out of {len(aaer_complaints_dict)} total"
    )

    return unprocessed_complaints, df


def initialize_output_file(original_df):
    """
    Initialize the output file if it doesn't exist, with a reformulated_complaints column set to None

    Args:
        original_df: Original DataFrame from the index file
    """
    if not INDEX_UPDATED.exists():
        df = original_df.copy()
        df["reformulated_complaints"] = None
        df.to_csv(INDEX_UPDATED, index=False)
        logging.info(f"Initialized output file at {INDEX_UPDATED}")


def get_already_processed_aaers() -> Set[str]:
    """
    Get the set of AAER numbers that have already been processed

    Returns:
        Set of AAER numbers (as strings) that already have reformulated complaints
    """
    if not INDEX_UPDATED.exists():
        return set()

    try:
        df = pd.read_csv(INDEX_UPDATED)
        # Get AAERs with non-null reformulated complaints
        df = df.dropna(subset="aaer_no")
        processed = set(
            df[df["reformulated_complaints"].notna()]["aaer_no"].astype(str).tolist()
        )
        return processed
    except Exception as e:
        logging.warning(f"Error reading already processed AAERs: {e}")
        return set()


def process_complaints(
    complaints_dict: Dict[str, str], api_keys: List[dict]
) -> Dict[str, str]:
    """
    Process complaints and generate reformulated versions.

    Args:
        complaints_dict: Dictionary of AAER numbers to complaint text
        api_keys: List of API key dictionaries

    Returns:
        Dictionary mapping AAER numbers to reformulated complaints
    """
    # Distribute complaints evenly among available API keys
    aaer_nos = list(complaints_dict.keys())
    random.shuffle(aaer_nos)

    chunks = []
    num_keys = len(api_keys)
    complaints_per_key = len(aaer_nos) // num_keys
    remainder = len(aaer_nos) % num_keys

    start_idx = 0
    for i in range(num_keys):
        chunk_size = complaints_per_key + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size

        aaer_chunk = aaer_nos[start_idx:end_idx]
        complaints_chunk = {aaer: complaints_dict[aaer] for aaer in aaer_chunk}
        chunks.append((complaints_chunk, api_keys[i]["key"], api_keys[i]["email"]))
        start_idx = end_idx

    # Process chunks in parallel
    with mp.Pool(processes=num_keys) as pool:
        results = pool.starmap(process_complaint_chunk, chunks)

    # Combine results
    reformulated_complaints = {}
    for result in results:
        reformulated_complaints.update(result)

    return reformulated_complaints


def process_complaint_chunk(
    complaints_chunk: Dict[str, str], api_key: str, email: str
) -> Dict[str, str]:
    """
    Process a chunk of complaints using a specific API key and save results progressively.

    Args:
        complaints_chunk: Dictionary mapping AAER numbers to complaint text
        api_key: API key to use
        email: Email associated with the API key

    Returns:
        Dictionary mapping AAER numbers to reformulated complaints
    """
    logging.info(
        f"Processing {len(complaints_chunk)} complaints with key for email: {email}"
    )

    # Set up process-specific temporary directory
    process_tmp_dir = TMP_OUTPUT_DIR / f"process_{email.split('@')[0]}"
    process_tmp_dir.mkdir(parents=True, exist_ok=True)

    def backoff_hdlr(details):
        """Handler function to force a wait between retries"""
        print(
            f"Backing off for 10 seconds after {details['tries']} tries. Exception: {details['exception']} {email=}"
        )
        time.sleep(20)

    @on_exception(
        expo, ratelimit.RateLimitException, max_tries=1000, on_backoff=backoff_hdlr
    )
    @ratelimit.limits(calls=MAX_RPM_GEMINI, period=60)
    def reformulate_complaint(complaint_text, temperature=0.01, model=SUMMARIZER_MODEL):
        """
        Call the Gemini LLM to reformulate the complaint.

        Args:
            complaint_text: The complaint text to reformulate

        Returns:
            str: The reformulated complaint
        """
        gemini_client = genai.Client(api_key=api_key)

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
            contents=[complaint_text],
        )

        # Extract the response text
        return response.text

    def process_single_complaint(aaer_no, complaint_text, num_retries=MAX_RETRIES):
        """
        Process a single complaint, save it to the index file, and return the reformulated text.
        """
        is_retrying = num_retries < MAX_RETRIES
        try:
            temperature = 0.01 if not is_retrying else 0.98

            # Call the LLM to reformulate the complaint
            model = (
                SUMMARIZER_MODEL
                if not is_retrying
                else SUMMARIZER_SURROGATES[MAX_RETRIES - num_retries - 1]
            )
            reformulated = reformulate_complaint(
                complaint_text, temperature, model=model
            )

            # Save to temporary file as backup
            temp_file = process_tmp_dir / f"{aaer_no}.txt"
            with open(temp_file, "w", encoding="utf-8") as file:
                file.write(reformulated)

            # Update the index file with this single result
            update_single_result(aaer_no, reformulated)

            return True, aaer_no, reformulated
        except Exception as e:
            if num_retries > 0:
                print(
                    f"Retrying the reformulation of AAER {aaer_no} - Attempt {MAX_RETRIES - num_retries + 2} - {e}"
                )
                return process_single_complaint(
                    aaer_no, complaint_text, num_retries=num_retries - 1
                )

            logging.error(f"Error processing AAER {aaer_no}: {str(e)} - {email=}")
            return False, aaer_no, ""

    results = {}
    for aaer_no, complaint_text in tqdm(
        complaints_chunk.items(), desc=f"Process {email}", position=0
    ):
        success, aaer, reformulated = process_single_complaint(aaer_no, complaint_text)
        if success:
            results[aaer] = reformulated

    # Clean up temporary directory
    try:
        shutil.rmtree(str(process_tmp_dir))
    except Exception as e:
        logging.warning(
            f"Failed to clean up temporary directory {process_tmp_dir}: {str(e)}"
        )

    return results


def update_single_result(aaer_no, reformulated_text):
    """
    Update a single result in the index file using file locking to avoid conflicts.

    Args:
        aaer_no: AAER number to update
        reformulated_text: Reformulated complaint text
    """
    max_retries = 5
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            # Open the file in read-write mode
            with open(INDEX_UPDATED, "r+") as f:
                # Get an exclusive lock on the file
                fcntl.flock(f, fcntl.LOCK_EX)

                try:
                    # Read the current data
                    df = pd.read_csv(f)

                    # Update the specific row
                    df.loc[
                        df["aaer_no"].astype(str) == str(aaer_no),
                        "reformulated_complaints",
                    ] = reformulated_text

                    # Go back to the beginning of the file and truncate it
                    f.seek(0)
                    f.truncate()

                    # Write the updated dataframe
                    df.to_csv(f, index=False)

                    # Flush to ensure data is written
                    f.flush()
                finally:
                    # Release the lock
                    fcntl.flock(f, fcntl.LOCK_UN)

                return

        except Exception as e:
            logging.warning(
                f"Attempt {attempt+1}/{max_retries} to update AAER {aaer_no} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logging.error(
                    f"Failed to update AAER {aaer_no} after {max_retries} attempts"
                )
                raise


def update_index_with_reformulated_complaints(
    original_df: pd.DataFrame, reformulated_complaints: Dict[str, str]
) -> pd.DataFrame:
    """
    Update the index DataFrame with reformulated complaints.
    This is a final verification to ensure all complaints were properly saved.

    Args:
        original_df: Original index DataFrame
        reformulated_complaints: Dictionary mapping AAER numbers to reformulated complaints

    Returns:
        Updated DataFrame with reformulated complaints
    """
    # This function is now mainly for verification since updates happen progressively
    # Read the current state of the output file
    try:
        df = pd.read_csv(INDEX_UPDATED)

        # Check for any missing reformulated complaints that should have been saved
        for aaer_no, reformulated in reformulated_complaints.items():
            if (
                df.loc[
                    df["aaer_no"].astype(str) == str(aaer_no), "reformulated_complaints"
                ]
                .isnull()
                .any()
            ):
                logging.warning(
                    f"AAER {aaer_no} was processed but not properly saved, updating now"
                )
                update_single_result(aaer_no, reformulated)

        # Read again after possible updates
        df = pd.read_csv(INDEX_UPDATED)
        logging.info(
            f"Verified all {len(reformulated_complaints)} reformulated complaints are in the index file"
        )

        return df
    except Exception as e:
        logging.error(f"Error during final verification: {e}")
        # Fall back to the old method if something went wrong
        df = original_df.copy()
        df["aaer_no"] = df["aaer_no"].astype(str)
        df["reformulated_complaints"] = df["aaer_no"].map(reformulated_complaints)
        df.to_csv(INDEX_UPDATED, index=False)
        logging.info(f"Updated index saved to {INDEX_UPDATED} using fallback method")
        return df


def load_keys():
    assert (
        GEMINI_JSON_KEYS_PATH.exists()
    ), "Gemini API Keys file not found, make sure to create that file with gemini keys, to run this script."

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


def main():
    """
    Main function to run the complaints reformulation process.
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
        PREPROCESSED_PATH / "extended_complaints_reformulation.log", logging.INFO
    )

    # Load API keys
    api_keys = load_keys()
    logging.info(f"Loaded {len(api_keys)} API keys for distributed processing")

    # Load the data index and complaints
    logging.info("Loading data index and complaints")
    unprocessed_complaints, original_df = load_and_data_index()

    if not unprocessed_complaints:
        logging.info("All complaints have already been processed. Nothing to do.")
        return

    logging.info(f"Found {len(unprocessed_complaints)} complaints to process")

    # Process complaints to get reformulated versions
    reformulated_complaints = process_complaints(unprocessed_complaints, api_keys)
    logging.info(f"Successfully reformulated {len(reformulated_complaints)} complaints")

    # Final verification of the index with reformulated complaints
    updated_df = update_index_with_reformulated_complaints(
        original_df, reformulated_complaints
    )
    logging.info("Index verification completed successfully")


if __name__ == "__main__":
    main()
