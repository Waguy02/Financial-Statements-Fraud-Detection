"""
Script to Summarize the content of the Management Discussiona and Analysis section of financial reports.

"""

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
from typing import List, Tuple

import dotenv
import pandas as pd
import ratelimit
from backoff import expo, on_exception
from google import genai
from google.genai import types

# Add Vertex AI imports
from google.oauth2 import service_account
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

from researchpkg.anomaly_detection.config import (
    VERTEX_JSON_KEYS_PATH,  # Add this to your config
)
from researchpkg.anomaly_detection.config import (
    GEMINI_JSON_KEYS_PATH,
    MDA_DATASET_PATH,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.models.utils import (
    load_cross_validation_path,
    load_full_data_path,
)
from researchpkg.anomaly_detection.preprocessing.utils import clean_mda_content
from researchpkg.utils import configure_logger

MAX_RPM_VERTEX = 1500
MAX_RPM_GEMINI = 15
MAX_SIMULATENOUS_REQUESTS_PER_KEY_VERTEX = 60
MAX_SIMULATENOUS_REQUESTS_PER_KEY_GEMINI = 5
MDA_SUMMARIZE_SYSTEM_PROMPT = """
You are a highly skilled financial analyst with deep expertise in evaluating corporate disclosures. 
You will be provided with the 'Management's Discussion and Analysis' (MD&A) section of a financial report for {quarter_info}.

Your task is to extract and present the **100 most important and distinct insights, observations, and factual statements** from the MD&A. Focus on the following categories:
– Strategic priorities and initiatives  
– Operational and segment performance  
– Financial results and key trends  
– Identified risks and uncertainties  
– Forward-looking statements and guidance  
– Significant changes, events, or developments impacting the business  
 
**Instructions:**
– Present the extracted content as a bulleted list using a dash (–) before each point  
– Be clear, specific, and concise in your wording  
– Avoid redundancy: each point should highlight a unique and meaningful piece of information  
– If a table is included in the MD&A:  
  – Write a bullet point with the table title (make sure to specify the unit/dimension if provided, e.g., "Table 1: Revenue by Segment (in millions)")
  – Recreate the table in clean markdown format below that point 

Do not include any introduction, conclusion, or extra commentary.
Don't use your own knowledge or any external knowledge to answer the question but simply summarize the content following the instructions.
Only output the finalized bullet list.
"""


SEC_MDA_OUTPUT_DIR = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED"
SEC_MDA_QUARTERLY_OUTPUT_DIR = SEC_MDA_OUTPUT_DIR / "quarterly"
TMP_OUTPUT_DIR = Path("/tmp/mda_summarized_temp")

SUMMARIZER_MODEL = "gemini-2.0-flash"
SUMMARIZER_SURROGATES = [
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
    SUMMARIZER_MODEL,
]
MAX_RETRIES = 5

MAX_NEW_TOKENS = 8000

# Constants
MDA_PATH = MDA_DATASET_PATH / "quarterly"


def __is_vertex(api_key_info):
    """
    Check if the API key info is for Vertex AI.
    """
    return "service_account" in api_key_info


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


def load_vertex_keys():
    """
    Load Vertex AI API keys from the JSON file.

    Returns:
        List[dict]: List of dicts with API key info including service account details
    """
    api_keys = []

    # Loading Vertex AI keys
    if VERTEX_JSON_KEYS_PATH.exists():
        logging.info("Loading Vertex API keys from JSON file")
        with open(VERTEX_JSON_KEYS_PATH, "r") as f:
            data = json.load(f)
            for entry in data:
                if not entry.get("disabled", False):
                    api_keys.append(entry)
                    logging.info(f"Loaded Vertex API key for email: {entry['email']}")

            # Loading Gemini keys
    if GEMINI_JSON_KEYS_PATH.exists():
        logging.info("Loading Gemini API keys from JSON file")
        with open(GEMINI_JSON_KEYS_PATH, "r") as f:
            data = json.load(f)
            for entry in data:
                if not entry.get("disabled", False):
                    api_keys.append(entry)
                    logging.info(f"Loaded Gemini API key for email: {entry['email']}")

    assert len(api_keys) > 0, "No valid API keys found in the JSON file."

    logging.info(f"Loaded {len(api_keys)} API keys from JSON file")

    return api_keys


def process_chunk_with_api_key(mda_quarter_ids_chunk, api_key_info):
    """
    Process a chunk of MDA files using a specific API key.

    Args:
        mda_quarter_ids_chunk (List[str]): A chunk of MDA quarter IDs to process
        api_key_info (dict): The API key info including service account details
    """
    # Set up process-specific logging
    process_id = mp.current_process().name
    # Extract email from process name if present
    email = api_key_info.get("email", "unknown")

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
        time.sleep(20)

    @on_exception(
        expo, ratelimit.RateLimitException, max_tries=10000, on_backoff=backoff_hdlr
    )
    @ratelimit.limits(
        calls=MAX_RPM_VERTEX if __is_vertex(api_key_info) else MAX_RPM_GEMINI, period=60
    )
    def call_llm_to_summarize(
        mda_content,
        temperature=0.01,
        model=SUMMARIZER_MODEL,
        quarter_info="Unknown Quarter",
    ):
        """
        Call the Gemini LLM via Vertex AI to summarize the MDA content.

        Args:
            mda_content (str): The MDA content to summarize

        Returns:
            str: The summarized MDA content
        """

        # Check if vertex AI
        if __is_vertex(api_key_info):
            # Get service account info from the api_key_info
            service_account_info = api_key_info["service_account"]
            project_id = api_key_info["project_id"]
            location = api_key_info["location"]

            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            # Initialize Gemini client with Vertex AI
            gemini_client = genai.Client(
                credentials=credentials,
                project=project_id,
                location=location,
                vertexai=True,
            )

        else:
            # not vertex
            api_key = api_key_info["key"]
            gemini_client = genai.Client(api_key=api_key)
        prompt = MDA_SUMMARIZE_SYSTEM_PROMPT.format(quarter_info=quarter_info)

        # Generate the response using Gemini via Vertex AI
        response = gemini_client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=prompt,
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
            contents=[mda_content],
        )

        # Extract the response text
        return response.text

    def process_single_mda_file(mda_quarter_id, num_retries=MAX_RETRIES):
        """
        Process a single MDA file and save the summarized content.
        """
        output_file = SEC_MDA_QUARTERLY_OUTPUT_DIR / f"{mda_quarter_id}.txt"
        if output_file.exists():
            logging.info(f"SKipping {output_file.name}")
            return
        is_retrying = num_retries < 5
        try:
            mda_content = load_mda_content(mda_quarter_id, clean_small_lines=False)

            temperature = 0.98

            # Call the LLM to summarize the MDA content
            model = (
                SUMMARIZER_MODEL
                if not is_retrying
                else SUMMARIZER_SURROGATES[MAX_RETRIES - num_retries - 1]
            )
            quarter_info = extract_quarter_info(mda_quarter_id)
            summary = call_llm_to_summarize(
                mda_content, temperature, model=model, quarter_info=quarter_info
            )

            # First write to a temporary file
            temp_file = process_tmp_dir / f"{mda_quarter_id}.txt"
            with open(temp_file, "w", encoding="utf-8") as file:
                file.write(summary)

            # Then move it to the final destination
            final_file = SEC_MDA_QUARTERLY_OUTPUT_DIR / f"{mda_quarter_id}.txt"
            shutil.move(str(temp_file), str(final_file))

            return True, mda_quarter_id
        except Exception as e:
            if num_retries > 0:
                print(
                    f"Retrying the extraction of {mda_quarter_id} - Attempt {5- num_retries+2} - {e} {email=}"
                )
                time.sleep(10)
                # Retry the process with a different model

                return process_single_mda_file(
                    mda_quarter_id, num_retries=num_retries - 1
                )

            logging.error(f"Error processing {mda_quarter_id}: {str(e)} - {email=}")
            return False, mda_quarter_id

    results = Parallel(
        n_jobs=MAX_SIMULATENOUS_REQUESTS_PER_KEY_VERTEX
        if __is_vertex(api_key_info)
        else MAX_SIMULATENOUS_REQUESTS_PER_KEY_GEMINI,
        prefer="threads",
    )(
        delayed(process_single_mda_file)(mda_id)
        for mda_id in tqdm(
            mda_quarter_ids_chunk,
            desc=f"Process {process_id}",
            unit="file",
            position=mp.current_process()._identity[0] % 10,
        )
    )

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
        api_keys (List[dict]): List of API key dictionaries with relevant Vertex AI info
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
            args=(chunk, api_key_dict),
            name=f"APIProcess-{i}-{api_key_dict['email']}",
        )
        processes.append(p)
        p.start()
        logging.info(f"Started process {p.name} for email: {api_key_dict['email']}")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    logging.info("All processes completed")


def extract_quarter_info(mda_quarter_id):
    """
    Extract quarter and year from the MDA quarter ID, e.g. '1000045_2010-Q1'.
    Returns a string like 'Q1 2010'.
    """
    try:
        parts = mda_quarter_id.split("_")
        if len(parts) > 1:
            date_part = parts[1]
            year, quarter = date_part.split("-")
            return f"{quarter} {year}"
    except:
        pass
    return "Unknown Quarter"


def main():
    """
    Main function to run the MDA summarization process.
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
    configure_logger(PREPROCESSED_PATH / "extended_v4_summarizing.log", logging.INFO)

    # Load API keys - now using Vertex AI keys
    api_keys = load_vertex_keys()
    logging.info(f"Loaded {len(api_keys)} API keys for distributed processing")

    # Load the data index
    logging.info("Loading data index")
    data_df = load_and_data_index()
    mda_quarter_ids = data_df["mda_quarter_id"].tolist()

    # Filter out already processed files
    processed_ids = [f.stem for f in SEC_MDA_QUARTERLY_OUTPUT_DIR.glob("*.txt")]
    mda_quarter_ids = [id for id in mda_quarter_ids if id not in processed_ids]

    logging.info(f"Found {len(mda_quarter_ids)} unprocessed MDA files")

    # Process the MDA files using distributed API keys
    process_mda_files(mda_quarter_ids, api_keys)


if __name__ == "__main__":
    main()
