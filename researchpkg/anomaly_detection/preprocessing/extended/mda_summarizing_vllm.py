"""
Script to Summarize the content of the Management Discussion and Analysis section of financial reports using multiple
OpenAI Chat Completions API endpoints hosted on VLLM.
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
from typing import Any, Dict, List, Tuple

import dotenv

# Import OpenAI library
import openai
import pandas as pd
import ratelimit
from backoff import expo, on_exception
from joblib import Parallel, delayed
from tqdm import tqdm

# Assuming these config paths are correctly defined for your VLLM setup
from researchpkg.anomaly_detection.config import (
    MDA_DATASET_PATH,
    PREPROCESSED_PATH,
    VLLM_ENDPOINTS_CONFIG_PATH,
)
from researchpkg.anomaly_detection.models.utils import load_full_data_path
from researchpkg.anomaly_detection.preprocessing.utils import (
    clean_mda_content,
    load_vllm_configs,
)
from researchpkg.utils import configure_logger

# --- Rate Limit and Concurrency (Adjust these based on your VLLM setup's capacity) ---
# These limits are applied per *process*, so ensure they are safe for a single endpoint.
MAX_RPM_VLLM = (
    10000  # Example: 10000 requests per minute per endpoint. Adjust as needed.
)
MAX_SIMULATENOUS_REQUESTS_PER_PROCESS = 40
MAX_RETRIES = 3

MDA_SUMMARIZE_SYSTEM_PROMPT = """
You are a highly skilled financial analyst with deep expertise in summarizing corporate disclosures.
You will be provided with the 'Management's Discussion and Analysis' (MD&A) section of a financial report for {quarter_info}.
Your task is to summarize it following the instructions below:

Extract and present the **distinct, and factual insights**, along with subjective statements, management commentary, and qualitative explanations, organized into the following sections:

---

** 1. Strategic Priorities and Initiatives**
   – Summarize key strategies, corporate objectives, growth plans, restructuring efforts, and major initiatives discussed by management.
   – Capture significant strategic shifts, operational transformations, ambitious targets, or business model changes.
   – Highlight subjective language, including optimistic tone, vague descriptions of progress, or assertions lacking clear supporting evidence.

** 2. Operational and Segment Performance**
   – Summarize operational results and segment-level performance, including production metrics, KPIs, challenges, and improvements.
   – Pay special attention to:
     – Unexplained variances in performance.
     – Misalignment between narrative explanations and operational metrics.
     – Subjective, vague, or generic explanations (e.g., “seasonality,” “market dynamics,” “operational excellence”) without adequate quantification.
     – Unusual operational trends, sales fluctuations, production shifts, or inventory movements.

** 3. Financial Results and Key Trends**
   – Capture **all financial metrics**, including revenue, profitability, margins, cost drivers, liquidity trends, capital structure, and debt along with financial ratios.
   – If the metrics are presented in tables, rewrite them in the section “Important Figures and Tables” instead of here.
   – Also Include commentary on:
     – Revenue recognition patterns or timing shifts.
     – Significant margin changes or cost structure shifts.
     – Increases in accounts receivable, inventory, or other working capital components relative to sales without clear justification.
     – Use of non-recurring items, adjustments, or changes in estimates that materially impact results.
     – Subjective rationalizations for financial outcomes (e.g., references to “strong demand” or “improved efficiencies”) that lack numeric validation.

** 4. Identified Risks and Uncertainties**
   – Summarize disclosed risks, including operational, supply chain, regulatory, competitive, legal, and macroeconomic risks.
   – Capture both concrete risks and:
     – Subjective assessments of risk severity.
     – Ambiguous or hedged language (e.g., “may,” “could,” “uncertain”).
     – Shifts in tone, emphasis, or presentation of risks compared to prior periods.

** 5. Forward-Looking Statements and Guidance**
   – Capture management’s expectations, forecasts, assumptions, and outlook for future periods.
   – Highlight:
     – Changes in guidance or underlying assumptions.
     – Optimistic tone, hedging, or caveats (e.g., “expects,” “believes,” “aims”).
     – Whether forward-looking statements are grounded in quantifiable drivers or rely mainly on qualitative assertions.

** 6. Significant Changes, Events, or Developments**
   – Summarize material recent or upcoming events affecting the business, such as mergers, acquisitions, divestitures, leadership changes, legal proceedings, regulatory actions, or external shocks.
   – Note how management frames these events—whether impacts are clearly quantified or described with vague or qualitative language.

** 7. Important Figures and Tables**
   – Extract key figures, tables, or financial data that are critical to understanding the MD&A.
   – For each table:
        Recreate the exact table content in clean markdown format preceded by the table title.

** 8. Management Explanations and Justifications**
   – Capture how management explains or justifies operational and financial results, risks, or variances.
   – Pay attention to:
     – Vague, broad, or overly generic justifications.
     – Repetitive use of boilerplate terms (e.g., “market conditions,” “execution excellence”) without specific detail.
     – Narratives that shift accountability to external factors or uncontrollable circumstances without precise quantification.

** 9. Accounting Estimates, Judgments, and Policy Changes**
   – Summarize any disclosures related to:
     – Changes in accounting policies, methodologies, or estimates.
     – Adjustments to key assumptions (e.g., impairments, allowances, revenue recognition).
     – Areas where significant management judgment materially affects reported results.
   – Note whether explanations are clear, detailed, vague, hedged, or superficial.

** 10. Capital Allocation and Liquidity Management**
    – Summarize commentary on:
      – Cash management strategies, liquidity preservation, and debt management.
      – Capital expenditures, share repurchases, dividend policies, and financing activities.
    – Highlight any:
      – Indications of liquidity stress.
      – Mismatches between optimistic narratives and defensive liquidity actions (e.g., drawing on credit lines despite claimed strong financial performance).

** 11. Legal, Regulatory, and Compliance Matters**
    – Summarize discussions related to:
      – Ongoing or pending litigation.
      – Regulatory investigations or changes.
      – Compliance risks, including ESG-related disclosures that have material financial implications.
    – Note whether these issues are presented transparently, minimized, or framed with ambiguous language.

---

**Formatting Instructions:**
– Use section headers exactly as written above ie with the numbers and titles.
– Present each point as a bullet (–) under the appropriate section.
– Include both objective data and subjective commentary.
– Explicitly note subjective explanations, optimistic framing, hedging, or vague descriptions wherever they appear.
– Be precise and factual but the summary should be detailed 
– Avoid redundancy; each bullet must convey a distinct, meaningful insight.

**Critical Constraint:**
– Base the summary **strictly on the content explicitly stated in the MD&A.** 
– Do not include any external knowledge, assumptions, interpretations, or analysis beyond the document provided.
– Only output the summary — do not include any commentary, explanations, or meta-text.
"""


SEC_MDA_OUTPUT_DIR = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED"
SEC_MDA_QUARTERLY_OUTPUT_DIR = SEC_MDA_OUTPUT_DIR / "quarterly"
TMP_OUTPUT_DIR = Path("/tmp/mda_summarized_temp")

# Constants
MDA_PATH = MDA_DATASET_PATH / "quarterly"


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

        if clean_small_lines:
            # Limit the content to 400,000 characters(~ equcivalent to 100,000 words)
            mda_content = mda_content[:400000]

        return mda_content


def load_and_data_index():
    """
    Load the dataset index corresponding to the V4 configuration (ie. 95%, 5% split )
    """
    full_df = pd.read_csv(load_full_data_path({"dataset_version": "company_isolated_splitting"}))

    return full_df[["mda_quarter_id"]]


def process_chunk_with_api_key(
    mda_quarter_ids_chunk: List[str], vllm_config: Dict[str, Any]
):
    """
    Process a chunk of MDA files using the configured VLLM API.

    Args:
        mda_quarter_ids_chunk (List[str]): A chunk of MDA quarter IDs to process
        vllm_config (Dict[str, Any]): Configuration for the VLLM endpoint (endpoint, model, name)
    """
    endpoint_name = vllm_config.get("name", vllm_config["endpoint"])
    vllm_endpoint = vllm_config["endpoint"]
    vllm_model = vllm_config["model"]

    process_id = mp.current_process().name
    logging.info(
        f"Process {process_id} starting with {len(mda_quarter_ids_chunk)} files using VLLM endpoint: {endpoint_name}"
    )

    process_tmp_dir = TMP_OUTPUT_DIR / f"process_{process_id}"
    process_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client once per process with its assigned endpoint
    vllm_client = openai.OpenAI(
        base_url=vllm_endpoint, api_key="sk-no-key-required"  # Placeholder for VLLM
    )

    def backoff_hdlr(details):
        """Handler function to force a 10 second wait between retries"""
        print(
            f"Backing off for 10 seconds after {details['tries']} tries. Exception: {details['exception']} for endpoint: {endpoint_name}"
        )
        time.sleep(20)

    @on_exception(
        expo,
        (ratelimit.RateLimitException,),
        max_tries=MAX_RETRIES + 2,  # More retries for API issues
        on_backoff=backoff_hdlr,
    )
    @ratelimit.limits(calls=MAX_RPM_VLLM, period=60)
    def call_llm_to_summarize(
        mda_content,
        temperature=0.7,
        model=vllm_model,  # Use the model specified in the config
        quarter_info="Unknown Quarter",
    ):
        """
        Call the VLLM-hosted LLM to summarize the MDA content.
        """
        system_prompt = MDA_SUMMARIZE_SYSTEM_PROMPT.format(quarter_info=quarter_info)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mda_content},
        ]

        response = vllm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.8,
            # top_logprobs=20,
            max_tokens=8000,
            timeout=1e5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        return response.choices[0].message.content

    def process_single_mda_file(mda_quarter_id, num_retries=MAX_RETRIES):
        """
        Process a single MDA file and save the summarized content.
        """
        output_file = SEC_MDA_QUARTERLY_OUTPUT_DIR / f"{mda_quarter_id}.txt"
        if output_file.exists():
            logging.info(f"SKipping {output_file.name}")
            return True, mda_quarter_id

        try:
            is_retrying = num_retries < MAX_RETRIES
            mda_content = load_mda_content(
                mda_quarter_id, clean_small_lines=is_retrying
            )
            quarter_info = extract_quarter_info(mda_quarter_id)

            summary = call_llm_to_summarize(
                mda_content,
                temperature=0.7,
                model=vllm_model,
                quarter_info=quarter_info,
            )

            temp_file = process_tmp_dir / f"{mda_quarter_id}.txt"
            with open(temp_file, "w", encoding="utf-8") as file:
                file.write(summary)

            final_file = SEC_MDA_QUARTERLY_OUTPUT_DIR / f"{mda_quarter_id}.txt"
            shutil.move(str(temp_file), str(final_file))

            return True, mda_quarter_id
        except Exception as e:
            if num_retries > 0:
                logging.warning(
                    f"Retrying extraction of {mda_quarter_id} with {endpoint_name} - Attempt {MAX_RETRIES - num_retries + 1} - Error: {e}"
                )
                time.sleep(10)
                return process_single_mda_file(
                    mda_quarter_id, num_retries=num_retries - 1
                )
            else:
                logging.error(
                    f"Failed to process {mda_quarter_id} with {endpoint_name} after multiple retries: {str(e)}"
                )
                return False, mda_quarter_id

    results = Parallel(n_jobs=MAX_SIMULATENOUS_REQUESTS_PER_PROCESS, prefer="threads",)(
        delayed(process_single_mda_file)(mda_id)
        for mda_id in tqdm(
            mda_quarter_ids_chunk,
            desc=f"Process {process_id} ({endpoint_name})",
            unit="file",
            position=mp.current_process()._identity[0] % 10
            if mp.current_process()._identity
            else 0,
        )
    )

    successes = sum(1 for result, _ in results if result)
    failures = sum(1 for result, _ in results if not result)

    logging.info(
        f"Process {process_id} ({endpoint_name}): {successes} successes, {failures} failures"
    )

    try:
        shutil.rmtree(str(process_tmp_dir))
    except Exception as e:
        logging.warning(
            f"Failed to clean up temporary directory {process_tmp_dir}: {str(e)}"
        )

    return results


def process_mda_files(mda_quarter_ids: List[str], vllm_configs: List[Dict[str, Any]]):
    """
    Distribute MDA files across multiple processes, each using a distinct VLLM endpoint.

    Args:
        mda_quarter_ids (List[str]): List of MDA quarter IDs to process
        vllm_configs (List[Dict[str, Any]]): List of VLLM endpoint configurations
    """
    if not vllm_configs:
        logging.error("No VLLM configurations provided. Exiting.")
        return

    random.shuffle(mda_quarter_ids)

    num_endpoints = len(vllm_configs)
    # Distribute files as evenly as possible among available endpoints/processes
    chunk_size = (len(mda_quarter_ids) + num_endpoints - 1) // num_endpoints

    chunks_for_processes = []
    current_idx = 0
    for i in range(num_endpoints):
        chunk = mda_quarter_ids[current_idx : current_idx + chunk_size]
        chunks_for_processes.append(chunk)
        current_idx += chunk_size
        if current_idx >= len(mda_quarter_ids):
            break  # No more files to distribute

    # Ensure we use exactly as many processes as we have active VLLM configs
    processes = []
    for i in range(len(vllm_configs)):
        config = vllm_configs[i]
        # Assign a chunk to each process (handle cases where there are more endpoints than files)
        chunk = chunks_for_processes[i] if i < len(chunks_for_processes) else []

        if not chunk:  # Skip process if no files assigned to it
            logging.info(
                f"Skipping process for endpoint {config.get('name', config['endpoint'])}: No files assigned."
            )
            continue

        p = mp.Process(
            target=process_chunk_with_api_key,
            args=(chunk, config),
            name=f"VLLMProcess-{i}-{config.get('name', 'UnnamedEndpoint')}",
        )
        processes.append(p)
        p.start()
        logging.info(
            f"Started process {p.name} for endpoint: {config.get('name', config['endpoint'])}"
        )

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
    except Exception:
        pass  # Fallback to default
    return "Unknown Quarter"


def main():
    """
    Main function to run the MDA summarization process.
    """
    mp.set_start_method("spawn", force=True)

    initialize_dirs()

    if TMP_OUTPUT_DIR.exists():
        try:
            shutil.rmtree(str(TMP_OUTPUT_DIR))
            logging.info(f"Cleaned up old temporary directory: {TMP_OUTPUT_DIR}")
        except Exception as e:
            logging.warning(f"Failed to clean up old temporary directory: {str(e)}")

    TMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configure_logger(PREPROCESSED_PATH / "vllm_mda_summarizing.log", logging.INFO)

    try:
        vllm_configs = load_vllm_configs()
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading VLLM configurations: {e}")
        return

    logging.info("Loading data index")
    data_df = load_and_data_index()
    mda_quarter_ids = data_df["mda_quarter_id"].tolist()

    processed_ids = [f.stem for f in SEC_MDA_QUARTERLY_OUTPUT_DIR.glob("*.txt")]
    mda_quarter_ids = [id for id in mda_quarter_ids if id not in processed_ids]

    logging.info(f"Found {len(mda_quarter_ids)} unprocessed MDA files")

    num_processes = len(vllm_configs)  # One process per active VLLM endpoint
    if num_processes == 0:
        logging.error("No active VLLM endpoints configured. Exiting.")
        return

    logging.info(
        f"Starting summarization with {num_processes} worker processes, each assigned a distinct VLLM endpoint."
    )

    process_mda_files(mda_quarter_ids, vllm_configs)


if __name__ == "__main__":
    main()
