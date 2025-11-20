import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from researchpkg.anomaly_detection.config import (
    PREPROCESSED_PATH,
    PREPROCESSED_PATH_EXTENDED,
    PREPROCESSED_PATH_NOAUG,
    SEED_TRAINING,
    SIC_INDEX_FILE,
)


def get_last_checkpoint(checkpoint_dir: Path):
    checkpoints = [
        d
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    latest_checkpoint = (
        max(checkpoints, key=lambda x: int(str(x).split("-")[-1]))
        if checkpoints
        else None
    )
    return checkpoint_dir / latest_checkpoint if latest_checkpoint else None


def get_tokenizer_completion_instruction(tokenizer):
    dummy_chat = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "Hi!^mx"},
    ]
    long_seq = tokenizer.apply_chat_template(
        dummy_chat, add_generation_prompt=True, tokenize=False
    )
    small_seq = tokenizer.apply_chat_template(dummy_chat, tokenize=False)
    out = long_seq.replace(small_seq, "")
    return out


def get_tokenizer_start_instruction(tokenizer):
    dummy_chat = [
        {"role": "system", "content": "#12345#"},
        {"role": "user", "content": "hi"},
    ]
    long_seq = tokenizer.apply_chat_template(
        dummy_chat, add_special_tokens=True, tokenize=False
    )
    out = long_seq.split("#12345#")[0]
    return out


def load_sic_industry_title_index():
    """
    Load SIC aggregation data from a CSV file.

    Args:
        sic_agg_path (str): Path to the SIC aggregation CSV file.

    Returns:
        dict: A dictionary mapping SIC codes to their corresponding categories.
    """
    sic_df = pd.read_csv(SIC_INDEX_FILE, sep="|", dtype={"sic": str})
    return sic_df.set_index("sic")["industry_title"].to_dict()


def load_cik_company_mapping(dataset_version="company_isolated_splitting"):
    """
    Load CIK to Company mapping from a CSV file.

    Returns:
        dict: A dictionary mapping CIK codes to their corresponding company names.
    """
    MAPPING_FILE = PREPROCESSED_PATH_EXTENDED / dataset_version / "cik_to_company.yaml"
    with open(MAPPING_FILE, "r") as file:
        cik_company_mapping = yaml.safe_load(file)
    return cik_company_mapping


# GRPO Rewards function

# GRPO Rewrad functions (From : https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=cXk993X6C2ZZ)
def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        # print("Answer not found in text")
        # print("Text:", text)
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_reasoning(text: str) -> str:
    if "<reasoning>" not in text or "</reasoning>" not in text:
        return ""
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()


def extract_confidence(text: str) -> str:
    if "<confidence>" not in text or "</confidence>" not in text:
        return 0
    confidence = text.split("<confidence>")[-1]
    confidence = confidence.split("</confidence>")[0]
    return int(confidence.strip())


def extract_xml_reasoning(text: str) -> str:
    if "<reasoning>" not in text or "</reasoning>" not in text:
        return ""
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()


def is_correct_answer(ref_answer, completion):
    """
    Check if the reasongin final answer is correct
    """
    # print(f"Reference answer: {ref_answer}, Completion: {completion}")
    if ref_answer == "Not Fraud":
        return "not fraud" in completion.lower() or "no fraud" in completion.lower()
    elif ref_answer == "Fraud":
        return "fraud" in completion.lower() and not (
            "not fraud" in completion.lower() or "no fraud" in completion.lower()
        )
    return False


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # print("Prompts:", prompts)
    responses = [completion for completion in completions]

    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = [
        2.0 if is_correct_answer(ref, output) else 0
        for output, ref in zip(extracted_responses, answer)
    ]
    return rewards
    # extracted_confidences = [
    #     extract_confidence(r) for r in responses
    # ]

    # weighted_rewards = [
    #     reward*confidence for reward, confidence in zip(rewards, extracted_confidences)
    # ]
    # return weighted_rewards


def format_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # print(f"Completions: {completions}")
    is_well_formed = (
        lambda x: extract_xml_answer(x) != ""
        and extract_xml_reasoning(x) != ""
        and extract_confidence(x) != 0
    )
    responses = [completion[0]["content"] for completion in completions]
    return [1.0 if is_well_formed(r) else 0 for r in responses]


def count_xml(text, hard=True) -> float:
    if hard:
        count = 0.0

        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001

        if count == 0:
            return count_xml(text, hard=False)

        return count
    else:
        # Recherche des sous-chaînes partielles les plus longues pour chaque type de balise
        sub_tags = {
            "<reasoning": [
                "<re",
                "<reas",
                "<reason",
                "<reasoni",
                "<reasonin",
                "<reasoning",
            ],
            "<answer": ["<an", "<ans", "<answ", "<answe", "<answer"],
            "</answer": ["</an", "</ans", "</answ", "</answe", "</answer"],
            "</reasoning": [
                "</re",
                "</reas",
                "</reason",
                "</reasoni",
                "</reasonin",
                "</reasoning",
            ],
        }

        count = 0
        for root, subs in sub_tags.items():
            longest_match = max(
                (tag for tag in subs if tag in text), key=len, default=""
            )
            count += (
                len(re.findall(re.escape(longest_match), text)) * 0.1
                if longest_match
                else 0
            )

        return max(count, 0)


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_length(x, target_size=1024) -> float:
    """
    Computes a reward based on how close x is to the target size.

    """
    if x == 0:
        return 0

    # Compute the Gaussian value
    return -abs(target_size - x)


def reward_reponse_length(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks the length of the answer."""
    rewards = [
        reward_length(len((completion[0]["content"]))) for completion in completions
    ]
    return rewards


def llm_generate(
    model,
    tokenizer,
    prompts,
    max_length,
    max_new_tokens,
    return_ids=False,
):
    """
    Generate completions using the model's fast generate method, supports batch processing.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer for encoding/decoding
        prompts: A string prompt or a list of prompts
        max_length: Maximum context length
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        List of decoded outputs if input was a list, otherwise a single string
    """

    single_input = isinstance(prompts, str)
    if single_input:
        inputs = tokenizer.encode(prompts, return_tensors="pt").to(model.device)

        prediction = model.generate(
            inputs,
            temperature=0.001,
            max_new_tokens=max_new_tokens,
        )[0]

        if return_ids:
            # Do not decode the prediction
            return prediction

        # Decode all outputs
        return tokenizer.decode(prediction, skip_special_tokens=False)

    else:  # Tokenize all prompts as a batch
        # CUrrently very slow with batching sow we will call it one by
        return [
            llm_generate(
                model,
                tokenizer,
                prompt,
                max_length,
                max_new_tokens,
                return_ids=return_ids,
            )
            for prompt in prompts
        ]

        # We experimentlly saw that batching makes the generation ten times slower.
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        ).to(model.device)


def llm_generate_with_logits(
    model,
    tokenizer,
    prompts,
    max_length,
    max_new_tokens,
    return_ids=False,
):
    """
    Generate completions using the model's fast generate method, supports batch processing.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer for encoding/decoding
        prompts: A string prompt or a list of prompts
        max_length: Maximum context length
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        List of decoded outputs if input was a list, otherwise a single string
    """

    single_input = isinstance(prompts, str)
    if single_input:
        inputs = tokenizer.encode(prompts, return_tensors="pt").to(model.device)

        prediction = model.generate(
            inputs,
            temperature=0.001,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )[0]

        if return_ids:
            # Do not decode the prediction
            return prediction, prediction.scores

        # Decode all outputs
        return (
            tokenizer.decode(prediction, skip_special_tokens=False),
            prediction.scores,
        )

    else:  # Tokenize all prompts as a batch
        # CUrrently very slow with batching sow we will call it one by
        outputs = [
            llm_generate(
                model,
                tokenizer,
                prompt,
                max_length,
                max_new_tokens,
                return_ids=return_ids,
            )
            for prompt in prompts
        ]
        predictions = [output[0] for output in outputs]
        scores = [output[1] for output in outputs]
        return predictions, scores

        # We experimentlly saw that batching makes the generation ten times slower.
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        ).to(model.device)


def llm_fast_generate(model, prompts, max_length, max_new_tokens, lora_request):
    """
    Generate completions using the model's fast generate method, supports batch processing.
    Using internal unsloth VLLM implementation.

    text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "Calculate pi."},
        ], tokenize = False, add_generation_prompt = True)

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature = 0.8,
            top_p = 0.95,
            max_tokens = 1024,
        )
        output = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = model.load_lora("grpo_saved_lora"),
        )[0].outputs[0].text

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer for encoding/decoding
        prompts: A string prompt or a list of prompts
        max_length: Maximum context length
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        List of decoded outputs if input was a list, otherwise a single string
    """

    single_input = isinstance(prompts, str)
    if single_input:
        prompts = [prompts]
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=max_new_tokens,
    )
    outputs = model.fast_generate(
        prompts, sampling_params=sampling_params, lora_request=lora_request
    )
    # Extract completions
    results = []
    for output in outputs:
        results.append(outputs)
    # Return single string if input was single, otherwise return list
    return results[0] if single_input else results


def llm_vllm_generate(
    model, tokenizer, vllm_client, prompts, max_length, max_new_tokens
):
    """
    Generate completions using vLLM, supports batch processing.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer for encoding/decoding
        vllm_client: The vLLM client
        prompts: A string prompt or a list of prompts
        max_length: Maximum context length
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        List of decoded outputs if input was a list, otherwise a single string
    """
    single_input = isinstance(prompts, str)
    if single_input:
        prompts = [prompts]

    outputs = vllm_client.generate(prompts, temperature=0.01, max_tokens=max_new_tokens)

    # Extract completions
    results = []
    for output in outputs:
        results.append(tokenizer.decode(output, skip_special_tokens=False))

    # Return single string if input was single, otherwise return list
    return results[0] if single_input else results


def drop_random_keys(d, percentage):
    """Randomly drops x% of keys from a dictionary."""
    num_to_drop = int(len(d) * (percentage / 100))
    keys_to_drop = set(random.sample(list(d.keys()), num_to_drop))
    return {k: v for k, v in d.items() if k not in keys_to_drop}


def split_dataset_randomly(df, test_size=0.1,seed=None):
    if seed is not None:
        logging.info(f"Seeding random number generator with seed: {seed}")
        #Seed the random number generator for reproducibility
        random.seed(seed)
        np.random.seed(seed)
    logging.info("Creating validation split from data, with random splintg")
    df_train, d_val = train_test_split(df, test_size=test_size)
    return df_train, d_val


def calculate_split_metrics(df, unique_companies_df, label="Set", log_output=True):
    """Calculates and optionally logs key metrics for a given dataset split."""
    
    
    
    metrics = {}

    if df.empty or unique_companies_df.empty:
        # Initialize metrics for empty sets
        metrics = {
            "num_reports": 0,
            "num_companies": 0,
            "num_fraud_reports": 0,
            "overall_fraud_ratio": 0.0,
            "company_distribution": {},
            "report_distribution": {},
            "sector_fraud_ratio": {},
            "sector_fraud_report_counts": {},
        }
        if log_output:
            logging.info(f"--- {label} Metrics ---")
            logging.info("  Set is empty.")
        return metrics

    num_reports = len(df)
    num_companies = len(unique_companies_df)
    num_fraud_reports = df["is_fraud"].sum()
    overall_fraud_ratio = num_fraud_reports / num_reports if num_reports > 0 else 0.0

    metrics["num_reports"] = num_reports
    metrics["num_companies"] = num_companies
    metrics["num_fraud_reports"] = num_fraud_reports
    metrics["overall_fraud_ratio"] = overall_fraud_ratio

    # Company distribution by sector
    company_dist = unique_companies_df["sicagg"].value_counts().to_dict()
    metrics["company_distribution"] = company_dist

    # Report distribution by sector
    report_dist = df["sicagg"].value_counts().to_dict()
    metrics["report_distribution"] = report_dist

    # Per-sector fraud ratio and counts
    # Per-sector fraud ratio and counts
    sector_metrics = (
        df.groupby("sicagg")
        .agg(total_reports=("report_id", "count"), fraud_reports=("is_fraud", "sum"))
        .reset_index()
    )
    # Ensure 'fraud_ratio' column is created even if no fraud exists
    if "fraud_reports" not in sector_metrics.columns:
        sector_metrics["fraud_reports"] = 0
    if "total_reports" not in sector_metrics.columns:
        sector_metrics[
            "total_reports"
        ] = 0  # Should not happen with report_id aggregation

    sector_metrics["fraud_ratio"] = sector_metrics.apply(
        lambda row: row["fraud_reports"] / row["total_reports"]
        if row["total_reports"] > 0
        else 0.0,
        axis=1,
    )
    metrics["sector_fraud_ratio"] = (
        sector_metrics.set_index("sicagg")["fraud_ratio"].fillna(0.0).to_dict()
    )  # Fill NaN if sector has 0 reports
    metrics["sector_fraud_report_counts"] = sector_metrics.set_index("sicagg")[
        "fraud_reports"
    ].to_dict()

    # Ensure all sectors present in the company list are represented, even if they have 0 reports/fraud in this subset
    all_sectors_in_split = unique_companies_df["sicagg"].unique()
    for sector in all_sectors_in_split:
        metrics["sector_fraud_ratio"].setdefault(sector, 0.0)
        metrics["sector_fraud_report_counts"].setdefault(sector, 0)

    if log_output:
        logging.info(f"--- {label} Metrics ---")
        logging.info(f"  Companies: {num_companies}")
        logging.info(
            f"  Reports: {num_reports} (Fraud: {num_fraud_reports}, Non-Fraud: {num_reports - num_fraud_reports})"
        )
        logging.info(f"  Overall Fraud Ratio: {overall_fraud_ratio:.4f}")
        logging.info(f"  Company Distribution by Sector: {company_dist}")
        # logging.info(f"  Report Distribution by Sector: {report_dist}") # Can be verbose
        logging.info(
            f"  Fraud Reports by Sector: {metrics['sector_fraud_report_counts']}"
        )
        logging.info(
            f"  Fraud Ratio by Sector: {{ {', '.join([f'{k}: {v:.3f}' for k, v in sorted(metrics['sector_fraud_ratio'].items())])} }}"
        )

    return metrics


# --- Simplified splitting function with Randomness ---
def split_dataset_by_cik(
    df,
    test_size=0.1,
    adjustment_iterations=20,
    tolerance_overall_ratio=0.01,
    swap_candidate_pool_factor=2, 
    seed=None
):
    """
    Splits dataset by CIK using a simplified, randomized approach:
    1. Calculate target fraud report count per sector for the test set.
    2. Randomly select fraud-containing companies within sectors to meet targets.
    3. Randomly fill remaining test set size based on sector company targets.
    4. Adjust final test set via randomized swaps to match overall fraud ratio.

    Args:
        df (pd.DataFrame): Input dataframe with 'cik', 'sicagg', 'is_fraud'.
        test_size (float): Target proportion of companies in the test set.
        adjustment_iterations (int): Max iterations for the final overall ratio adjustment.
        tolerance_overall_ratio (float): Allowed absolute deviation for overall fraud ratio.
        swap_candidate_pool_factor (int): Multiplier for swap candidate pool size before sampling.

    Returns:
        tuple: (train_df, test_df)
    """
    if seed is not None:
        logging.info(f"Seeding random number generator with seed: {seed}")
        #Seed the random number generator for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
    
    
    logging.info("Starting SIMPLIFIED & RANDOMIZED CIK-based split.")
    if "report_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "report_id"})
        logging.warning("Added 'report_id' column based on index.")
    for col in ["cik", "sicagg", "is_fraud", "report_id"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    df["sicagg"] = df["sicagg"].fillna("Unknown")

    # Company-level aggregation
    unique_companies = (
        df.groupby("cik")
        .agg(
            sicagg=("sicagg", "first"),
            total_reports=("report_id", "count"),
            fraud_reports=("is_fraud", "sum"),
        )
        .reset_index()
    )
    unique_companies["company_fraud_ratio"] = (
        unique_companies["fraud_reports"] / unique_companies["total_reports"]
    )
    unique_companies["company_fraud_ratio"] = unique_companies[
        "company_fraud_ratio"
    ].fillna(0.0)
    logging.info(f"Aggregated {len(unique_companies)} unique companies.")

    # Global Stats & Targets (same as before)
    global_metrics = calculate_split_metrics(df, unique_companies, "Global Dataset")
    target_test_companies_total = int(
        round(global_metrics["num_companies"] * test_size)
    )
    target_overall_fraud_ratio = global_metrics["overall_fraud_ratio"]
    sector_fraud_report_counts_global = (
        df[df["is_fraud"] == 1]["sicagg"].value_counts().to_dict()
    )
    target_fraud_reports_per_sector = {
        sector: int(round(count * test_size))
        for sector, count in sector_fraud_report_counts_global.items()
    }
    for sector in unique_companies["sicagg"].unique():
        target_fraud_reports_per_sector.setdefault(sector, 0)
    sector_company_counts_global = unique_companies["sicagg"].value_counts().to_dict()
    target_companies_per_sector = {
        sector: int(round(count * test_size))
        for sector, count in sector_company_counts_global.items()
    }
    current_co_sum = sum(target_companies_per_sector.values())
    co_diff = target_test_companies_total - current_co_sum
    if co_diff != 0 and target_companies_per_sector:
        adj_co_sector = sorted(
            target_companies_per_sector.items(), key=lambda item: item[1], reverse=True
        )[0][0]
        target_companies_per_sector[adj_co_sector] = max(
            0, target_companies_per_sector[adj_co_sector] + co_diff
        )

    logging.info(f"Target Test Set Size: ~{target_test_companies_total} companies.")
    logging.info(f"Target Overall Fraud Ratio: {target_overall_fraud_ratio:.4f}")
    logging.info(f"Target Fraud Reports per Sector: {target_fraud_reports_per_sector}")
    logging.info(f"Target Companies per Sector: {target_companies_per_sector}")

    # --- Phase 1: Randomized Fraud-Driven Company Selection ---
    logging.info(
        "--- Phase 1: Selecting Companies to Meet Sector Fraud Report Targets (Randomized) ---"
    )
    selected_test_ciks = set()
    fraud_contributing_ciks = unique_companies[
        unique_companies["fraud_reports"] > 0
    ].copy()
    # Initial shuffle of the pool helps break ties randomly if needed later

    sectors_by_fraud_target = sorted(
        target_fraud_reports_per_sector.items(), key=lambda item: item[1], reverse=True
    )
    current_fraud_reports_in_test = defaultdict(int)

    for sector, target_reports in sectors_by_fraud_target:
        if target_reports <= 0:
            continue

        logging.debug(
            f"  Processing Sector '{sector}', Target Fraud Reports: {target_reports}"
        )
        # Candidates from this sector that have fraud and are not already selected
        candidates = (
            fraud_contributing_ciks[
                (fraud_contributing_ciks["sicagg"] == sector)
                & (~fraud_contributing_ciks["cik"].isin(selected_test_ciks))
            ]
            .sort_values("fraud_reports")
            .copy()
        )

        for _, company in candidates.iterrows():
            cik = company["cik"]
            reports_to_add = company["fraud_reports"]

            if current_fraud_reports_in_test[sector] < target_reports:
                selected_test_ciks.add(cik)
                current_fraud_reports_in_test[sector] += reports_to_add
                logging.debug(
                    f"    Selected CIK {cik} (Adds {reports_to_add} fraud reports). Sector total now: {current_fraud_reports_in_test[sector]}"
                )
            else:
                # Stop searching this sector once target is met/exceeded
                # Because candidates are shuffled, we don't necessarily hit target exactly
                # or pick the highest fraud first.
                logging.debug(f"    Target met/exceeded for sector '{sector}'.")
                break

    logging.info(
        f"Phase 1 Result: Selected {len(selected_test_ciks)} companies containing fraud."
    )
    logging.info(
        f"  Achieved Fraud Reports per Sector: {dict(current_fraud_reports_in_test)}"
    )

    # --- Phase 2: Randomized Fill Remaining Company Slots ---
    logging.info("--- Phase 2: Filling Remaining Test Set Slots (Randomized) ---")
    companies_needed_total = target_test_companies_total - len(selected_test_ciks)
    logging.info(f"  Need to select {companies_needed_total} more companies.")

    current_companies_per_sector = defaultdict(int)
    if selected_test_ciks:
        selected_co_df = unique_companies[
            unique_companies["cik"].isin(selected_test_ciks)
        ]
        counts = selected_co_df["sicagg"].value_counts().to_dict()
        for sec, count in counts.items():
            current_companies_per_sector[sec] = count

    companies_needed_per_sector = {}
    for sector, target_co_count in target_companies_per_sector.items():
        needed = target_co_count - current_companies_per_sector.get(sector, 0)
        if needed > 0:
            companies_needed_per_sector[sector] = needed

    available_ciks_for_fill = unique_companies[
        ~unique_companies["cik"].isin(selected_test_ciks)
    ].copy()
    non_fraud_available = available_ciks_for_fill[
        available_ciks_for_fill["fraud_reports"] == 0
    ].copy()
    fraud_available_rem = available_ciks_for_fill[
        available_ciks_for_fill["fraud_reports"] > 0
    ].copy()

    sectors_by_need = sorted(
        companies_needed_per_sector.items(), key=lambda item: item[1], reverse=True
    )
    companies_added_phase2 = 0

    for sector, needed in sectors_by_need:
        if companies_added_phase2 >= companies_needed_total:
            break

        # Try adding non-fraud first
        candidates_nf = non_fraud_available[
            non_fraud_available["sicagg"] == sector
        ].copy()
        # *** ADD RANDOMNESS: Shuffle candidates ***
        candidates_nf = candidates_nf.sample(frac=1, random_state=hash(sector) + 1)

        can_add_nf = min(
            needed, len(candidates_nf), companies_needed_total - companies_added_phase2
        )
        if can_add_nf > 0:
            ciks_to_add = candidates_nf.head(can_add_nf)["cik"].tolist()
            selected_test_ciks.update(ciks_to_add)
            companies_added_phase2 += can_add_nf
            needed -= can_add_nf
            non_fraud_available = non_fraud_available[
                ~non_fraud_available["cik"].isin(ciks_to_add)
            ]
            logging.debug(
                f"    Added {can_add_nf} non-fraud CIKs for Sector '{sector}'. Total added: {companies_added_phase2}"
            )

        # If still need companies for this sector, try adding remaining fraud-contributors
        if needed > 0 and companies_added_phase2 < companies_needed_total:
            candidates_f = fraud_available_rem[
                fraud_available_rem["sicagg"] == sector
            ].copy()
            # *** ADD RANDOMNESS: Shuffle candidates ***
            candidates_f = candidates_f.sample(frac=1, random_state=hash(sector) + 2)

            can_add_f = min(
                needed,
                len(candidates_f),
                companies_needed_total - companies_added_phase2,
            )
            if can_add_f > 0:
                ciks_to_add = candidates_f.head(can_add_f)["cik"].tolist()
                selected_test_ciks.update(ciks_to_add)
                companies_added_phase2 += can_add_f
                fraud_available_rem = fraud_available_rem[
                    ~fraud_available_rem["cik"].isin(ciks_to_add)
                ]
                logging.debug(
                    f"    Added {can_add_f} remaining fraud CIKs for Sector '{sector}'. Total added: {companies_added_phase2}"
                )

    # Global fill if needed (already randomized)
    if companies_added_phase2 < companies_needed_total:
        remaining_needed_global = companies_needed_total - companies_added_phase2
        logging.info(
            f"  Target not met by sector fill, needing {remaining_needed_global} more globally."
        )
        remaining_pool = pd.concat([non_fraud_available, fraud_available_rem]).sample(
            frac=1
        )
        can_add_globally = min(remaining_needed_global, len(remaining_pool))
        if can_add_globally > 0:
            ciks_to_add_global = remaining_pool.head(can_add_globally)["cik"].tolist()
            selected_test_ciks.update(ciks_to_add_global)
            companies_added_phase2 += can_add_globally
            logging.debug(
                f"    Added {can_add_globally} more CIKs globally to reach target count. Total added: {companies_added_phase2}"
            )

    logging.info(
        f"Phase 2 Result: Added {companies_added_phase2} companies. Total test CIKs: {len(selected_test_ciks)}"
    )

    # --- Phase 3: Randomized Adjustment for Overall Ratio ---
    logging.info(
        "--- Phase 3: Adjusting Test Set for Overall Fraud Ratio (Randomized Swaps) ---"
    )
    current_test_ciks = selected_test_ciks
    current_train_ciks = set(unique_companies["cik"]) - current_test_ciks

    # Define pool size for sampling swap candidates
    SWAP_SAMPLE_SIZE = 5  # How many candidates to actually sample and check
    POOL_SIZE = SWAP_SAMPLE_SIZE * swap_candidate_pool_factor

    for i in range(adjustment_iterations):
        test_df_current = df[df["cik"].isin(current_test_ciks)]
        test_companies_current = unique_companies[
            unique_companies["cik"].isin(current_test_ciks)
        ]
        train_companies_current = unique_companies[
            unique_companies["cik"].isin(current_train_ciks)
        ]

        if test_companies_current.empty or train_companies_current.empty:
            logging.warning(" Stop adjustment: Train or Test set is empty.")
            break

        test_metrics = calculate_split_metrics(
            test_df_current,
            test_companies_current,
            f"Adj Iter {i+1} Test",
            log_output=False,
        )
        current_overall_ratio = test_metrics["overall_fraud_ratio"]
        ratio_diff = current_overall_ratio - target_overall_fraud_ratio

        logging.info(
            f"  Iter {i+1}/{adjustment_iterations}: Current Ratio={current_overall_ratio:.4f}, Target={target_overall_fraud_ratio:.4f}, Diff={ratio_diff:+.4f}"
        )

        if abs(ratio_diff) <= tolerance_overall_ratio:
            logging.info("  ✓ Overall ratio within tolerance. Adjustment complete.")
            break

        best_swap = None
        max_improvement = -float("inf")  # Max reduction in absolute difference

        # *** ADD RANDOMNESS: Select candidates from a larger pool ***
        if ratio_diff > 0:  # Ratio too high
            # Get pool of high-ratio test CIKs
            test_cand_pool = test_companies_current.sort_values(
                "company_fraud_ratio", ascending=False
            ).head(POOL_SIZE)
            # Get pool of low-ratio train CIKs
            train_cand_pool = train_companies_current.sort_values(
                "company_fraud_ratio", ascending=True
            ).head(POOL_SIZE)
        else:  # Ratio too low
            # Get pool of low-ratio test CIKs
            test_cand_pool = test_companies_current.sort_values(
                "company_fraud_ratio", ascending=True
            ).head(POOL_SIZE)
            # Get pool of high-ratio train CIKs
            train_cand_pool = train_companies_current.sort_values(
                "company_fraud_ratio", ascending=False
            ).head(POOL_SIZE)

        if test_cand_pool.empty or train_cand_pool.empty:
            logging.warning("  Cannot find swap candidate pools. Stopping adjustment.")
            break

        # *** ADD RANDOMNESS: Sample N candidates from the pools ***
        test_cands_sample = test_cand_pool.sample(
            min(SWAP_SAMPLE_SIZE, len(test_cand_pool))
        )
        train_cands_sample = train_cand_pool.sample(
            min(SWAP_SAMPLE_SIZE, len(train_cand_pool))
        )

        initial_abs_diff = abs(ratio_diff)

        # Evaluate potential swaps from the *sampled* candidates
        for _, test_cand in test_cands_sample.iterrows():
            for _, train_cand in train_cands_sample.iterrows():
                # Estimate new overall ratio after swap (same calculation as before)
                new_total = (
                    test_metrics["num_reports"]
                    - test_cand["total_reports"]
                    + train_cand["total_reports"]
                )
                new_fraud = (
                    test_metrics["num_fraud_reports"]
                    - test_cand["fraud_reports"]
                    + train_cand["fraud_reports"]
                )
                if new_total <= 0:
                    continue
                sim_ratio = new_fraud / new_total
                sim_abs_diff = abs(sim_ratio - target_overall_fraud_ratio)
                improvement = initial_abs_diff - sim_abs_diff

                if improvement > max_improvement:
                    max_improvement = improvement
                    best_swap = (test_cand["cik"], train_cand["cik"])

        # Perform the best swap if it improves the ratio
        if best_swap and max_improvement > 0:
            test_swap_cik, train_swap_cik = best_swap
            logging.info(
                f"    Performing swap: Test CIK {test_swap_cik} <-> Train CIK {train_swap_cik} (Ratio diff improvement: {max_improvement:.4f})"
            )
            current_test_ciks.remove(test_swap_cik)
            current_train_ciks.add(test_swap_cik)
            current_train_ciks.remove(train_swap_cik)
            current_test_ciks.add(train_swap_cik)
        else:
            logging.warning(
                "  No improving swap found in this iteration among sampled candidates. Stopping adjustment."
            )
            break

    if i == adjustment_iterations - 1 and abs(ratio_diff) > tolerance_overall_ratio:
        logging.warning(
            f"Reached max adjustment iterations ({adjustment_iterations}). Final overall ratio may deviate."
        )

    # --- Phase 4: Final Split & Reporting ---
    # (Finalization and Reporting logic remains the same as the previous simplified version)
    logging.info("\n--- Finalizing Split ---")
    final_test_ciks = current_test_ciks
    final_train_ciks = set(unique_companies["cik"]) - final_test_ciks

    overlap = final_train_ciks.intersection(final_test_ciks)
    assert len(overlap) == 0, f"FATAL: Overlap detected: {overlap}"

    final_train_df = df[df["cik"].isin(final_train_ciks)].copy().reset_index(drop=True)
    final_test_df = df[df["cik"].isin(final_test_ciks)].copy().reset_index(drop=True)

    logging.info("\n--- Final Split Evaluation ---")
    logging.info("Targets:")
    logging.info(f"  Target Test Companies: {target_test_companies_total}")
    logging.info(f"  Target Overall Fraud Ratio: {target_overall_fraud_ratio:.4f}")
    logging.info(f"  Target Companies per Sector: {target_companies_per_sector}")

    final_test_metrics = calculate_split_metrics(
        final_test_df,
        unique_companies[unique_companies["cik"].isin(final_test_ciks)],
        "Final Test Set",
    )
    final_train_metrics = calculate_split_metrics(
        final_train_df,
        unique_companies[unique_companies["cik"].isin(final_train_ciks)],
        "Final Train Set",
    )

    logging.info("\nFinal Test Set vs Targets:")
    logging.info(
        f"  Companies: {final_test_metrics['num_companies']} (Target: {target_test_companies_total}, Diff: {final_test_metrics['num_companies'] - target_test_companies_total:+d})"
    )
    logging.info(
        f"  Overall Fraud Ratio: {final_test_metrics['overall_fraud_ratio']:.4f} (Target: {target_overall_fraud_ratio:.4f}, Diff: {final_test_metrics['overall_fraud_ratio'] - target_overall_fraud_ratio:+.4f})"
    )
    logging.info(
        f"  --> Train Set Overall Fraud Ratio: {final_train_metrics['overall_fraud_ratio']:.4f}"
    )

    logging.info("  Sector Metrics (Achieved vs Target):")
    all_final_sectors = set(target_companies_per_sector.keys()) | set(
        final_test_metrics["company_distribution"].keys()
    )
    for sector in sorted(list(all_final_sectors)):
        achieved_co = final_test_metrics["company_distribution"].get(sector, 0)
        target_co = target_companies_per_sector.get(sector, 0)
        achieved_fraud_reports = final_test_metrics["sector_fraud_report_counts"].get(
            sector, 0
        )
        target_fraud_reports = target_fraud_reports_per_sector.get(sector, 0)
        logging.info(f"    {sector}:")
        logging.info(
            f"      Companies: {achieved_co} (Target: {target_co}, Diff: {achieved_co - target_co:+d})"
        )
        logging.info(
            f"      Fraud Reports: {achieved_fraud_reports} (Target: {target_fraud_reports}, Diff: {achieved_fraud_reports - target_fraud_reports:+d})"
        )

    final_train_df = final_train_df.sample(frac=1).reset_index(drop=True)

    return final_train_df, final_test_df


def generate_k_folds_subsets_by_cik(index_df, k):
    """
    Generate a list with n subsets :  [index_df1 index_df2 ... index_dfk]
    The fraud non fraud ratio should be preseverd in each subset.
    There should be no overlap company overlap between the subsets.
    return a dict:
    {
        "subsets":{
            "1": index_df1,
            "2": index_df2,
            ...
            "k": index_dfk
        }
        "subsets_stats":{
            "1": {
                "num_reports": 100,
                "num_fraud_samples,
                "num_non_fraud_samples,
                "distribution per industrion,
                "distribution per industry_non fraud
                "distribution per industry_fraud.

            }
        }
    }
    """
    logging.info(f"Generating {k} folds with CIK-based splitting (no company overlap)")

    # Ensure required columns exist
    required_cols = ["cik", "is_fraud", "sicagg"]
    missing_cols = [col for col in required_cols if col not in index_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create a copy to avoid modifying the original DataFrame
    df = index_df.copy()

    # Group by CIK and calculate metrics per company
    company_metrics = (
        df.groupby("cik")
        .agg(
            total_reports=("report_id", "count"),
            fraud_reports=("is_fraud", "sum"),
            sicagg=("sicagg", "first"),
        )
        .reset_index()
    )

    company_metrics["fraud_ratio"] = (
        company_metrics["fraud_reports"] / company_metrics["total_reports"]
    )
    company_metrics["non_fraud_reports"] = (
        company_metrics["total_reports"] - company_metrics["fraud_reports"]
    )

    # Sort companies by fraud ratio (descendingly) and then by total_reports
    # This helps with stratification when distributing companies to folds
    company_metrics = company_metrics.sort_values(
        by=["fraud_ratio", "total_reports", "sicagg"], ascending=[False, False, True]
    )

    # Initialize folds
    folds = {str(i + 1): [] for i in range(k)}
    fold_metrics = {
        str(i + 1): {"fraud_reports": 0, "non_fraud_reports": 0, "total_reports": 0}
        for i in range(k)
    }

    # Distribute companies across folds using round-robin approach
    # This helps maintain similar fraud ratios across folds
    for idx, row in company_metrics.iterrows():
        # Find the fold with minimum total reports
        min_reports_fold = min(
            fold_metrics, key=lambda x: fold_metrics[x]["total_reports"]
        )

        folds[min_reports_fold].append(row["cik"])
        fold_metrics[min_reports_fold]["fraud_reports"] += row["fraud_reports"]
        fold_metrics[min_reports_fold]["non_fraud_reports"] += row["non_fraud_reports"]
        fold_metrics[min_reports_fold]["total_reports"] += row["total_reports"]

    # Create DataFrames for each fold
    subsets = {}
    subsets_stats = {}

    for fold_id, ciks in folds.items():
        # Get data for this fold
        fold_df = df[df["cik"].isin(ciks)].copy().reset_index(drop=True)
        subsets[fold_id] = fold_df

        # Calculate detailed statistics
        num_reports = len(fold_df)
        num_fraud = fold_df["is_fraud"].sum()
        num_non_fraud = num_reports - num_fraud

        # Industry distribution
        industry_dist = fold_df["sicagg"].value_counts().to_dict()

        # Fraud distribution by industry
        industry_fraud = (
            fold_df[fold_df["is_fraud"] == 1]["sicagg"].value_counts().to_dict()
        )

        # Non-fraud distribution by industry
        industry_non_fraud = (
            fold_df[fold_df["is_fraud"] == 0]["sicagg"].value_counts().to_dict()
        )

        # Store statistics
        subsets_stats[fold_id] = {
            "num_reports": num_reports,
            "num_fraud_samples": int(num_fraud),
            "num_companies": fold_df["cik"].nunique(),
            "num_non_fraud_samples": int(num_non_fraud),
            "fraud_ratio": float(num_fraud / num_reports if num_reports > 0 else 0),
            "distribution_per_industry": industry_dist,
            "distribution_per_industry_fraud": industry_fraud,
            "distribution_per_industry_non_fraud": industry_non_fraud,
        }

    # Log summary statistics
    logging.info(f"K-folds generation complete. Summary stats:")
    for fold_id, stats in subsets_stats.items():
        logging.info(
            f"Fold {fold_id}: {stats['num_reports']} reports, "
            f"fraud ratio: {stats['fraud_ratio']:.4f}, "
            f"unique companies: {len(subsets[fold_id]['cik'].unique())}"
        )

    return {"subsets": subsets, "subsets_stats": subsets_stats}


def generate_k_folds_random(
    index_df,
    k,
):
    """
    Generate k random folds from the dataset.
    Unlike the CIK-based splitting, this method does not guarantee company separation between folds.

    Args:
        index_df (pd.DataFrame): Input dataframe with reports
        k (int): Number of folds to generate


    Returns:
        dict: Dictionary with subsets and their statistics in the same format as generate_k_folds_subsets_by_cik
    """
    logging.info(f"Generating {k} folds with random splitting")

    # Create a copy to avoid modifying the original DataFrame
    df = index_df.copy()

    # Ensure we have report_id column for indexing
    if "report_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "report_id"})
        logging.warning("Added 'report_id' column based on index.")

    # Use scikit-learn's KFold for random splitting
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True)

    # Create folds
    subsets = {}
    subsets_stats = {}

    for i, (_, test_idx) in enumerate(kf.split(df)):
        fold_id = str(i + 1)
        fold_df = df.iloc[test_idx].copy().reset_index(drop=True)
        subsets[fold_id] = fold_df

        # Calculate statistics (same as in generate_k_folds_subsets_by_cik)
        num_reports = len(fold_df)
        num_fraud = fold_df["is_fraud"].sum()
        num_non_fraud = num_reports - num_fraud

        # Industry distribution
        industry_dist = (
            fold_df["sicagg"].value_counts().to_dict()
            if "sicagg" in fold_df.columns
            else {}
        )

        # Fraud distribution by industry
        industry_fraud = (
            fold_df[fold_df["is_fraud"] == 1]["sicagg"].value_counts().to_dict()
            if "sicagg" in fold_df.columns
            else {}
        )

        # Non-fraud distribution by industry
        industry_non_fraud = (
            fold_df[fold_df["is_fraud"] == 0]["sicagg"].value_counts().to_dict()
            if "sicagg" in fold_df.columns
            else {}
        )

        # Store statistics
        subsets_stats[fold_id] = {
            "num_reports": num_reports,
            "num_fraud_samples": int(num_fraud),
            "num_non_fraud_samples": int(num_non_fraud),
            "fraud_ratio": float(num_fraud / num_reports if num_reports > 0 else 0),
            "distribution_per_industry": industry_dist,
            "distribution_per_industry_fraud": industry_fraud,
            "distribution_per_industry_non_fraud": industry_non_fraud,
        }

    # Log summary statistics
    logging.info(f"K-folds generation complete. Summary stats:")
    for fold_id, stats in subsets_stats.items():
        logging.info(
            f"Fold {fold_id}: {stats['num_reports']} reports, "
            f"fraud ratio: {stats['fraud_ratio']:.4f}"
        )
        if "cik" in df.columns:
            unique_companies = len(subsets[fold_id]["cik"].unique())
            logging.info(f"  Unique companies in fold {fold_id}: {unique_companies}")

    return {"subsets": subsets, "subsets_stats": subsets_stats}


def split_dataset_by_cik_label_agnostic(df, test_size=0.1):
    """
    Split a dataset by company (CIK) without considering label distribution.
    This ensures strict CIK separation between train and test sets.

    Args:
        df (pd.DataFrame): DataFrame containing 'cik' column
        test_size (float): Proportion of data to use for validation/test
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df) - Split DataFrames
    """
    logging.info(
        f"Creating label-agnostic validation split based on CIK with test_size={test_size}"
    )

    # Get unique companies
    unique_ciks = df["cik"].unique()

    # Split companies randomly
    n_test = int(len(unique_ciks) * test_size)

    # Shuffle and split
    np.random.shuffle(unique_ciks)
    test_ciks = set(unique_ciks[:n_test])
    train_ciks = set(unique_ciks[n_test:])

    # Create train and validation dataframes
    train_df = df[df["cik"].isin(train_ciks)].copy().reset_index(drop=True)
    val_df = df[df["cik"].isin(test_ciks)].copy().reset_index(drop=True)

    # Log statistics
    logging.info(f"Total companies: {len(unique_ciks)}")
    logging.info(f"Train set: {len(train_df)} samples, {len(train_ciks)} companies")
    logging.info(f"Validation set: {len(val_df)} samples, {len(test_ciks)} companies")

    # Log fraud statistics if available
    if "is_fraud" in df.columns:
        train_fraud = len(train_df[train_df.is_fraud == True])
        val_fraud = len(val_df[val_df.is_fraud == True])
        train_fraud_ratio = train_fraud / len(train_df) if len(train_df) > 0 else 0
        val_fraud_ratio = val_fraud / len(val_df) if len(val_df) > 0 else 0

        logging.info(f"Train fraud ratio: {train_fraud_ratio:.2%}")
        logging.info(f"Validation fraud ratio: {val_fraud_ratio:.2%}")

    # Shuffle the training data to prevent any order biases
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    return train_df, val_df


def split_dataset_by_time(df, keep_fraud_ratio=False, test_size=0.1,seed=None):
    """
    Split a dataset by time, ensuring all train samples are anterior to test samples,
    with fraud ratio in test matching the original dataset's fraud ratio.

    Args:
        df (pd.DataFrame): DataFrame containing 'year', 'quarter' and 'is_fraud' columns
        test_size (float): Proportion of data to use for test (approximate)

    Returns:
        tuple: (train_df, val_df) - Split DataFrames
    """
    if seed is not None:
        logging.info(f"Seeding random number generator with seed: {seed}")
        #Seed the random number generator for reproducibility
        random.seed(seed)
        np.random.seed(seed)
    
    logging.info("Creating test or val split from training data based on time.")

    # Create a time index for sorting (e.g., 2020Q1 -> 20201)
    df = df.copy()
    if isinstance(df["quarter"].iloc[0], str):
        df["time_idx"] = df["year"].astype(int) * 10 + df["quarter"].apply(
            lambda x: int(x[-1])
        )
    else:
        df["time_idx"] = df["year"].astype(int) * 10 + df["quarter"].astype(int)

    # Sort by time value
    df = df.sort_values("time_idx")

    # Identify unique time periods
    time_periods = (
        df.groupby("time_idx")[["is_fraud", "cik"]]
        .agg(
            {
                "is_fraud": "sum",  # Count of fraud cases per period
                "cik": "count",  # Total count of records per period
            }
        )
        .reset_index()
    )
    time_periods.rename(
        columns={"is_fraud": "fraud_count", "cik": "total_count"}, inplace=True
    )
    time_periods = time_periods.sort_values("time_idx")

    # Ensure 'fraud_count' exists even if there are no fraud cases
    if "fraud_count" not in time_periods.columns:
        time_periods["fraud_count"] = 0

    # Calculate cumulative counts
    time_periods["cum_total"] = time_periods["total_count"].cumsum()
    time_periods["cum_fraud"] = time_periods["fraud_count"].cumsum()

    # Calculate original dataset fraud ratio
    total_records = time_periods["total_count"].sum()
    total_fraud = time_periods["fraud_count"].sum()
    original_fraud_ratio = total_fraud / total_records if total_records > 0 else 0

    # Find the cutoff point based on target test size
    target_train_size = int((1 - test_size) * total_records)
    cutoff_idx = time_periods[time_periods["cum_total"] >= target_train_size].iloc[0][
        "time_idx"
    ]

    # Split the data
    train_df = df[df["time_idx"] < cutoff_idx].copy()
    test_df = df[df["time_idx"] >= cutoff_idx].copy()

    # Check if validation set is empty
    if test_df.empty:
        logging.warning("Validation set is empty. Using a different cutoff point.")
        cutoff_idx = time_periods.iloc[-2]["time_idx"]
        train_df = df[df["time_idx"] < cutoff_idx].copy()
        test_df = df[df["time_idx"] >= cutoff_idx].copy()

    # Calculate fraud ratio in validation set
    test_fraud_count = test_df["is_fraud"].sum()
    test_size = len(test_df)
    test_fraud_ratio = test_fraud_count / test_size if test_size > 0 else 0

    logging.info(
        f"Initial validation set: {test_size} samples, {test_fraud_count} fraud, ratio: {test_fraud_ratio:.4f}"
    )
    logging.info(f"Original dataset fraud ratio: {original_fraud_ratio:.4f}")

    # Downsample non-fraud samples in validation set to match original fraud ratio
    tolerance = 0.01
    if test_fraud_ratio < original_fraud_ratio - tolerance and keep_fraud_ratio:
        # Calculate number of non-fraud samples needed to achieve target ratio
        # If original_ratio = fraud / (fraud + non_fraud), then:
        # non_fraud = fraud * (1 - original_ratio) / original_ratio
        target_non_fraud_count = int(
            test_fraud_count * (1 - original_fraud_ratio) / original_fraud_ratio
        )
        current_non_fraud_count = test_size - test_fraud_count

        if target_non_fraud_count < current_non_fraud_count:
            # Separate fraud and non-fraud samples
            test_fraud_df = test_df[test_df["is_fraud"] == True]
            test_non_fraud_df = test_df[test_df["is_fraud"] == False]

            # Use the grouped sampling strategy like in base_preprocessing.py
            sampled_non_fraud = group_sample_non_fraud(
                test_non_fraud_df, target_non_fraud_count
            )

            # Combine back
            test_df = pd.concat([test_fraud_df, sampled_non_fraud])
            logging.info(
                f"Downsampled test non-fraud from {current_non_fraud_count} to {len(sampled_non_fraud)} samples"
            )

            # Calculate new fraud ratio
            new_val_fraud_ratio = test_fraud_count / len(test_df)
            logging.info(f"New test(Val) fraud ratio: {new_val_fraud_ratio:.4f}")

    # Drop the time index
    train_df = train_df.drop("time_idx", axis=1)
    test_df = test_df.drop("time_idx", axis=1)

    # Calculate final statistics
    train_size = len(train_df)
    test_size = len(test_df)
    train_fraud = train_df["is_fraud"].sum()
    test_fraud = test_df["is_fraud"].sum()

    actual_test_size = test_size / (train_size + test_size)
    test_fraud_ratio = test_fraud / test_size if test_size > 0 else 0

    logging.info(f"Time split cutoff: period index {cutoff_idx}")
    logging.info(
        f"Target test(val)_size: {test_size:.2%}, Actual: {actual_test_size:.2%}"
    )
    logging.info(
        f"Train set: {train_size} rows, {train_fraud} fraud ({train_fraud/train_size:.2%})"
    )
    logging.info(
        f"Test(Val) set: {test_size} rows, {test_fraud} fraud ({test_fraud/test_size:.2%})"
    )

    # Check company distribution
    train_ciks = set(train_df["cik"])
    test_ciks = set(test_df["cik"])
    overlap_ciks = train_ciks.intersection(test_ciks)
    overlap_percentage = len(overlap_ciks) / len(train_ciks.union(test_ciks))

    logging.info(
        f"Train set: data from {len(train_ciks)} companies: Period : {train_df['year'].min()}-{train_df['year'].max()}"
    )
    logging.info(
        f"Test(Val) set: data from {len(test_ciks)} companies. Period : {test_df['year'].min()}-{test_df['year'].max()}"
        f"(Num fraud test companies: {test_df.query('is_fraud==True').cik.nunique()}"
    )
    logging.info(
        f"CIK overlap: {len(overlap_ciks)} companies ({overlap_percentage:.2%} of all companies)"
    )

    # Shuffle the training data to prevent any order biases during training
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def group_sample_non_fraud(non_fraud_df, target_size):
    """
    Sample non-fraud cases while preserving the distribution of subgroups defined by 'year' and 'sicagg'.

    Args:
        non_fraud_df: DataFrame containing only non-fraud cases
        target_size: Target number of non-fraud samples to select

    Returns:
        DataFrame of sampled non-fraud cases
    """
    logging.info(f"Sampling non-fraud cases, preserving year and sicagg distribution")

    if "report_id" not in non_fraud_df.columns:
        non_fraud_df = non_fraud_df.reset_index().rename(columns={"index": "report_id"})
        logging.debug("Added 'report_id' column based on index.")

    sampled_non_fraud_list = []
    total_sampled = 0

    # Get group sizes in the original data
    group_sizes = non_fraud_df.groupby(["year", "sicagg"]).size()
    total_records = len(non_fraud_df)

    # Calculate downsampling factor
    downsampling_factor = target_size / total_records

    # Iterate through each group
    for (year, sicagg), group_size in group_sizes.items():
        # Calculate the target size for this group proportionally
        group_target_size = int(group_size * downsampling_factor)
        group_target_size = max(
            1, min(group_target_size, group_size)
        )  # Ensure at least 1 and not more than available

        # Sample from the group
        group_data = non_fraud_df[
            (non_fraud_df["year"] == year) & (non_fraud_df["sicagg"] == sicagg)
        ]
        if not group_data.empty:
            sampled_group = group_data.sample(
                n=group_target_size, random_state=SEED_TRAINING
            )
            sampled_non_fraud_list.append(sampled_group)
            total_sampled += len(sampled_group)

    # If we didn't get enough samples, take more from the largest groups
    if total_sampled < target_size and sampled_non_fraud_list:
        remaining_needed = target_size - total_sampled
        # Get groups with remaining samples
        remaining_groups = []
        for (year, sicagg), group_size in group_sizes.items():
            current_sampled = len(
                sampled_non_fraud_list[0][
                    (sampled_non_fraud_list[0]["year"] == year)
                    & (sampled_non_fraud_list[0]["sicagg"] == sicagg)
                ]
            )
            if current_sampled < group_size:
                remaining_groups.append((year, sicagg, group_size - current_sampled))

        # Sort by remaining capacity (descending)
        remaining_groups.sort(key=lambda x: x[2], reverse=True)

        # Take additional samples from each group until we reach the target
        for year, sicagg, remaining in remaining_groups:
            if remaining_needed <= 0:
                break

            # How many more to take from this group
            to_take = min(remaining, remaining_needed)

            # Get samples that weren't already selected
            already_sampled = set()
            for df in sampled_non_fraud_list:
                group_samples = df[(df["year"] == year) & (df["sicagg"] == sicagg)][
                    "report_id"
                ].tolist()
                already_sampled.update(group_samples)

            group_data = non_fraud_df[
                (non_fraud_df["year"] == year)
                & (non_fraud_df["sicagg"] == sicagg)
                & (~non_fraud_df["report_id"].isin(already_sampled))
            ]

            if len(group_data) > 0:
                to_take = min(to_take, len(group_data))
                additional_samples = group_data.sample(
                    n=to_take, random_state=SEED_TRAINING
                )
                sampled_non_fraud_list.append(additional_samples)
                total_sampled += to_take
                remaining_needed -= to_take

    # Concatenate all sampled groups
    if sampled_non_fraud_list:
        sampled_non_fraud = pd.concat(sampled_non_fraud_list)
        logging.info(
            f"Sampled {len(sampled_non_fraud)} non-fraud cases using group-based strategy"
        )
        return sampled_non_fraud
    else:
        logging.warning(
            "No non-fraud samples were selected. Returning empty DataFrame."
        )
        return pd.DataFrame(columns=non_fraud_df.columns)


# def load_cross_validation_path(config):
#     dataset_version = config.get("dataset_version", "company_isolated_splitting")
#     assert dataset_version in ["company_isolated_splitting", "time_splitting", "time_splitting"]
#     train_path = PREPROCESSED_PATH_EXTENDED / dataset_version / "train_index.csv"
#     test_path = PREPROCESSED_PATH_EXTENDED / dataset_version / "test_index.csv"

#     return train_path, test_path


def load_cross_validation_path(config):
    fold_id = config["fold_id"]
    dataset_version = config.get("dataset_version", "company_isolated_splitting")

    test_path = (
        PREPROCESSED_PATH_EXTENDED / dataset_version / f"fold_{fold_id}" / "test.csv"
    )
    train_path = (
        PREPROCESSED_PATH_EXTENDED / dataset_version / f"fold_{fold_id}" / "train.csv"
    )

    assert train_path.exists(), f"Train path {train_path} does not exist"
    assert test_path.exists(), f"Test path {test_path} does not exist"
    return train_path, test_path


def load_full_data_path(config):
    dataset_version = config.get("dataset_version", "company_isolated_splitting")
    full_data_path = PREPROCESSED_PATH_EXTENDED / dataset_version / "all_index.csv"
    return full_data_path


def get_train_test_splitter(config):
    if config.get("random_split", False):
        logging.info("Using random train/val split")
        return split_dataset_randomly

    dataset_version = config.get("dataset_version", "v3")
    if dataset_version in ["company_isolated_splitting"]:
        splitter = split_dataset_by_cik
    elif dataset_version in ["time_splitting"]:
        splitter = split_dataset_by_time
    elif dataset_version in ["time_splitting"]:
        splitter = split_dataset_randomly

    else:
        raise Exception(
            f"Dataset version {dataset_version} not supported. No data splitting method."
        )
    return splitter


def load_numpy_dataset(
    dataset_version,
    train_path=None,
    test_path=None,
    val_path=None,
    full_financial_path=None,
    fold_id=None,
    return_raw_data=False,
):
    """
    Generic dataset loader for LightGBM models.

    Args:
        dataset_version (str): Version of the dataset (e.g., 'v4', 'v5', 'v6', 'v6')
        train_path (Path, optional): Path to training data. If None, uses default path.
        test_path (Path, optional): Path to test data. If None, uses default path.
        val_path (Path, optional): Path to validation data. If None, creates validation from train.
        full_financial_path (Path, optional): Path to full financial data. If None, uses default path.
        fold_id (int, optional): Current fold ID for cross-validation.
        n_folds (int, optional): Number of folds for cross-validation.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    """
    logging.info(f"Loading dataset version: {dataset_version}")

    # Determine paths based on dataset version
    if train_path is None or test_path is None:
        train_path, test_path = load_cross_validation_path(
            {
                "dataset_version": dataset_version,
                "cross_validation": True,
                "fold_id": fold_id,
                "dataset_name": dataset_version.replace("_kfolds", ""),
            }
        )

    # Determine feature set based on dataset version
    if dataset_version.startswith(("company_isolated_splitting", "time_splitting", "time_splitting")):
        from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
            EXTENDED_FINANCIAL_FEATURES,
            EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
        )

        features = EXTENDED_FINANCIAL_FEATURES + ["sicagg"]
        count_cols = EXTENDED_FINANCIAL_FEATURES_COUNT_COLS

        if full_financial_path is None:
            from researchpkg.anomaly_detection.config import FINANCIALS_DIR_EXTENDED

            full_financial_path = (
                FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
            )
    else:
        raise ValueError(f"Unsupported dataset version: {dataset_version}")

    # Load full financial data
    logging.info(f"Loading full financials data from {full_financial_path}")
    full_df = pd.read_csv(full_financial_path)
    full_df = full_df[["cik", "year", "quarter"] + features]

    # Helper function to merge financial data
    def merge_with_financials(df):
        # Drop count columns from the dataset if they exist
        df = df.drop(columns=count_cols + ["sicagg"])
        # Merge with full financial data
        df = df.merge(full_df, on=["cik", "year", "quarter"], how="left")
        return df

    # Load training data
    logging.info(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    train_df = merge_with_financials(train_df)

    assert "sicagg_x" not in train_df.columns
    assert "sicagg" in train_df.columns

    # Handle validation data
    if val_path is not None:
        logging.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path)
        val_df = merge_with_financials(val_df)
    else:
        # Create validation split from training data
        splitter = get_train_test_splitter({"dataset_version": dataset_version})
        train_df, val_df = splitter(train_df, test_size=0.1)

    # Load test data
    logging.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    test_df = merge_with_financials(test_df)

    # Clean up memory
    del full_df

    if return_raw_data:
        return train_df, val_df, test_df

    # Define target variables
    y_train = train_df["is_fraud"].astype(int)
    y_val = val_df["is_fraud"].astype(int)
    y_test = test_df["is_fraud"].astype(int)

    feature_cols = [col for col in features if col in train_df.columns]
    assert "sicagg" in feature_cols

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    logging.info(f"Selected {len(feature_cols)} features for training")

    # Handle missing values
    for name, data in [("train", X_train), ("validation", X_val), ("test", X_test)]:
        if data.isna().sum().sum() > 0:
            logging.warning(
                f"Found {data.isna().sum().sum()} missing values in {name} data. Filling with zeros."
            )
            data.fillna(0, inplace=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols



def load_dechow_numpy_dataset( dataset_version,
    train_path=None,
    test_path=None,
    val_path=None,
    full_financial_path=None,
    fold_id=None,
    return_raw_data=False):
    """
    Load dataset with financial features proposed by Patricia Dechow, 2011.
    - WC_acc : DIFF WC_ACCRUALS / Average
    - rsst_acc
    - ch_rec : Changes in receivables DIFF_RECEIVABLES
    - ch_inv : DIFF_INVENTORIES
    - soft_assets: (ASSETS - PROPERTY_PLANT_EQUIPMENT_NET_TAG - CASH_TAG)/ ASSETS
    - Modified Jones discretionnary accruals: \alpha + \↓eta (1/starts_assets)+ \gamma(\delta sales - \delta recv)/start_assets+
    \rho(Delta PPE/ Start assets) + \epsilon. Epsilon is the discretionary accurals.

    """









def calculate_cik_macro_f1(true_labels, predicted_labels, ciks, average="macro"):
    """
    Calculates the CIK Macro F1 score.

    This is the unweighted average of the F1 scores calculated per unique CIK.

    Args:
        predicted_labels (list): List of predicted labels.
        ciks (list): List of CIKs corresponding to each sample.

    Returns:
        float: The CIK Macro F1 score. Returns 0 if no samples are provided.
    """

    assert average in [
        "macro",
        "weighted",
    ], "average must be either 'macro' or 'weighted'"
    if true_labels is None:
        return 0.0

    # Create a DataFrame for easy grouping
    df = pd.DataFrame({"cik": ciks, "y_true": true_labels, "y_pred": predicted_labels})

    cik_f1_scores = []

    # Group by CIK and calculate F1 for each group
    for cik, group_df in df.groupby("cik"):
        # Ensure there's more than one class for F1 calculation within the group
        if (
            len(group_df["y_true"].unique()) > 1
            or "Fraud" in group_df["y_true"].unique()
        ):
            # Calculate F1 for the current CIK
            # zero_division=0 ensures that if a group has no true positives
            # or no predicted positives, the F1 is calculated as 0.
            f1 = f1_score(
                group_df["y_true"],
                group_df["y_pred"],
                pos_label="YES",
                zero_division=0,
            )
            if average == "weighted":
                group_size = len(group_df)
                # Normalize the F1 score by the number of samples in the group
                f1 = f1 * (group_size / len(df))
            cik_f1_scores.append(f1)
        else:

            f1 = f1_score(
                group_df["y_true"],
                group_df["y_pred"],
                pos_label="YES",
                zero_division=0,
            )
            if average == "weighted":
                group_size = len(group_df)
                # Normalize the F1 score by the number of samples in the group
                f1 = f1 * (group_size / len(df))
            cik_f1_scores.append(f1)

    if not cik_f1_scores:
        # This case should ideally not happen if true_labels is not empty,
        # unless all groups had issues, but as a safeguard:
        logging.warning("No CIK F1 scores calculated.")
        return 0.0

    # Calculate the unweighted average of CIK F1 scores
    if average == "macro":
        f1 = np.mean(cik_f1_scores)
    else:
        # Sum of weighted F1 scores because we already normalized them
        f1 = np.sum(cik_f1_scores)
    return f1


def update_experiment_dir_with_cik_macro(experiment_dir: Path):
    """
    Updates all metrics JSON files in an experiment directory with CIK macro F1 scores.

    For each epoch's prediction CSV file, calculates the CIK macro F1 score and adds
    it to the corresponding metrics JSON file.

    Args:
        experiment_dir (Path): Path to the experiment directory containing metrics and prediction files

    Returns:
        int: Number of metrics files updated
    """
    logging.info(f"Updating metrics in {experiment_dir} with CIK macro F1 scores")

    # Find all validation prediction CSVs
    val_pred_files = list(experiment_dir.glob("val_predictions_epoch_*.csv"))

    if not val_pred_files:
        logging.warning(f"No validation prediction files found in {experiment_dir}")
        return 0

    count_updated = 0

    for pred_file in val_pred_files:
        # Extract epoch number from filename
        epoch_num = int(pred_file.stem.split("_")[-1])
        metrics_file = experiment_dir / f"metrics_epoch_{epoch_num}.json"

        if not metrics_file.exists():
            logging.warning(
                f"Metrics file not found for epoch {epoch_num}: {metrics_file}"
            )
            continue

        # Load prediction data
        try:
            pred_df = pd.read_csv(pred_file)
            true_labels = pred_df["y_true"].tolist()
            predicted_labels = pred_df["y_pred"].tolist()
            ciks = pred_df["cik"].tolist()

            # Calculate CIK macro F1
            cik_macro_f1_macro = calculate_cik_macro_f1(
                true_labels, predicted_labels, ciks
            )

            # Update metrics file
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Add or update the CIK macro F1 value
            metrics["cik_macro_f1_macro"] = float(cik_macro_f1_macro)

            # Calculate the CIK macro F1 score weighted by the number of samples
            cik_macro_f1_weighted = calculate_cik_macro_f1(
                true_labels, predicted_labels, ciks, average="weighted"
            )
            metrics["cik_macro_f1_weighted"] = float(cik_macro_f1_weighted)

            # Write updated metrics back to file
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            logging.info(
                f"Updated metrics for epoch {epoch_num} with CIK macro F1 Macro:"
                f" {cik_macro_f1_macro:.4f} and weighted: {cik_macro_f1_weighted:.4f}"
            )
            count_updated += 1

        except Exception as e:
            logging.error(f"Error processing {pred_file}: {str(e)}")

    # Also update the test metrics if available
    test_pred_file = experiment_dir / "test_predictions.csv"
    test_metrics_file = experiment_dir / "test_metrics.json"

    if test_pred_file.exists() and test_metrics_file.exists():
        try:
            pred_df = pd.read_csv(test_pred_file)
            true_labels = pred_df["y_true"].tolist()
            predicted_labels = pred_df["y_pred"].tolist()
            ciks = pred_df["cik"].tolist()

            # Calculate CIK macro F1
            cik_macro_f1_macro = calculate_cik_macro_f1(
                true_labels, predicted_labels, ciks
            )

            # Update metrics file
            with open(test_metrics_file, "r") as f:
                metrics = json.load(f)

            # Add or update the CIK macro F1 value
            metrics["cik_macro_f1"] = float(cik_macro_f1_macro)

            # Write updated metrics back to file
            with open(test_metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            logging.info(
                f"Updated test metrics with CIK macro F1: {cik_macro_f1_macro:.4f}"
            )
            count_updated += 1

        except Exception as e:
            logging.error(f"Error processing test predictions: {str(e)}")

    logging.info(f"Updated {count_updated} metrics files with CIK macro F1 scores")
    return count_updated



