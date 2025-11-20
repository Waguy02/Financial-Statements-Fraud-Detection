import json
import logging
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from wordcloud import WordCloud
import yaml
from matplotlib.pylab import Enum
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_DECHOW,
    FINANCIALS_DIR_EXTENDED,
    LIST_MISTATEMENT_TYPE_FOR_TRAINING,
    PREPROCESSED_PATH_EXTENDED,
    SEED_TRAINING,
    SIC_INDEX_FILE,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_dechow import (
    DECHOW_FEATURES,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
)

DECHOW_FIN_PATH = FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv"
FINANCIALS_FIN_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"


def get_last_checkpoint(checkpoint_dir: Path):
    checkpoints = [
        d
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint")
    ]
    logging.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
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


def split_dataset_randomly(df, test_size=0.1, seed=None):
    if seed is not None:
        logging.info(f"Seeding random number generator with seed: {seed}")
        # Seed the random number generator for reproducibility
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


def generate_k_folds_subsets_by_cik(index_df, k, refinement_iterations=100):
    """
    Generate a list with n subsets :  [index_df1 index_df2 ... index_dfk]
    The fraud non fraud ratio should be preseverd in each subset.
    There should be no overlap company overlap between the subsets.
    This version includes an iterative refinement process to balance misstatement types.
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
    logging.info(
        f"Generating {k} folds with CIK-based splitting and misstatement balancing."
    )

    # Ensure required columns exist
    required_cols = ["cik", "is_fraud", "sicagg"] + LIST_MISTATEMENT_TYPE_FOR_TRAINING
    missing_cols = [col for col in required_cols if col not in index_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create a copy to avoid modifying the original DataFrame
    df = index_df.copy()

    # --- Company-level aggregation ---
    agg_dict = {
        "total_reports": ("report_id", "count"),
        "fraud_reports": ("is_fraud", "sum"),
        "sicagg": ("sicagg", "first"),
    }
    for mis_col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
        agg_dict[mis_col] = (mis_col, "sum")

    company_metrics = df.groupby("cik").agg(**agg_dict).reset_index()

    company_metrics["fraud_ratio"] = (
        company_metrics["fraud_reports"] / company_metrics["total_reports"]
    )
    company_metrics["non_fraud_reports"] = (
        company_metrics["total_reports"] - company_metrics["fraud_reports"]
    )

    company_metrics = company_metrics.sort_values(
        by=["fraud_ratio", "total_reports", "sicagg"], ascending=[False, False, True]
    )

    # --- Initial Fold Creation (Round-Robin) ---
    folds_ciks = {str(i + 1): [] for i in range(k)}
    fold_report_counts = {str(i + 1): 0 for i in range(k)}

    for _, company in company_metrics.iterrows():
        min_reports_fold = min(fold_report_counts, key=lambda x: fold_report_counts[x])
        folds_ciks[min_reports_fold].append(company["cik"])
        fold_report_counts[min_reports_fold] += company["total_reports"]

    logging.info("Initial folds created. Starting refinement process...")

    # --- Iterative Refinement for Misstatement Distribution ---
    # 1. Calculate global target distribution
    total_reports_global = company_metrics["total_reports"].sum()
    target_dist = {
        col: company_metrics[col].sum() / total_reports_global
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING
    }

    # Helper to calculate cost (sum of squared errors from target)
    def calculate_cost(fold_ciks_list, company_metrics_df):
        if not fold_ciks_list:
            return float("inf")
        fold_companies = company_metrics_df[
            company_metrics_df["cik"].isin(fold_ciks_list)
        ]
        total_reports_fold = fold_companies["total_reports"].sum()
        if total_reports_fold == 0:
            return float("inf")
        cost = 0
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
            current_ratio = fold_companies[col].sum() / total_reports_fold
            cost += (current_ratio - target_dist[col]) ** 2
        return cost

    for i in tqdm(
        range(refinement_iterations), desc="Refining folds for misstatement balance"
    ):
        # Calculate current distributions and costs for all folds
        fold_costs = {
            f_id: calculate_cost(ciks, company_metrics)
            for f_id, ciks in folds_ciks.items()
        }

        # Find the worst feature to balance
        max_dev = -1
        worst_col = None
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
            ratios = []
            for f_id, ciks in folds_ciks.items():
                fold_comps = company_metrics[company_metrics["cik"].isin(ciks)]
                total_reports = fold_comps["total_reports"].sum()
                if total_reports > 0:
                    ratios.append(fold_comps[col].sum() / total_reports)
            if ratios:
                dev = np.std(ratios)
                if dev > max_dev:
                    max_dev = dev
                    worst_col = col

        if worst_col is None:
            break  # No columns to improve

        # Find most over- and under-represented folds for this feature
        ratios_for_worst_col = {}
        for f_id, ciks in folds_ciks.items():
            fold_comps = company_metrics[company_metrics["cik"].isin(ciks)]
            total_reports = fold_comps["total_reports"].sum()
            ratios_for_worst_col[f_id] = (
                (fold_comps[worst_col].sum() / total_reports)
                if total_reports > 0
                else 0
            )

        over_fold_id = max(ratios_for_worst_col, key=ratios_for_worst_col.get)
        under_fold_id = min(ratios_for_worst_col, key=ratios_for_worst_col.get)

        if over_fold_id == under_fold_id:
            continue  # Cannot improve

        # Find best swap between these two folds
        best_swap = None
        max_cost_reduction = 0
        initial_cost = fold_costs[over_fold_id] + fold_costs[under_fold_id]

        over_ciks = company_metrics[
            company_metrics["cik"].isin(folds_ciks[over_fold_id])
        ]
        under_ciks = company_metrics[
            company_metrics["cik"].isin(folds_ciks[under_fold_id])
        ]

        # Find candidates for swapping
        over_cands = over_ciks[over_ciks[worst_col] > 0].sort_values(
            worst_col, ascending=False
        )
        under_cands = under_ciks[under_ciks[worst_col] == 0]

        if over_cands.empty or under_cands.empty:
            continue

        # Limit search space for performance
        for _, over_cand in over_cands.head(10).iterrows():
            for _, under_cand in under_cands.head(10).iterrows():
                # Simulate swap
                temp_over_ciks = folds_ciks[over_fold_id][:]
                temp_under_ciks = folds_ciks[under_fold_id][:]
                temp_over_ciks.remove(over_cand["cik"])
                temp_over_ciks.append(under_cand["cik"])
                temp_under_ciks.remove(under_cand["cik"])
                temp_under_ciks.append(over_cand["cik"])

                new_cost = calculate_cost(
                    temp_over_ciks, company_metrics
                ) + calculate_cost(temp_under_ciks, company_metrics)
                cost_reduction = initial_cost - new_cost

                if cost_reduction > max_cost_reduction:
                    max_cost_reduction = cost_reduction
                    best_swap = (over_cand["cik"], under_cand["cik"])

        # Perform the best swap
        if best_swap and max_cost_reduction > 0:
            over_cik, under_cik = best_swap
            folds_ciks[over_fold_id].remove(over_cik)
            folds_ciks[over_fold_id].append(under_cik)
            folds_ciks[under_fold_id].remove(under_cik)
            folds_ciks[under_fold_id].append(over_cik)
            logging.debug(
                f"Iter {i+1}: Swapped {over_cik} (over) with {under_cik} (under) for feature {worst_col}. Cost reduction: {max_cost_reduction:.6f}"
            )
        else:
            logging.info(
                f"No improving swap found for feature {worst_col}. Stopping refinement."
            )
            break

    # --- Finalize Folds and Calculate Stats ---
    subsets = {}
    subsets_stats = {}

    for fold_id, ciks in folds_ciks.items():
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

        # Misttatement stats:
        total_count_per_misstatement = fold_df[LIST_MISTATEMENT_TYPE_FOR_TRAINING].sum(
            axis=0
        )

        # mistatment distirubiont in percentage
        total_count_per_misstatement_percentage = (
            total_count_per_misstatement / num_reports * 100 if num_reports > 0 else 0
        )

        # mistatement distirubtion amon fraud
        total_count_per_misstatement_fraud = (
            total_count_per_misstatement / num_fraud * 100 if num_fraud > 0 else 0
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
            "mistatement_distribution": total_count_per_misstatement.to_dict(),
            "mistatement_distribution_percentage": total_count_per_misstatement_percentage.to_dict()
            if isinstance(total_count_per_misstatement_percentage, pd.Series)
            else total_count_per_misstatement_percentage,
            "mistatement_distribution_fraud": total_count_per_misstatement_fraud.to_dict()
            if isinstance(total_count_per_misstatement_fraud, pd.Series)
            else total_count_per_misstatement_fraud,
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

        # Mistatement stats:
        total_count_per_misstatement = fold_df[LIST_MISTATEMENT_TYPE_FOR_TRAINING].sum(
            axis=0
        )
        # mistatment distirubiont in percentage
        total_count_per_misstatement_percentage = (
            total_count_per_misstatement / num_reports * 100
        ).to_dict()
        # mistatement distirubtion amon fraud
        total_count_per_misstatement_fraud = (
            total_count_per_misstatement / num_fraud * 100
        ).to_dict()

        # Store statistics
        subsets_stats[fold_id] = {
            "num_reports": num_reports,
            "num_fraud_samples": int(num_fraud),
            "num_non_fraud_samples": int(num_non_fraud),
            "fraud_ratio": float(num_fraud / num_reports if num_reports > 0 else 0),
            "distribution_per_industry": industry_dist,
            "distribution_per_industry_fraud": industry_fraud,
            "distribution_per_industry_non_fraud": industry_non_fraud,
            "num_companies": fold_df["cik"].nunique()
            if "cik" in fold_df.columns
            else 0,
            "mistatement_distribution": total_count_per_misstatement.to_dict(),
            "mistatement_distribution_percentage": total_count_per_misstatement_percentage,
            "mistatement_distribution_fraud": total_count_per_misstatement_fraud,
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


def split_dataset_by_time(df, keep_fraud_ratio=False, test_size=0.1, seed=None):
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
        # Seed the random number generator for reproducibility
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


class NumericalFeaturesType(str, Enum):
    """
    Enum for numerical features types.
    """

    EXTENDED = "EXTENDED"
    DECHOW = "DECHOW"
    EXTENDED_DECHOW = "EXTENDED_DECHOW"


def load_numpy_dataset(
    dataset_version,
    features,
    train_path=None,
    test_path=None,
    val_path=None,
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
    use_extended_features = features in [
        NumericalFeaturesType.EXTENDED.value,
        NumericalFeaturesType.EXTENDED_DECHOW.value,
    ]
    use_dechow_features = features in [
        NumericalFeaturesType.DECHOW.value,
        NumericalFeaturesType.EXTENDED_DECHOW.value,
    ]

    features = ["sicagg"]
    if use_extended_features:
        features.extend(EXTENDED_FINANCIAL_FEATURES)
    if use_dechow_features:

        features.extend(DECHOW_FEATURES)

    full_df = None
    if use_extended_features:
        # Load full financial data
        full_df = pd.read_csv(
            FINANCIALS_FIN_PATH,
            usecols=["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES,
        )
        full_df = full_df.drop_duplicates(subset=["cik", "year", "quarter"])

    if use_dechow_features:
        # Load full financial data for Dechow features
        full_df_dechow = pd.read_csv(
            DECHOW_FIN_PATH, usecols=["cik", "year", "quarter"] + DECHOW_FEATURES
        )
        full_df_dechow = full_df_dechow.drop_duplicates(
            subset=["cik", "year", "quarter"]
        )

        if full_df is not None:
            # Merge Dechow features into the full DataFrame
            full_df = full_df.merge(
                full_df_dechow, on=["cik", "year", "quarter"], how="left"
            )
        else:
            full_df = full_df_dechow

    # Helper function to merge financial data
    def merge_with_financials(df):
        # Merge with full financial data
        df = df.merge(full_df, on=["cik", "year", "quarter"], how="left")
        return df

    # Load training data
    logging.info(f"Loading train data from {train_path}")
    features_to_load = [
        "cik",
        "year",
        "quarter",
        "is_fraud",
        "sicagg",
    ] + LIST_MISTATEMENT_TYPE_FOR_TRAINING
    train_df = pd.read_csv(train_path, usecols=features_to_load)
    train_df = merge_with_financials(train_df)

    # Handle validation data
    if val_path is not None:
        logging.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path, usecols=features_to_load)
        val_df = merge_with_financials(val_df)
    else:
        # Create validation split from training data
        splitter = get_train_test_splitter({"dataset_version": dataset_version})
        train_df, val_df = splitter(train_df, test_size=0.1)

    # Load test data
    logging.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, usecols=features_to_load)
    test_df = merge_with_financials(test_df)

    # Clean up memory
    del full_df

    if return_raw_data:
        return train_df, val_df, test_df

    # Define target variables
    y_train = train_df["is_fraud"].astype(int)
    y_val = val_df["is_fraud"].astype(int)
    y_test = test_df["is_fraud"].astype(int)

    X_train = train_df[features]
    X_val = val_df[features]
    X_test = test_df[features]

    logging.info(f"Selected {len(features)} features for training")

    # Handle missing values
    for name, data in [("train", X_train), ("validation", X_val), ("test", X_test)]:
        if data.isna().sum().sum() > 0:
            logging.warning(
                f"Found {data.isna().sum().sum()} missing values in {name} data. Filling with zeros."
            )
            data.fillna(0, inplace=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, features


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


def split_dataset_by_cik(
    df,
    test_size=0.1,
    adjustment_iterations=100,
    swap_candidate_pool_size=20,
    seed=None,
):
    """
    Splits dataset by CIK, iteratively refining the test set to match the
    global distribution of misstatement types.

    This ensures strict company separation between train and test sets while creating
    a test set that is representative in terms of specific fraud characteristics,
    not just the overall fraud ratio.

    Methodology:
    1.  Calculates global target distributions for each misstatement type.
    2.  Performs an initial random split of companies (CIKs).
    3.  Iteratively refines the split by swapping companies between the train and
        test sets. The best swap is chosen based on which one most reduces the
        "cost" (sum of squared errors) between the test set's misstatement
        distribution and the global target distribution.

    Args:
        df (pd.DataFrame): Input dataframe with 'cik', 'sicagg', 'is_fraud',
                           and misstatement type columns.
        test_size (float): Target proportion of companies in the test set.
        adjustment_iterations (int): Max iterations for the refinement process.
        swap_candidate_pool_size (int): Size of the candidate pool for finding
                                        the best swap in each iteration.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (train_df, test_df)
    """
    if seed is not None:
        logging.info(f"Seeding random number generator with seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)

    logging.info("Starting CIK-based split with misstatement distribution balancing.")

    # --- Phase 0: Preparation and Target Calculation ---
    required_cols = ["cik", "is_fraud"] + LIST_MISTATEMENT_TYPE_FOR_TRAINING
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    if "report_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "report_id"})

    # Company-level aggregation
    agg_dict = {
        "total_reports": ("report_id", "count"),
        "fraud_reports": ("is_fraud", "sum"),
    }
    for mis_col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
        agg_dict[mis_col] = (mis_col, "sum")

    company_metrics = df.groupby("cik").agg(**agg_dict).reset_index()
    logging.info(f"Aggregated {len(company_metrics)} unique companies.")

    # Stats on total reports and fraud reports  per misstatement type
    logging.info(
        f"Total reports: {company_metrics['total_reports'].sum()}, "
        f"Total fraud reports: {company_metrics['fraud_reports'].sum()}"
    )
    logging.info("Total fraud reports per misstatement type:")
    for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
        total_misstatements = company_metrics[col].sum()
        logging.info(
            f"  {col}: {total_misstatements} ({total_misstatements / company_metrics['total_reports'].sum():.3%})"
        )

    # Calculate global target distributions for misstatements
    total_reports_global = company_metrics["total_reports"].sum()
    target_misstatement_dist = {
        col: company_metrics[col].sum() / total_reports_global
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING
    }
    target_test_companies_total = int(round(len(company_metrics) * test_size))

    logging.info(f"Target Test Set Size: ~{target_test_companies_total} companies.")
    logging.info(
        f"Target Misstatement Distribution (% of reports): { {k: f'{v:.3%}' for k, v in target_misstatement_dist.items()} }"
    )

    # --- Phase 1: Initial Random Split of Companies ---
    all_ciks = company_metrics["cik"].tolist()
    random.shuffle(all_ciks)

    current_test_ciks = set(all_ciks[:target_test_companies_total])
    current_train_ciks = set(all_ciks[target_test_companies_total:])

    logging.info(
        f"Initial random split: {len(current_test_ciks)} test CIKs, {len(current_train_ciks)} train CIKs."
    )

    # --- Phase 2: Iterative Refinement via Swapping ---
    logging.info(f"--- Starting refinement for {adjustment_iterations} iterations ---")

    def calculate_cost(ciks_set, company_metrics_df):
        """Helper to calculate cost (sum of squared errors from target distribution)."""
        if not ciks_set:
            return float("inf")
        fold_companies = company_metrics_df[company_metrics_df["cik"].isin(ciks_set)]
        total_reports_fold = fold_companies["total_reports"].sum()
        if total_reports_fold == 0:
            return float("inf")

        cost = 0.0
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
            current_ratio = fold_companies[col].sum() / total_reports_fold
            cost += (current_ratio - target_misstatement_dist[col]) ** 2
        return cost

    for i in tqdm(range(adjustment_iterations), desc="Refining split"):
        initial_cost = calculate_cost(current_test_ciks, company_metrics)

        # Find the misstatement type with the largest deviation in the test set
        test_companies_df = company_metrics[
            company_metrics["cik"].isin(current_test_ciks)
        ]
        total_reports_test = test_companies_df["total_reports"].sum()

        if total_reports_test == 0:
            break

        max_deviation = -1
        worst_col = None
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
            current_ratio = test_companies_df[col].sum() / total_reports_test
            deviation = abs(current_ratio - target_misstatement_dist[col])
            if deviation > max_deviation:
                max_deviation = deviation
                worst_col = col

        if worst_col is None:
            break  # Should not happen if there are misstatements

        # Find best swap candidates for the 'worst_col'
        # We want to swap a company from test with one from train to reduce the deviation
        test_ratio_for_worst_col = (
            test_companies_df[worst_col].sum() / total_reports_test
        )

        train_companies_df = company_metrics[
            company_metrics["cik"].isin(current_train_ciks)
        ]

        # If test ratio is too high, we need to swap a high-value test CIK for a low-value train CIK
        if test_ratio_for_worst_col > target_misstatement_dist[worst_col]:
            # Sort by the target column, then by CIK to break ties deterministically
            test_candidates = test_companies_df.sort_values(
                [worst_col, "cik"], ascending=[False, True]
            )
            train_candidates = train_companies_df.sort_values(
                [worst_col, "cik"], ascending=[True, True]
            )
        # If test ratio is too low, we do the opposite
        else:
            test_candidates = test_companies_df.sort_values(
                [worst_col, "cik"], ascending=[True, True]
            )
            train_candidates = train_companies_df.sort_values(
                [worst_col, "cik"], ascending=[False, True]
            )

        # Limit the search space for performance
        test_candidates = test_candidates.head(swap_candidate_pool_size)
        train_candidates = train_candidates.head(swap_candidate_pool_size)

        if test_candidates.empty or train_candidates.empty:
            logging.info(
                f"Iter {i+1}: No swap candidates found for {worst_col}. Skipping."
            )
            continue

        # Find the best swap among the candidates
        best_swap = None
        max_cost_reduction = 0

        for _, test_cand in test_candidates.iterrows():
            for _, train_cand in train_candidates.iterrows():
                # Simulate the swap
                temp_test_ciks = current_test_ciks.copy()
                temp_test_ciks.remove(test_cand["cik"])
                temp_test_ciks.add(train_cand["cik"])

                new_cost = calculate_cost(temp_test_ciks, company_metrics)
                cost_reduction = initial_cost - new_cost

                if cost_reduction > max_cost_reduction:
                    max_cost_reduction = cost_reduction
                    best_swap = (test_cand["cik"], train_cand["cik"])

        # Perform the best swap if it improves the cost
        if best_swap and max_cost_reduction > 0:
            test_swap_cik, train_swap_cik = best_swap
            current_test_ciks.remove(test_swap_cik)
            current_train_ciks.add(test_swap_cik)
            current_train_ciks.remove(train_swap_cik)
            current_test_ciks.add(train_swap_cik)
            logging.info(
                f"Iter {i+1}: Swapped Test CIK {test_swap_cik} <-> Train CIK {train_swap_cik}. "
                f"Cost reduced by {max_cost_reduction:.6f}."
            )

        else:
            logging.info(f"Iter {i+1}: No improving swap found. Stopping refinement.")
            break

    # --- Phase 3: Finalization and Reporting ---
    logging.info("\n--- Finalizing Split ---")
    final_train_df = (
        df[df["cik"].isin(current_train_ciks)].copy().reset_index(drop=True)
    )
    final_test_df = df[df["cik"].isin(current_test_ciks)].copy().reset_index(drop=True)

    # Final verification and reporting
    logging.info("\n--- Final Split Evaluation ---")

    final_test_companies = company_metrics[
        company_metrics["cik"].isin(current_test_ciks)
    ]
    final_total_reports = final_test_companies["total_reports"].sum()
    n_fraud_test = final_test_companies["fraud_reports"].sum()
    n_fraud_train = final_train_df["is_fraud"].sum()

    logging.info(
        f"Final Test Set: {len(final_test_df)} reports from {len(current_test_ciks)} companies with {n_fraud_test} fraud reports."
    )
    logging.info(
        f"Final Train Set: {len(final_train_df)} reports from {len(current_train_ciks)} companies with {n_fraud_train} fraud reports."
    )

    logging.info("\nMisstatement Distribution Comparison (Achieved vs. Target):")
    for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
        achieved_ratio = (
            final_test_companies[col].sum() / final_total_reports
            if final_total_reports > 0
            else 0
        )
        target_ratio = target_misstatement_dist[col]
        logging.info(
            f"  {col:<40}: "
            f"Achieved = {achieved_ratio:.3%}, "
            f"Target = {target_ratio:.3%}, "
            f"Diff = {achieved_ratio - target_ratio:+.3%}"
        )

    # Shuffle the final dataframes to prevent any order biases
    final_train_df = final_train_df.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )
    final_test_df = final_test_df.sample(frac=1, random_state=seed).reset_index(
        drop=True
    )

    return final_train_df, final_test_df


from sklearn.base import BaseEstimator, TransformerMixin


class IdentityScaler(BaseEstimator, TransformerMixin):
    """A Scaler with the same interface as MinMaxScaler, but that does nothing."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # does nothing, but returns self for chaining
        return self

    def transform(self, X):
        # returns X unchanged
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)



def _get_color_for_relevance(relevance, cmap_name="tab10"):
    """
    Applies a matplotlib colormap to a relevance value.

    Parameters
    ----------
    relevance : float
        A value between -1 and 1.
    cmap_name : str
        The name of the matplotlib colormap to use.

    Returns
    -------
    tuple
        An RGBA tuple with values between 0.0 and 1.0.
    """
    
    
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    colormap = cm.get_cmap(cmap_name)
    # Normalize the relevance value from [-1, 1] to [0, 1] for the colormap
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    return colormap(norm(relevance))


def pdf_heatmap(words, relevances, path, cmap="bwr"):
    """
    Generates a PDF file with a heatmap of word relevances without using LaTeX.

    This function directly creates a PDF using the ReportLab library. Each word is
    highlighted with a background color corresponding to its relevance score.
    The text will wrap automatically within the page margins.

    Parameters
    ----------
    words : list of str
        The words in the sentence, pre-processed by `clean_tokens`.
    relevances : np.ndarray or list of float
        The relevance scores for the words, normalized between -1 and 1.
    cmap : str, optional
        The name of the matplotlib colormap to use (default is "bwr").
    path : str, optional
        The path to save the output PDF file (default is 'heatmap.pdf').
    """
    # --- Input Validation ---
    import matplotlib.colors as mcolors
    import numpy as np
    from pathlib import Path
    from xml.sax.saxutils import escape

    # ReportLab imports for direct PDF generation
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    
    if not isinstance(relevances, np.ndarray):
        relevances = np.array(relevances)

    assert len(words) == len(relevances), "The number of words and relevances must be the same."
    assert relevances.min() >= -1 and relevances.max() <= 1, "Relevance scores must be normalized between -1 and 1."

    # --- PDF Document Setup ---
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    # Use a basic style that allows for inline formatting
    style = styles['Normal']
    style.leading = 14  # Add some line spacing for better readability

    # --- Generate Styled Text ---
    styled_text_parts = []
    for word, relevance in zip(words, relevances):
        # Get the RGBA color from the colormap
        rgba_color = _get_color_for_relevance(relevance, cmap)
        
        # Convert the RGBA color (0-1 floats) to a hex string for ReportLab
        hex_color = mcolors.to_hex(rgba_color)

        # Escape special XML characters in the word to prevent parsing errors
        # This is crucial because ReportLab's Paragraph uses an XML-like syntax.
        escaped_word = escape(word)

        # Create the styled snippet for this word.
        # The <backColor> tag sets the background color.
        # ReportLab preserves whitespace, so spaces in `escaped_word` are handled correctly.
        styled_text_parts.append(f'<backColor="{hex_color}">{escaped_word}</backColor>')

    # Join all styled parts into a single string for the Paragraph object
    full_styled_text = "".join(styled_text_parts)

    # --- Build the PDF ---
    story = [Paragraph(full_styled_text, style)]
    
    try:
        doc.build(story)
        print(f"PDF heatmap generated successfully at: {output_path}")
    except Exception as e:
        print(f"An error occurred during PDF generation: {e}")


# def clean_tokens(tokens):
#     """
#     Cleans wordpiece/tokenizer tokens into a human-readable format for display.

#     Handles different tokenization schemes (SentencePiece, WordPiece) and escapes
#     special LaTeX/XML characters.

#     Parameters
#     ----------
#     tokens : list of str
#         A list of tokens from a tokenizer.

#     Returns
#     -------
#     list of str
#         A list of cleaned words ready for display.
#     """
#     # Detect tokenization scheme and merge subwords
#     if any(" " in token for token in tokens):  # SentencePiece (e.g., XLNet, T5)
#         # The space character ' ' indicates a new word.
#         cleaned_tokens = [token.replace(" ", " ") for token in tokens]
#     elif any("Ġ" in token for token in tokens):  # GPT-2 / RoBERTa BPE
#         # The 'Ġ' character indicates a new word.
#         cleaned_tokens = [token.replace("Ġ", " ") for token in tokens]
#     elif any("##" in token for token in tokens):  # BERT WordPiece
#         # '##' indicates a subword that should be attached to the previous one.
#         cleaned_tokens = []
#         for token in tokens:
#             if token.startswith("##"):
#                 cleaned_tokens.append(token[2:])
#             else:
#                 # Add a space before new words, but not for the very first token.
#                 prefix = " " if cleaned_tokens else ""
#                 cleaned_tokens.append(prefix + token)
#     else:
#         # Fallback for simple tokenization (e.g., split by space)
#         cleaned_tokens = [" " + token for token in tokens]
#         if cleaned_tokens:
#             cleaned_tokens[0] = cleaned_tokens[0].lstrip()

#     return cleaned_tokens



def clean_tokens(words):
    """
    Clean wordpiece tokens by removing special characters and splitting them into words.
    """
    if any("▁" in word for word in words):
        words = [word.replace("▁", " ") for word in words]
            
    elif any("Ġ" in word for word in words):
        words = [word.replace("Ġ", " ") for word in words]
    
    elif any("##" in word for word in words):
        words = [word.replace("##", "") if "##" in word else " " + word for word in words]
        words[0] = words[0].strip()
    
    elif any("Ċ" in word for word in words):
        #Replace by \n
        words = [word.replace("Ċ", "\\newline") for word in words]
    
    elif any("Ĉ" in word for word in words):
        #Replace by \n
        words = [word.replace("Ĉ", "\\newline") for word in words]

    
    else:
        raise ValueError("The tokenization scheme is not recognized.")
    
    special_characters = ['&', '%', '$', '#', '_', '{', '}', '\\']
    for i, word in enumerate(words):
        for special_character in special_characters:
            if special_character in word:
                words[i] = word.replace(special_character, '\\' + special_character)

    # return words
    return [w.replace('Ċ', '\\newline').replace('âĢĵ', '--').replace('Ġ', '  ')  for w in words]

def generate_word_cloud_from_attributions(
    explanations_root_dir: Path,
    min_word_length = 5,
    lowercase = True,
    only_positive_attributions = False,
    only_negative_attributions = False,
    threshold_probability = 0.9
):
    """_summary_

    Get all the csv files in the directory, extract the words and the attributions (and weights) and then generate a word cloud from the words and their attributions.
    """
    
    assert not (only_positive_attributions and only_negative_attributions), "You cannot use both only_positive_attributions and only_negative_attributions at the same time."
    
    #Recursively find all files in the directory that match the pattern "*word_attributions.csv"
    all_files = list(explanations_root_dir.glob("**/*word_attributions.csv"))
    words_with_cumulative_relevance = {}
   
    test_prediction_files = list(explanations_root_dir.parent.rglob("test_predictions.csv"))
    if test_prediction_files is None:
        logging.warning("No test_predictions.csv file found in the parent directory. Cannot filter CIKs based on fraud probability.")
        return
    
    # Merge all test predictions into a single DataFrame
    df_predictions = pd.concat(
        [pd.read_csv(file,
                     usecols=["cik", "quarter", "fraud_probability", "y_true_id", "y_pred_id"]
                     ) for file in test_prediction_files],
        ignore_index=True,
    )
    # Only where y_pred ==y_true
    df_predictions = df_predictions[df_predictions.y_pred == df_predictions.y_true]
    
    if only_negative_attributions:
        df_predictions["fraud_probability"] = 1 - df_predictions["fraud_probability"]
    
    
    df_predictions["cik"] = df_predictions["cik"].astype(str)
    initial_length = len(df_predictions)
    df_predictions = df_predictions[df_predictions["fraud_probability"] >= threshold_probability]
    print(f"Filtered {initial_length - len(df_predictions)} reports based on fraud probability threshold of {threshold_probability}. Remaining reports: {len(df_predictions)}")
    all_ciks_quarter = set(zip(df_predictions["cik"], df_predictions["quarter"]))
        
    
    for file in tqdm(all_files, desc="Processing word attributions files for word cloud"):
        filename = file.name
        cik, quarter = filename.split("_")[:2]
        
        if (cik, quarter) not in all_ciks_quarter:
            continue
        
        
        
        df = pd.read_csv(file)
        if "word" not in df.columns or "relevance" not in df.columns:
            logging.warning(f"Skipping {file} as it does not contain 'word' or 'relevance' columns.")
            continue
        if only_positive_attributions:
            df = df[df["relevance"] > 0]
        elif only_negative_attributions:
            df = df[df["relevance"] < 0]
        
        
        
        for _, row in df.iterrows():
            
            word = str(row["word"])
            
            relevance = abs(row["relevance"])
            
            if lowercase:
                word = word.lower()
            
            #remove "\n"
            word = word.replace("\n", " ")
          
                
            # Remove special characters (", ., !, ?, etc.) except for alphanumeric characters and spaces
            word = ''.join(char for char in word if char.isalnum() or char.isspace() or char in ["'"])
            
            
            if len(word) < min_word_length:
                continue
            
            if word not in words_with_cumulative_relevance:
                words_with_cumulative_relevance[word] = 0.0
            
            words_with_cumulative_relevance[word] += relevance
    
    
    # Max normalization of relevance values
    max_relevance = max(words_with_cumulative_relevance.values())
    if max_relevance != 0:
        words_with_cumulative_relevance = {
            word: relevance / max_relevance for word, relevance in words_with_cumulative_relevance.items()
        }
    

    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        colormap="tab10",
        max_words=100,
        normalize_plurals=True,
    )
    
    wc.generate_from_frequencies(
        words_with_cumulative_relevance)
    
    file_name = explanations_root_dir / "word_cloud.png"
    if only_positive_attributions:
        file_name = explanations_root_dir / "word_cloud_positive_attributions.png"
    elif only_negative_attributions:
        file_name = explanations_root_dir / "word_cloud_negative_attributions.png"
    
    wc.to_file(explanations_root_dir / file_name)
    
    
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    config = {
        "dataset_version": "company_isolated_splitting",
        "fold_id": 1,
        "random_split": False,
    }
    train_path, test_path = load_cross_validation_path(config)
    logging.info(f"Train path: {train_path}, Test path: {test_path}")

    train_df = pd.read_csv(train_path)

    spitter = get_train_test_splitter(config)

    train_df, val_df = spitter(train_df, test_size=0.1, seed=5)

