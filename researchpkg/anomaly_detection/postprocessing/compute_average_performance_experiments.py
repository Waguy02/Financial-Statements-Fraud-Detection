import json
import multiprocessing
import os
import random
import sys  # Import sys for stderr printing
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from colorama import Fore, Style, init
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")  # Using a common style


from researchpkg.anomaly_detection.models.utils import load_cross_validation_path
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    get_sicagg,  # For better plot styling
)

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)


from researchpkg.anomaly_detection.config import (
    COMPANY_NAMES_INDEX,
    LIST_MISTATEMENT_TYPE_FOR_TRAINING,
    MDA_DATASET_PATH,
    PREPROCESSED_PATH,
    PREPROCESSED_PATH_EXTENDED,
    ROOT_DIR,
    SEED_TRAINING,
    SICAGG_INDEX_FILE,
)

MDA_PATH = MDA_DATASET_PATH / "quarterly"
MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"

# Ensure SICAGG_INDEX_FILE exists before trying to read it
if not Path(SICAGG_INDEX_FILE).exists():
    print(
        f"{Fore.RED}Error: SICAGG_INDEX_FILE not found at {SICAGG_INDEX_FILE}{Style.RESET_ALL}"
    )
    # Provide a dummy dictionary or raise an error, depending on whether this is critical
    # For this script, it's mainly for plotting subgroup performance, so a dummy might suffice
    # but let's assume it exists or the script will fail early.
    # As a workaround for testing if needed:
    # SIC_TO_INDUSTRY = {"01": "Unknown", "99": "Unknown"}
    raise FileNotFoundError(f"SICAGG_INDEX_FILE not found at {SICAGG_INDEX_FILE}")


SIC_TO_INDUSTRY = (
    pd.read_csv(SICAGG_INDEX_FILE, dtype={"sicagg": str})
    .set_index("sicagg")["industry_title"]
    .to_dict()
)

# Ensure COMPANY_NAMES_INDEX exists before trying to read it
if not Path(COMPANY_NAMES_INDEX).exists():
    print(
        f"{Fore.RED}Error: COMPANY_NAMES_INDEX not found at {COMPANY_NAMES_INDEX}{Style.RESET_ALL}"
    )
    # Similar to SIC_TO_INDUSTRY, provide a dummy or fail
    # CIK_TO_COMPANY_NAME = {123: "Dummy Corp"}
    raise FileNotFoundError(f"COMPANY_NAMES_INDEX not found at {COMPANY_NAMES_INDEX}")

CIK_TO_COMPANY_NAME = yaml.load(open(COMPANY_NAMES_INDEX), Loader=yaml.Loader)


def __map_sic(df):
    """Maps raw SIC codes to aggregated industry titles."""
    if "sic" in df.columns:
        # Ensure column is string type to handle potential NaNs or float representations
        df["sic_original"] = df["sic"].astype(str)  # Keep original if needed
        df["sic_agg"] = df["sic_original"].apply(
            lambda x: SIC_TO_INDUSTRY.get(get_sicagg(x[:2]), "Unknown")
            if len(x) > 0
            else "Unknown"
        )
        # Use the aggregated one for grouping, can rename back if preferred
        df["sic"] = df["sic_agg"].str.capitalize()
        df.drop(columns=["sic_agg"], inplace=True)  # Clean up temporary column
    else:
        print(
            f"{Fore.YELLOW}Warning: 'sic' column not found in DataFrame for SIC mapping.{Style.RESET_ALL}"
        )


def __map_cik_name(df):
    """Maps CIK numbers to company names, preserving original CIK."""
    # First, ensure 'cik_original' exists by copying 'cik' if it's there
    if "cik" in df.columns:
        if "cik_original" not in df.columns:
            try:
                # Attempt to convert to int, fallback to string if fails (e.g., NaN, non-numeric)
                df["cik_original"] = df["cik"].apply(
                    lambda x: int(x) if pd.notna(x) else np.nan
                )
            except ValueError:
                df["cik_original"] = df["cik"].astype(str)
                print(
                    f"{Fore.YELLOW}Warning: Could not convert 'cik' column to integer for 'cik_original' in __map_cik_name. Some CIKs might be non-numeric.{Style.RESET_ALL}",
                    file=sys.stderr,
                )

        # Now apply the mapping using cik_original and put result into 'cik'
        df["cik"] = df["cik_original"].apply(
            lambda x: CIK_TO_COMPANY_NAME.get(int(x), f"Unknown_CIK_{x}").capitalize()
            if pd.notna(x) and isinstance(x, (int, np.integer))
            else "Unknown_CIK"
        )
    else:
        print(
            f"{Fore.YELLOW}Warning: 'cik' column not found in DataFrame for CIK mapping.{Style.RESET_ALL}"
        )


# --- MDA Token Counting ---
# This part seems crucial and might be slow. It runs once when the script is imported.

TOKEN_CACHE_FILE = PREPROCESSED_PATH / "mda_token_counts_cache.json"


def __count_all_mda_tokens(mda_path):
    """
    Counts tokens for all MDA files indexed in all_index.csv.
    Returns a dictionary mapping 'CompanyName_quarter' to token count.
    Requires transformers library. Caches results to a JSON file.
    """
    if TOKEN_CACHE_FILE.exists():
        try:
            with open(TOKEN_CACHE_FILE, "r") as f:
                cached_data = json.load(f)
                print(
                    f"{Fore.GREEN}Loaded MDA token counts from cache: {TOKEN_CACHE_FILE}{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                return cached_data
        except Exception as e:
            print(
                f"{Fore.RED}Error loading token cache: {e}. Recalculating...{Style.RESET_ALL}",
                file=sys.stderr,
            )

    print(
        f"{Fore.CYAN}Starting MDA token counting...{Style.RESET_ALL}", file=sys.stderr
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # from transformers import AutoTokenizer # Commented out for faster execution during testing, uncomment for actual tokenization

    LLAMA_TOKENIZER = None  # Initialize to None
    try:
        from transformers import AutoTokenizer

        LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(
            "unsloth/Llama-3.1-8B-unsloth-bnb-4bit", trust_remote_code=True  #
        )
    except ImportError:
        print(
            f"{Fore.RED}Warning: transformers library not found. Cannot perform actual token counting. Using dummy count.{Style.RESET_ALL}",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"{Fore.RED}Error loading tokenizer: {e}. Cannot perform actual token counting. Using dummy count.{Style.RESET_ALL}",
            file=sys.stderr,
        )

    def count_mda_text_length(mda_quarter_id_cik_format):
        """Load and tokenize MDA text for a given quarter ID (CIK_YEARqQUARTER format)."""

        if LLAMA_TOKENIZER is None:
            # If tokenizer failed to load, return a random dummy length
            return random.randint(500, 5000)

        try:
            # The input ID is CIK_YEARqQUARTER
            mda_file = mda_path / f"{mda_quarter_id_cik_format}.txt"

            if not mda_file.exists():
                # print(f"MDA file not found: {mda_file}", file=sys.stderr)
                return None  # Return None or 0, None might be better to distinguish from empty file

            # Use 'ignore' for error handling if encoding is problematic
            with open(mda_file, "r", encoding="utf-8", errors="ignore") as f:
                mda_text = f.read()

            # Handle potential empty files gracefully
            if not mda_text.strip():
                return 0

            tokens = LLAMA_TOKENIZER.encode(
                mda_text, add_special_tokens=False
            )  # Don't add special tokens for length count
            return len(tokens)

        except Exception as e:
            print(
                f"{Fore.RED}Error loading or tokenizing MDA text for ID {mda_quarter_id_cik_format}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
            return None  # Indicate failure

    # all_index.csv contains mda_quarter_id in CIK_YEARqQUARTER format
    all_index_file = PREPROCESSED_PATH_EXTENDED / "v4/all_index.csv"
    if not all_index_file.exists():
        print(
            f"{Fore.RED}Error: all_index.csv not found at {all_index_file}. Cannot count MDA tokens.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return {}

    try:
        all_index = pd.read_csv(
            all_index_file, usecols=["mda_quarter_id", "cik"]
        )  # Also get CIK to avoid re-parsing string
        all_index["cik"] = all_index["cik"].astype(int)  # Ensure CIK is int
        mda_quarters_id_list_cik_format = all_index["mda_quarter_id"].unique().tolist()
        # Create a temp map for efficient company name lookup during multiprocessing if CIK_TO_COMPANY_NAME is large
        _cik_to_company_name_map = {
            row["cik"]: CIK_TO_COMPANY_NAME.get(row["cik"], f"Unknown_CIK_{row['cik']}")
            for _, row in all_index[["cik"]].drop_duplicates().iterrows()
        }

    except Exception as e:
        print(
            f"{Fore.RED}Error reading all_index.csv: {e}. Cannot count MDA tokens.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return {}

    def process_chunk(mda_quarter_id_cik_list_chunk, cik_to_name_map):
        """Process a chunk of CIK_YEARqQUARTER IDs and return results."""
        results = {}
        for mda_id_cik_format in tqdm(
            mda_quarter_id_cik_list_chunk,
            desc=f"Counting MDA tokens ({len(mda_quarter_id_cik_list_chunk)} files)",
        ):
            try:
                # Assumes mda_id_cik_format is "CIK_YEARqQUARTER"
                cik_str, quarter_str = mda_id_cik_format.split("_")
                cik = int(cik_str)
                company_name = cik_to_name_map.get(
                    cik, f"Unknown_CIK_{cik}"
                )  # Use passed map
                dict_key = f"{company_name}_{quarter_str}"

                token_count = count_mda_text_length(mda_id_cik_format)

                if token_count is not None:  # Only add if successful
                    results[dict_key] = token_count

            except Exception as e:
                print(
                    f"{Fore.RED}Error processing MDA ID {mda_id_cik_format}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                continue  # Skip this ID

        return results

    chunks = []
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Use N-1 cores
    # n_jobs = 1 # Revert this for actual multiprocessing

    # Adjust chunk size dynamically or use a fixed reasonable size
    chunk_size = max(1, len(mda_quarters_id_list_cik_format) // n_jobs)
    if (
        chunk_size == 0 and len(mda_quarters_id_list_cik_format) > 0
    ):  # Ensure at least one chunk if list is not empty
        chunk_size = 1

    for i in range(0, len(mda_quarters_id_list_cik_format), chunk_size):
        chunks.append(mda_quarters_id_list_cik_format[i : i + chunk_size])

    print(
        f"{Fore.CYAN}Processing {len(mda_quarters_id_list_cik_format)} MDA files in {len(chunks)} chunks using {n_jobs} processes.{Style.RESET_ALL}",
        file=sys.stderr,
    )

    # Use 'loky' backend which is more robust than 'multiprocessing' default
    mda_quarters_sizes_dict_list = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_chunk)(chunk, _cik_to_company_name_map) for chunk in chunks
    )

    mda_quarters_sizes_dict = {}
    for chunk_results in mda_quarters_sizes_dict_list:
        mda_quarters_sizes_dict.update(chunk_results)

    print(
        f"{Fore.CYAN}Finished MDA token counting. Counted tokens for {len(mda_quarters_sizes_dict)} files.{Style.RESET_ALL}",
        file=sys.stderr,
    )

    # Cache results
    try:
        from researchpkg.utils import (
            numpy_to_scalar,  # Ensure this is accessible or define a local version
        )

        with open(TOKEN_CACHE_FILE, "w") as f:
            json.dump(numpy_to_scalar(mda_quarters_sizes_dict), f)
        print(
            f"{Fore.GREEN}Saved MDA token counts to cache: {TOKEN_CACHE_FILE}{Style.RESET_ALL}",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"{Fore.RED}Error saving token cache: {e}{Style.RESET_ALL}", file=sys.stderr
        )

    return mda_quarters_sizes_dict


# Execute token counting once upon import
TOKEN_PER_RAW_MDA_DICT = __count_all_mda_tokens(MDA_PATH)
TOKEN_PER_SUMMARIZED_MDA_DICT = __count_all_mda_tokens(MDA_PATH_SUMMARIZED)
# --- End MDA Token Counting ---

# Save stats on tokens counts (average, std, mean)
def compute_stats_token_count(mda_token_dict):
    if not mda_token_dict:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "max": 0,
            "min": 0,
            "median": 0.0,
        }
    counts = np.array(list(mda_token_dict.values()))
    size = len(mda_token_dict)
    mean = np.mean(counts)
    std = np.std(counts)
    min_val, max_val = np.min(counts), np.max(counts)
    median = np.median(counts)

    return {
        "count": size,
        "mean": float(mean),
        "std": float(std),
        "max": int(max_val),
        "min": int(min_val),
        "median": float(median),
    }


token_stats_file = PREPROCESSED_PATH / "mda_token_counts_stats.json"

stats = {
    "raw_mda": compute_stats_token_count(TOKEN_PER_RAW_MDA_DICT),
    "summarized": compute_stats_token_count(TOKEN_PER_SUMMARIZED_MDA_DICT),
}

try:
    with open(token_stats_file, "w") as f:
        # Assuming numpy_to_scalar is available, if not, define it simply
        def numpy_to_scalar_local(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: numpy_to_scalar_local(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_scalar_local(elem) for elem in obj]
            return obj

        json.dump(numpy_to_scalar_local(stats), f, indent=4)
except Exception as e:
    print(
        f"{Fore.RED}Error saving token stats to file: {e}{Style.RESET_ALL}",
        file=sys.stderr,
    )


def standardize_metrics(content):
    """Standardize metric names and calculate derived metrics if needed."""

    # Handle old SFT format
    if "f1" in content and "f1_fraud_optimized" not in content:
        # Assuming 'f1' in old format was for the fraud class
        print(
            f"{Fore.YELLOW}Warning: Found 'f1' key. Assuming it's f1_fraud_optimized.{Style.RESET_ALL}"
        )
        content["f1_fraud_optimized"] = content["f1"]
        # Attempt to map other old keys if they exist
        if "precision" in content and "precision_fraud" not in content:
            content["precision_fraud"] = content["precision"]
        if "recall" in content and "recall_fraud" not in content:
            content["recall_fraud"] = content["recall"]

    # Calculate macro_f1 and weighted_f1 if confusion matrix is available and valid
    if "confusion_matrix" in content and isinstance(content["confusion_matrix"], dict):
        # Ensure all required keys are present
        cm_keys = [
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        ]
        if all(k in content["confusion_matrix"] for k in cm_keys):
            tp = content["confusion_matrix"]["true_positives"]
            fp = content["confusion_matrix"]["false_positives"]
            tn = content["confusion_matrix"]["true_negatives"]
            fn = content["confusion_matrix"]["false_negatives"]

            # Avoid division by zero
            f1_positive = (
                (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            )
            # Correct F1 for negative class:
            f1_negative = (
                (2 * tn) / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
            )

            content["macro_f1"] = (f1_positive + f1_negative) / 2.0

            # Support (number of actual instances of each class)
            support_positive = tp + fn
            support_negative = tn + fp
            total_support = support_positive + support_negative

            # Weighted F1 - only calculate if total support > 0
            if total_support > 0:
                weighted_f1 = (
                    support_positive * f1_positive + support_negative * f1_negative
                ) / total_support
                content["weighted_f1"] = weighted_f1
            else:
                content["weighted_f1"] = 0.0  # Or NaN
        else:
            print(
                f"{Fore.YELLOW}Warning: Confusion matrix dict is missing keys in {content}{Style.RESET_ALL}"
            )
            content["macro_f1"] = -1  # Indicate calculation failure
            content["weighted_f1"] = -1  # Indicate calculation failure

    elif "macro_f1" not in content:
        # If CM isn't available or invalid, set default values
        content["macro_f1"] = -1
        content["weighted_f1"] = -1
    # Else, use existing macro_f1/weighted_f1 if already calculated/present

    # Extract and standardize metrics using .get with default -1 for missing values
    auc_score = content.get(
        "auc_score", content.get("auc", -1.0)
    )  # Use -1.0 for float metrics
    f1_score_fraud = content.get(
        "f1_fraud_optimized", content.get("f1_fraud", content.get("f1_score", -1.0))
    )  # 'f1_score' might be macro or weighted in some outputs? Assuming fraud F1.
    accuracy = content.get("accuracy", -1.0)
    precision = content.get("precision_fraud", content.get("precision", -1.0))
    recall = content.get("recall_fraud", content.get("recall", -1.0))
    macro_f1 = content.get("macro_f1", -1.0)
    weighted_f1 = content.get("weighted_f1", -1.0)
    threshold = content.get(
        "best_threshold", content.get("threshold_used", content.get("threshold", -1.0))
    )  # Use -1.0 for float

    return {
        "auc_score": auc_score,
        "f1_score_fraud": f1_score_fraud,  # Renamed for clarity
        "accuracy": accuracy,
        "precision_fraud": precision,  # Renamed for clarity
        "recall_fraud": recall,  # Renamed for clarity
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "threshold": threshold,
    }


def calculate_fold_averages(per_folds_metrics):
    """Calculate mean and standard deviation across folds for a dict of metrics."""
    if not per_folds_metrics:
        return {}

    average_metrics = {}
    # Get all unique metric keys from all folds
    all_keys = set()
    for fold_metrics in per_folds_metrics.values():
        all_keys.update(fold_metrics.keys())

    for key in all_keys:
        values = [
            per_folds_metrics[fold_num][key]
            for fold_num in per_folds_metrics
            if key in per_folds_metrics[fold_num]
        ]
        # Only calculate mean/std if there are actual numeric values
        if values and all(isinstance(v, (int, float)) for v in values):
            mean = float(np.mean(values))
            std = float(np.std(values))  # Use numpy std for population std by default
            average_metrics[key] = {"mean": mean, "std": std}
        else:
            # If no values or non-numeric, indicate missing/failure
            average_metrics[key] = {"mean": -1.0, "std": -1.0}

    return average_metrics


def compute_auc_single_experiment_best_test_metrics(experiment_dir: Path):
    """Compute average metrics from test results across folds."""
    per_folds_metrics = {}

    fold_dirs_found = False
    for fold_num in [1, 2, 3, 4, 5]:
        sub_dir = experiment_dir / f"fold_{fold_num}"

        if not sub_dir.exists():
            # print(f"Fold {fold_num} does not exist in {experiment_dir}") # Too verbose
            continue
        fold_dirs_found = True  # Mark that we found at least one fold

        test_metrics_files = list(sub_dir.rglob("test_metrics.json"))
        if not test_metrics_files:
            # print(f"No test metrics file found in {sub_dir}") # Too verbose
            continue

        # Prefer the root test_metrics.json if multiple exist (e.g., from sub-runs)
        root_test_metrics = sub_dir / "test_metrics.json"
        if root_test_metrics.exists():
            best_file = root_test_metrics
        elif test_metrics_files:
            best_file = test_metrics_files[
                0
            ]  # Use the first found if root doesn't exist
        else:
            continue  # Should not happen due to the check above, but for safety

        try:
            with open(best_file, "r") as f:
                content = json.load(f)
                per_folds_metrics[fold_num] = standardize_metrics(content)
        except Exception as e:
            print(
                f"{Fore.RED}Error reading or standardizing test metrics file {best_file}: {e}{Style.RESET_ALL}"
            )

    if not fold_dirs_found:
        print(
            f"{Fore.YELLOW}No fold directories found for {experiment_dir}{Style.RESET_ALL}"
        )
        return  # Exit if no folds were found at all

    average_metrics = calculate_fold_averages(per_folds_metrics)

    # Write results to file
    average_metrics_file = experiment_dir / "average_test_metrics.json"
    try:
        with open(average_metrics_file, "w") as f:
            json.dump(average_metrics, f, indent=4)
        # print(f"Average test metrics written to {average_metrics_file}") # Too verbose
    except Exception as e:
        print(
            f"{Fore.RED}Error writing average test metrics to {average_metrics_file}: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )


def compute_auc_single_experiment_best_validation_metrics(experiment_dir: Path):
    """Compute average metrics from best validation results across folds."""
    per_folds_metrics = {}

    fold_dirs_found = False
    for fold_num in [1, 2, 3, 4, 5]:
        sub_dir = experiment_dir / f"fold_{fold_num}"

        if not sub_dir.exists():
            # print(f"Fold {fold_num} does not exist in {experiment_dir}") # Too verbose
            continue
        fold_dirs_found = True  # Mark that we found at least one fold

        best_file = None

        # 1. Check for a specific 'best_val_metrics.json' or 'val_metrics.json'
        val_metrics_files = list(sub_dir.rglob("val_metrics.json"))
        if val_metrics_files:
            # Prefer root val_metrics.json
            root_val_metrics = sub_dir / "val_metrics.json"
            if root_val_metrics.exists():
                best_file = root_val_metrics
            else:
                best_file = val_metrics_files[
                    0
                ]  # Use the first found if root doesn't exist

        # 2. If not found, look for epoch metrics and find the best one
        if best_file is None:
            # Look for epoch metrics files
            valid_files = []

            for file in sub_dir.rglob("metrics_epoch_*.json"):
                try:
                    # Load content to check if it's valid and has F1
                    content = json.load(open(file))
                    # Check for a key that suggests it's a metric file, like 'f1' or 'accuracy'
                    if (
                        "f1" in content
                        or "accuracy" in content
                        or "f1_fraud_optimized" in content
                    ):
                        valid_files.append((file, content))
                    # standardize_metrics(content)  # Could call this, but might raise errors if keys are missing early
                    # valid_files.append(file) # Append file path if standardization *would* work
                except Exception as e:
                    # print(f"Error loading file {file} for best epoch search: {e}") # Too verbose
                    continue

            if not valid_files:
                # print(f"No validation metrics files found in {sub_dir} for fold {fold_num}") # Too verbose
                continue

            # Find the best file based on 'f1_fraud_optimized' or a fallback F1 metric
            def get_f1_for_sort(file_content_tuple):
                # Use the content loaded earlier
                content = file_content_tuple[1]
                # Prioritize f1_fraud_optimized, then f1_fraud, then general f1
                return content.get(
                    "f1_fraud_optimized", content.get("f1_fraud", content.get("f1", -1))
                )

            valid_files.sort(key=get_f1_for_sort, reverse=True)

            if valid_files:
                best_file = valid_files[0][0]  # Get the file path

        if best_file:
            try:
                with open(best_file, "r") as f:
                    content = json.load(f)
                    per_folds_metrics[fold_num] = standardize_metrics(content)
            except Exception as e:
                print(
                    f"{Fore.RED}Error reading or standardizing best validation metrics file {best_file}: {e}{Style.RESET_ALL}"
                )

    if not fold_dirs_found:
        print(
            f"{Fore.YELLOW}No fold directories found for {experiment_dir}{Style.RESET_ALL}"
        )
        return  # Exit if no folds were found at all

    average_metrics = calculate_fold_averages(per_folds_metrics)

    # Write results to file
    average_metrics_file = experiment_dir / "average_best_validation_metrics.json"
    try:
        with open(average_metrics_file, "w") as f:
            json.dump(average_metrics, f, indent=4)
        # print(f"Average best validation metrics written to {average_metrics_file}") # Too verbose
    except Exception as e:
        print(
            f"{Fore.RED}Error writing average best validation metrics to {average_metrics_file}: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )


def _load_and_standardize_predictions_df(file_path: Path) -> pd.DataFrame | None:
    """Loads a prediction CSV and ensures y_true_id, y_pred_id columns are correctly formatted."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(
                f"{Fore.YELLOW}Prediction file {file_path} is empty.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            return None

        # Check for and standardize y_true_id/y_pred_id
        if "y_true_id" not in df.columns or "y_pred_id" not in df.columns:
            if "y_true" in df.columns and "y_pred" in df.columns:
                df["y_true_id"] = (
                    df["y_true"]
                    .astype(str)
                    .str.lower()
                    .apply(lambda x: 1 if x == "fraud" else 0)
                )
                df["y_pred_id"] = (
                    df["y_pred"]
                    .astype(str)
                    .str.lower()
                    .apply(lambda x: 1 if x == "fraud" else 0)
                )
            elif "true_labels" in df.columns and "predicted_label" in df.columns:
                df["y_true_id"] = df["true_labels"]
                df["y_pred_id"] = df["predicted_label"]
                if (
                    "probability" in df.columns
                    and "fraud_probability" not in df.columns
                ):
                    df.rename(
                        columns={"probability": "fraud_probability"}, inplace=True
                    )
            else:
                print(
                    f"{Fore.RED}Error: Cannot find or derive y_true_id/y_pred_id in {file_path}. Required columns missing.{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                return None

        # Ensure cik is integer type before potential mapping or token lookup
        if "cik" in df.columns:
            if "cik_original" not in df.columns:  # Only create if not already present
                try:
                    # Attempt to convert to int, fallback to string if fails (e.g., NaN, non-numeric)
                    df["cik_original"] = df["cik"].apply(
                        lambda x: int(x) if pd.notna(x) else np.nan
                    )
                except ValueError:
                    df["cik_original"] = df["cik"].astype(str)
                    print(
                        f"{Fore.YELLOW}Warning: Could not convert 'cik' column to integer for 'cik_original' in {file_path}. Some CIKs might be non-numeric.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )

        return df
    except Exception as e:
        print(
            f"{Fore.RED}Error reading or processing prediction file {file_path}: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return None


def analyze_prediction_files(experiment_dir: Path):
    """
    Analyze prediction CSV files and generate detailed distribution statistics
    of fraud across enterprises (CIK), sectors (SIC), and time periods (quarters).
    columns cik,sic,quarter,y_true,y_pred,glabels

    Args:
        experiment_dir: Path to the experiment directory
    """
    fold_unseen_in_train_metrics = []

    if "rcma" in str(experiment_dir).lower():
        print(
            f"{Fore.YELLOW}Skipping distribution analysis for RCMA experiment directory: {experiment_dir}{Style.RESET_ALL}"
        )
        # RCMA might not have the same CIK/SIC structure or data files
        return

    # Determine dataset version from experiment directory path if possible
    dataset_version = "company_isolated_splitting"  # Default
    for v in ["company_isolated_splitting", "time_splitting", "time_splitting"]:  # Add other versions if needed
        if v in str(experiment_dir):
            dataset_version = v
            break
    print(
        f"Determined dataset version: {dataset_version} for {experiment_dir}",
        file=sys.stderr,
    )

    for fold_num in tqdm(
        [1, 2, 3, 4, 5],
        desc=f"Analyzing prediction files in {experiment_dir}",
        leave=False,
    ):
        sub_dir = experiment_dir / f"fold_{fold_num}"

        if not sub_dir.exists():
            # print(f"{Fore.YELLOW}Fold {fold_num} does not exist in {experiment_dir}{Style.RESET_ALL}") # Too verbose
            continue

        try:
            # Load the train test data of the fold for train_ciks
            train_path, _ = load_cross_validation_path(
                {
                    "dataset_version": dataset_version,
                    "fold_id": fold_num,
                }
            )
            # Only load 'cik' for train set to get the original integer CIKs
            train_df = pd.read_csv(train_path, usecols=["cik"])
            train_ciks = set(
                train_df["cik"].astype(int).unique()
            )  # Ensure int type for comparison

            print(
                f"Preloaded CIKs for fold {fold_num}: {len(train_ciks)} train",
                file=sys.stderr,
            )

        except Exception as e:
            print(
                f"{Fore.RED}Error loading CIKs for fold {fold_num} in {experiment_dir}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
            train_ciks = set()

        # Process test predictions
        test_summary = _process_prediction_file(
            sub_dir, "test_predictions.csv", train_ciks
        )
        if test_summary and isinstance(test_summary, dict):
            unseen_train = test_summary.get("unseen_in_train_metrics")
            if unseen_train:
                fold_unseen_in_train_metrics.append(unseen_train)

        # Process validation predictions - handle alternative filenames
        val_files = list(sub_dir.rglob("val_predictions.csv"))
        if val_files:
            _process_prediction_file(sub_dir, val_files[0], train_ciks)
        else:
            # Try to find val_prediction_epoch_*.csv files
            val_epoch_files = list(sub_dir.rglob("val_prediction_epoch_*.csv"))
            if val_epoch_files:
                # If we have validation epoch files, find the one from the best epoch
                # First check if there's a best_epoch.txt file
                best_epoch_file_path = (
                    sub_dir / "best_epoch.txt"
                )  # Assuming best_epoch.txt is at sub_dir root
                best_epoch_found = None
                if best_epoch_file_path.exists():
                    try:
                        with open(best_epoch_file_path, "r") as f:
                            best_epoch_found = f.read().strip()
                    except Exception as e:
                        print(
                            f"Error reading best_epoch.txt in {sub_dir}: {e}",
                            file=sys.stderr,
                        )

                best_val_file = None
                if best_epoch_found:
                    best_val_file_candidates = list(
                        sub_dir.rglob(f"val_prediction_epoch_{best_epoch_found}.csv")
                    )
                    if best_val_file_candidates:
                        best_val_file = best_val_file_candidates[0]
                        # print(f"Using best epoch ({best_epoch_found}) validation predictions: {best_val_file}", file=sys.stderr)
                    else:
                        print(
                            f"{Fore.YELLOW}Warning: best_epoch.txt exists ({best_epoch_found}) but corresponding val_prediction_epoch_{best_epoch_found}.csv not found in {sub_dir}{Style.RESET_ALL}",
                            file=sys.stderr,
                        )

                if best_val_file is None:
                    # If best_epoch.txt was not found or failed, use the most recent file among epoch files
                    if val_epoch_files:
                        newest_val_file = max(
                            val_epoch_files, key=lambda p: p.stat().st_mtime
                        )
                        # print(f"best_epoch.txt not found or invalid, using newest validation prediction file: {newest_val_file}", file=sys.stderr)
                        best_val_file = newest_val_file
                    else:
                        # print(f"No validation prediction files found in {sub_dir}", file=sys.stderr) # Already printed by rglob search
                        pass  # No validation files at all

                if best_val_file:
                    _process_prediction_file(sub_dir, best_val_file, train_ciks)
                # else: no validation files found at all, skip validation processing for this fold

    # Aggregate results for unseen metrics across folds
    aggregated = {}
    if fold_unseen_in_train_metrics:
        # Filter out any empty or invalid metric dictionaries
        valid_metrics = [
            m for m in fold_unseen_in_train_metrics if m and isinstance(m, dict)
        ]
        if valid_metrics:
            aggregated["unseen_in_train_avg"] = _average_unseen_metrics(valid_metrics)
        else:
            print(
                f"{Fore.YELLOW}No valid unseen_in_train_metrics collected across folds for {experiment_dir}{Style.RESET_ALL}",
                file=sys.stderr,
            )

    if aggregated:
        out_file = experiment_dir / "aggregated_unseen_metrics.json"
        try:
            with open(out_file, "w") as f:
                json.dump(aggregated, f, indent=4)
            print(f"Aggregated unseen metrics saved to {out_file}", file=sys.stderr)
        except Exception as e:
            print(
                f"{Fore.RED}Error writing aggregated unseen metrics to {out_file}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )


def _average_unseen_metrics(metrics_list: list[dict]) -> dict:
    """Compute average of unseen metrics across folds."""
    # Collect all unique metric keys
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())

    results = {}
    for k in keys:
        # Filter for valid numeric values for this key
        vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))  # Use numpy std for population std
            results[k] = {
                "mean": mean_val,
                "std": std_val,
                # Optional: add counts or list of values for debugging
                # "_values": vals,
                # "_n": len(vals)
            }
        else:
            # If no valid numeric values found, indicate it
            results[k] = {"mean": -1.0, "std": -1.0}
    return results


def _process_prediction_file(sub_dir, prediction_file_name, train_ciks):
    """Helper function to process a single prediction file and generate statistics"""
    summary = None
    try:
        # Find the specific prediction file by name within the subdir and its subdirectories
        pred_files = list(
            sub_dir.rglob(str(prediction_file_name))
        )  # Ensure prediction_file_name is a string if it comes as Path
        if not pred_files:
            # print(f"{Fore.YELLOW}Prediction file '{prediction_file_name}' not found in {sub_dir}{Style.RESET_ALL}", file=sys.stderr) # Too verbose
            return  # Exit if file not found

        # Prefer the file directly in the sub_dir if multiple are found (e.g., in epoch folders)
        pred_file = (
            sub_dir / Path(prediction_file_name).name
        )  # Use .name to strip parent dirs if prediction_file_name was a Path object
        if not pred_file.exists() and pred_files:
            pred_file = pred_files[0]  # Fallback to the first one found by rglob

        print(f"Processing prediction file: {pred_file}", file=sys.stderr)

        df = _load_and_standardize_predictions_df(pred_file)
        if df is None:
            return

        # Apply mapping functions
        __map_sic(df)  # Maps to 'sic' column
        __map_cik_name(
            df
        )  # Maps to 'cik' column (now contains company name), 'cik_original' has int CIK

        # Check if essential columns exist after mapping
        if (
            "cik" not in df.columns  # 'cik' now holds company name
            or "sic" not in df.columns
            or "quarter" not in df.columns
            or "cik_original" not in df.columns  # Need this for unseen/token lookups
        ):
            print(
                f"{Fore.RED}Error: Essential columns (cik_original, cik, sic, quarter) missing after mapping in {pred_file}. Skipping distribution analysis.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            return

        # Generate detailed distribution statistics
        # Calculate overall metrics first
        total_samples = len(df)
        fraud_samples = int(df["y_true_id"].sum())
        predicted_fraud = int(df["y_pred_id"].sum())

        # Calculate confusion matrix components
        true_positive = int(((df["y_true_id"] == 1) & (df["y_pred_id"] == 1)).sum())
        false_positive = int(((df["y_true_id"] == 0) & (df["y_pred_id"] == 1)).sum())
        true_negative = int(((df["y_true_id"] == 0) & (df["y_pred_id"] == 0)).sum())
        false_negative = int(((df["y_true_id"] == 1) & (df["y_pred_id"] == 0)).sum())

        summary = {
            "total_samples": total_samples,
            "fraud_samples": fraud_samples,
            "predicted_fraud": predicted_fraud,
            "confusion_matrix": {
                "true_positives": true_positive,
                "false_positives": false_positive,
                "true_negatives": true_negative,
                "false_negatives": false_negative,
            },
            # Distribution by quarter
            # Ensure quarter column exists and is valid
            "by_quarter": {},
            "fraud_trend_by_quarter": {},
            "by_sic": {},  # Will be filled below
            "top_fraud_enterprises": {},
            "top_false_positive_enterprises": {},
        }

        if "quarter" in df.columns:
            try:
                # Filter out potential non-string/invalid quarter values before grouping
                df_quarters = df[
                    df["quarter"].astype(str).str.match(r"\d{4}q[1-4]", na=False)
                ].copy()  # Basic pattern match
                if not df_quarters.empty:
                    quarter_fraud_dist = (
                        df_quarters.groupby("quarter")["y_true_id"].sum().to_dict()
                    )
                    quarter_fraud_rate = (
                        df_quarters.groupby("quarter")["y_true_id"].mean().to_dict()
                    )
                    quarter_detection_rate = (
                        df_quarters.groupby("quarter")["y_pred_id"].mean().to_dict()
                    )
                    summary["by_quarter"] = {
                        "fraud_distribution": {
                            str(k): int(v) for k, v in quarter_fraud_dist.items()
                        },
                        "fraud_rate": {
                            str(k): float(v) for k, v in quarter_fraud_rate.items()
                        },
                        "detection_rate": {
                            str(k): float(v) for k, v in quarter_detection_rate.items()
                        },
                    }

                    # Add time-based analysis: fraud trend over quarters
                    try:
                        # Convert quarters to a sortable structure (YYYY.Q)
                        df_quarters["quarter_sortable"] = df_quarters["quarter"].apply(
                            lambda x: float(x[:4]) + int(x[5]) / 10.0
                            if pd.notna(x) and len(x) >= 6 and x[4] == "q"
                            else np.nan
                        )
                        # Drop rows where quarter_sortable couldn't be created
                        df_quarters.dropna(subset=["quarter_sortable"], inplace=True)

                        if not df_quarters.empty:
                            quarters_sorted = (
                                df_quarters["quarter_sortable"].sort_values().unique()
                            )
                            # Map sortable back to original string for the dict key
                            sortable_to_quarter_str = df_quarters.set_index(
                                "quarter_sortable"
                            )["quarter"].to_dict()

                            fraud_by_time = {
                                sortable_to_quarter_str.get(q_sort, "Unknown"): float(
                                    df_quarters[
                                        df_quarters["quarter_sortable"] == q_sort
                                    ]["y_true_id"].mean()
                                )
                                for q_sort in quarters_sorted
                            }
                            summary["fraud_trend_by_quarter"] = fraud_by_time
                    except Exception as e:
                        print(
                            f"{Fore.YELLOW}Error processing time trend by quarter in {pred_file}: {e}{Style.RESET_ALL}",
                            file=sys.stderr,
                        )

            except Exception as e:
                print(
                    f"{Fore.RED}Error processing quarter distribution in {pred_file}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )

        # Distribution by sector (SIC aggregate)
        if "sic" in df.columns:
            try:
                # Filter out potential non-string/invalid sic values
                df_sic = df[
                    df["sic"].astype(str).str.len() > 0
                ].copy()  # Keep only non-empty strings
                if not df_sic.empty:
                    sic_fraud_dist = df_sic.groupby("sic")["y_true_id"].sum().to_dict()
                    sic_fraud_rate = df_sic.groupby("sic")["y_true_id"].mean().to_dict()
                    sic_sample_count = df_sic.groupby("sic").size().to_dict()
                    summary["by_sic"] = {
                        "fraud_distribution": {
                            str(k): int(v) for k, v in sic_fraud_dist.items()
                        },
                        "fraud_rate": {
                            str(k): float(v) for k, v in sic_fraud_rate.items()
                        },
                        "sample_count": {
                            str(k): int(v) for k, v in sic_sample_count.items()
                        },
                    }
            except Exception as e:
                print(
                    f"{Fore.RED}Error processing SIC distribution in {pred_file}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )

        # Top enterprises with fraud (using mapped company names)
        if "cik" in df.columns:  # 'cik' now holds company name
            try:
                top_fraud_companies = (
                    df[df["y_true_id"] == 1]
                    .groupby("cik")  # Group by company name
                    .size()
                    .sort_values(ascending=False)
                    .head(20)
                    .to_dict()
                )
                summary["top_fraud_enterprises"] = {
                    str(k): int(v) for k, v in top_fraud_companies.items()
                }

                # Enterprises with most false positives (using mapped company names)
                top_fp_companies = (
                    df[(df["y_true_id"] == 0) & (df["y_pred_id"] == 1)]
                    .groupby("cik")  # Group by company name
                    .size()
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict()
                )
                summary["top_false_positive_enterprises"] = {
                    str(k): int(v) for k, v in top_fp_companies.items()
                }

            except Exception as e:
                print(
                    f"{Fore.RED}Error processing top companies distribution in {pred_file}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )

        # Add unseen in train metrics if processing the test set
        if "test_predictions.csv" in str(pred_file).lower() and train_ciks is not None:
            try:
                # Define "unseen" as CIKs not in train AND not in val (true unseen in test)
                # Use 'cik_original' for lookup against train_ciks
                unseen_df = df[(~df["cik_original"].isin(train_ciks))].copy()
                print(
                    f"Found {len(unseen_df['cik_original'].unique()) if 'cik_original' in unseen_df.columns else 0} unseen CIKs ({len(unseen_df)} samples) in {pred_file}",
                    file=sys.stderr,
                )

                if not unseen_df.empty:
                    # Ensure sufficient samples for metrics and unique classes for AUC
                    if len(unseen_df) >= 2:  # Need at least two samples
                        acc = accuracy_score(
                            unseen_df["y_true_id"], unseen_df["y_pred_id"]
                        )
                        prec = precision_score(
                            unseen_df["y_true_id"],
                            unseen_df["y_pred_id"],
                            zero_division=0,
                        )
                        rec = recall_score(
                            unseen_df["y_true_id"],
                            unseen_df["y_pred_id"],
                            zero_division=0,
                        )
                        f1 = f1_score(
                            unseen_df["y_true_id"],
                            unseen_df["y_pred_id"],
                            zero_division=0,
                        )

                        # Compute AUC only if fraud_probability exists and there are at least two unique true labels
                        auc = -1.0  # Default AUC
                        if (
                            "fraud_probability" in unseen_df.columns
                            and len(unseen_df["y_true_id"].unique()) > 1
                        ):
                            try:
                                auc = roc_auc_score(
                                    unseen_df["y_true_id"],
                                    unseen_df["fraud_probability"],
                                )
                            except ValueError as e:
                                print(
                                    f"{Fore.YELLOW}Warning: AUC calculation failed for unseen data in {pred_file}: {e}{Style.RESET_ALL}",
                                    file=sys.stderr,
                                )
                                auc = -1.0  # Set to -1 if calculation fails

                        summary["unseen_in_train_metrics"] = {
                            "accuracy": float(acc),
                            "precision": float(prec),
                            "recall": float(rec),
                            "f1_score": float(f1),
                            "auc_score": float(auc),
                            "nb_unseen_ciks": int(
                                len(unseen_df["cik_original"].unique())
                                if "cik_original" in unseen_df.columns
                                else 0
                            ),
                            "nb_unseen_samples": int(len(unseen_df)),
                            "nb_unseen_fraud_samples": int(
                                unseen_df["y_true_id"].sum()
                            ),
                            "nb_unseen_fraud_rate": float(
                                unseen_df["y_true_id"].mean()
                            ),
                            "nb_test_ciks": int(
                                len(df["cik_original"].unique())
                                if "cik_original" in df.columns
                                else 0
                            ),
                            "nb_test_samples": int(len(df)),
                            "nb_test_fraud_samples": int(df["y_true_id"].sum()),
                        }
                        print(
                            f"Calculated unseen metrics for {pred_file}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"{Fore.YELLOW}Not enough samples ({len(unseen_df)}) in unseen data for {pred_file} to calculate metrics.{Style.RESET_ALL}",
                            file=sys.stderr,
                        )
                        # Add a placeholder indicating insufficient data
                        summary["unseen_in_train_metrics"] = {
                            "accuracy": -1.0,
                            "precision": -1.0,
                            "recall": -1.0,
                            "f1_score": -1.0,
                            "auc_score": -1.0,
                            "nb_unseen_ciks": int(
                                len(unseen_df["cik_original"].unique())
                                if "cik_original" in unseen_df.columns
                                else 0
                            ),
                            "nb_unseen_samples": int(len(unseen_df)),
                            "nb_unseen_fraud_samples": int(
                                unseen_df["y_true_id"].sum()
                            ),
                            "nb_unseen_fraud_rate": float(unseen_df["y_true_id"].mean())
                            if not unseen_df.empty
                            else 0.0,
                            "nb_test_ciks": int(
                                len(df["cik_original"].unique())
                                if "cik_original" in df.columns
                                else 0
                            ),
                            "nb_test_samples": int(len(df)),
                            "nb_test_fraud_samples": int(df["y_true_id"].sum()),
                        }
                else:
                    print(
                        f"{Fore.YELLOW}No unseen samples found in {pred_file}. Skipping unseen metrics calculation.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    # Add a placeholder indicating no unseen data
                    summary["unseen_in_train_metrics"] = {
                        "accuracy": -1.0,
                        "precision": -1.0,
                        "recall": -1.0,
                        "f1_score": -1.0,
                        "auc_score": -1.0,
                        "nb_unseen_ciks": 0,
                        "nb_unseen_samples": 0,
                        "nb_unseen_fraud_samples": 0,
                        "nb_unseen_fraud_rate": 0.0,
                        "nb_test_ciks": int(
                            len(df["cik_original"].unique())
                            if "cik_original" in df.columns
                            else 0
                        ),
                        "nb_test_samples": int(len(df)),
                        "nb_test_fraud_samples": int(df["y_true_id"].sum()),
                    }

            except Exception as e:
                print(
                    f"{Fore.RED}Error calculating unseen in train metrics for {pred_file}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                # Add a placeholder indicating failure
                summary["unseen_in_train_metrics"] = {
                    "accuracy": -1.0,
                    "precision": -1.0,
                    "recall": -1.0,
                    "f1_score": -1.0,
                    "auc_score": -1.0,
                    "nb_unseen_ciks": -1,
                    "nb_unseen_samples": -1,
                    "nb_unseen_fraud_samples": -1,
                    "nb_unseen_fraud_rate": -1.0,
                    "nb_test_ciks": -1,
                    "nb_test_samples": -1,
                    "nb_test_fraud_samples": -1,
                }

        # Save summary
        summary_file = pred_file.parent / f"{pred_file.stem}_summary.json"
        try:
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=4)
            # print(f"Saved enhanced prediction summary to {summary_file}") # Too verbose
        except Exception as e:
            print(
                f"{Fore.RED}Error writing summary to {summary_file}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )

        # Optional: Save distribution plots - only for Test predictions
        if "test_predictions.csv" in str(pred_file).lower():
            try:
                # Plot fraud distribution by sector
                if "sic" in df.columns and not df["sic"].empty:
                    plt.figure(figsize=(10, 6))
                    # Only plot sectors with enough samples (e.g., > 5 samples)
                    sic_counts = df["sic"].value_counts()
                    sectors_to_plot = sic_counts[
                        sic_counts > 5
                    ].index  # Threshold can be adjusted
                    if not sectors_to_plot.empty:
                        fraud_by_sector = (
                            df[df["sic"].isin(sectors_to_plot)]
                            .groupby("sic")["y_true_id"]
                            .mean()
                            .sort_values(ascending=False)
                        )
                        if not fraud_by_sector.empty:
                            fraud_by_sector.plot(kind="bar")
                            plt.title("Fraud Rate by Industry Sector (Test Set)")
                            plt.ylabel("Fraud Rate")
                            plt.xticks(rotation=45, ha="right")  # Rotate labels
                            plt.tight_layout()
                            plt.savefig(
                                pred_file.parent / f"{pred_file.stem}_sector_fraud.png"
                            )
                            plt.close()
                        else:
                            print(
                                f"{Fore.YELLOW}No fraud data in sectors with > 5 samples for plotting in {pred_file}{Style.RESET_ALL}",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            f"{Fore.YELLOW}No sectors with > 5 samples for plotting in {pred_file}{Style.RESET_ALL}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"{Fore.YELLOW}No 'sic' column for plotting in {pred_file}{Style.RESET_ALL}",
                        file=sys.stderr,
                    )

                # Plot fraud distribution by quarter
                if "quarter" in df.columns and len(df["quarter"].unique()) > 1:
                    try:
                        # Use the sorted data prepared for the JSON summary if available

                        if (
                            "fraud_trend_by_quarter" in summary
                            and summary["fraud_trend_by_quarter"]
                        ):
                            quarters = list(summary["fraud_trend_by_quarter"].keys())
                            rates = list(summary["fraud_trend_by_quarter"].values())
                            plt.figure(figsize=(12, 6))
                            plt.plot(quarters, rates, marker="o", linestyle="-")
                            plt.title("Fraud Rate Trend by Quarter (Test Set)")
                            plt.ylabel("Fraud Rate")
                            plt.xlabel("Quarter")

                            plt.xticks(
                                rotation=0, fontsize=20, ha="right"
                            )  # Rotate labels

                            plt.grid(True, linestyle="--", alpha=0.7)
                            plt.tight_layout()
                            plt.savefig(
                                pred_file.parent / f"{pred_file.stem}_quarter_fraud.png"
                            )
                            plt.close()
                        else:
                            print(
                                f"{Fore.YELLOW}No quarter trend data for plotting in {pred_file}{Style.RESET_ALL}",
                                file=sys.stderr,
                            )
                    except Exception as e:
                        print(
                            f"{Fore.YELLOW}Error plotting quarter trend in {pred_file}: {e}{Style.RESET_ALL}",
                            file=sys.stderr,
                        )

            except Exception as e:
                print(
                    f"{Fore.RED}Error generating plots for {pred_file}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )

    except Exception as e:
        print(
            f"{Fore.RED}An unexpected error occurred while processing prediction file {prediction_file_name} in {sub_dir}: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )

    return summary


def compute_subgroup_performance(experiment_dir: Path):
    """Combine all test_predictions.csv across folds and compute worst-10 group performances."""
    all_dfs = []
    print(f"Computing subgroup performance for {experiment_dir}", file=sys.stderr)
    for fold_num in [1, 2, 3, 4, 5]:  # Iterate through folds
        sub_dir = experiment_dir / f"fold_{fold_num}"
        if not sub_dir.exists():
            continue
        # Look for the test_predictions.csv file
        test_files = list(sub_dir.rglob("test_predictions.csv"))
        if not test_files:
            # print(f"No test_predictions.csv found in {sub_dir}", file=sys.stderr) # Too verbose
            continue

        # Prefer the root test_predictions.csv if exists
        test_file = sub_dir / "test_predictions.csv"
        if not test_file.exists() and test_files:
            test_file = test_files[0]  # Fallback

        if test_file.exists():
            try:
                df = _load_and_standardize_predictions_df(test_file)
                if df is None:
                    continue

                # Add fold number for potential later analysis if needed (optional)
                df["fold"] = fold_num
                all_dfs.append(df)
            except Exception as e:
                print(
                    f"{Fore.RED}Error reading or processing {test_file}: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                continue

    if not all_dfs:
        print(
            f"{Fore.YELLOW}No valid test_predictions.csv files found across folds in {experiment_dir}. Skipping subgroup performance.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    print(
        f"Combined data from {len(all_dfs)} folds ({len(combined)} samples)",
        file=sys.stderr,
    )

    # Apply mapping functions *before* grouping
    __map_sic(combined)  # Maps to 'sic' column
    __map_cik_name(
        combined
    )  # Maps to 'cik' column (overwrites original CIK number with name)

    # derive fiscal year
    if "quarter" in combined.columns:
        combined["fiscal_year"] = combined["quarter"].astype(str).str[:4]
    else:
        print(
            f"{Fore.YELLOW}Warning: 'quarter' column not found. Skipping fiscal year subgroup analysis.{Style.RESET_ALL}",
            file=sys.stderr,
        )

    def _metrics(g):
        """Calculate subgroup metrics, handling zero division."""
        y_true = g["y_true_id"]
        y_pred = g["y_pred_id"]

        n_samples = len(g)
        n_fraud = int(y_true.sum())
        n_predicted_fraud = int(y_pred.sum())

        # Calculate confusion matrix for the subgroup
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        # Calculate metrics, handling zero division explicitly
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        auc = -1.0  # Default if not available or calculable
        if "fraud_probability" in g.columns and len(y_true.unique()) > 1:
            try:
                auc = roc_auc_score(y_true, g["fraud_probability"])
            except ValueError:  # e.g., only one class present
                pass  # Keep default -1.0

        fraud_rate = y_true.mean() if n_samples > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),  # This is TPR (True Positive Rate)
            "f1_score": float(f1),
            "auc": float(auc),  # Use auc if available
            "n_samples": int(n_samples),
            "n_actual_fraud": int(n_fraud),  # Renamed for clarity
            "n_predicted_fraud": int(n_predicted_fraud),  # Added predicted fraud count
            "n_fraud_detected": int(tp),  # TP count
            "n_false_positives": int(fp),  # FP count
            "n_true_negatives": int(tn),  # TN count
            "n_false_negatives": int(fn),  # FN count
            "fraud_rate": float(fraud_rate),
            # Add confusion matrix rates
            "tpr": float(recall),  # True Positive Rate = Recall
            "fpr": float(fp / (fp + tn))
            if (fp + tn) > 0
            else 0.0,  # False Positive Rate
            "tnr": float(tn / (tn + fp))
            if (tn + fp) > 0
            else 0.0,  # True Negative Rate
            "fnr": float(fn / (tp + fn))
            if (tp + fn) > 0
            else 0.0,  # False Negative Rate
        }

    results = {}
    results_to_plot = {}
    # Process subgroups, skipping if the column doesn't exist after mapping
    for subgroup_col in ["cik", "fiscal_year", "sic"]:
        if subgroup_col not in combined.columns:
            print(
                f"{Fore.YELLOW}Skipping subgroup analysis for '{subgroup_col}' as column is missing.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            continue

        print(f"Calculating metrics by {subgroup_col}...", file=sys.stderr)
        metrics_by_group = {}
        # Group by the mapped column
        grouped = combined.groupby(subgroup_col)
        for name, group in tqdm(
            grouped, desc=f"Computing metrics for {subgroup_col} groups"
        ):
            # Only compute metrics if the group has at least one actual fraud sample
            # This focuses the "worst/best" lists on groups with fraud
            if group["y_true_id"].sum() > 0:
                metrics_by_group[name] = _metrics(group)

        if not metrics_by_group:
            print(
                f"{Fore.YELLOW}No groups with actual fraud found for {subgroup_col}. Skipping analysis.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            results[subgroup_col] = {"worst_10": {}, "best_10": {}}  # Add empty entry
            continue

        top_k = 15 if subgroup_col == "fiscal_year" else 10

        worst_dict_to_plot = dict(
            sorted(metrics_by_group.items(), key=lambda x: x[1]["recall"])[:top_k]
        )
        # Pick 10 best by recall (FN is low, TP is high relative to actual fraud)
        # Sorting by 'recall' descending gets the best (highest recall)
        best_dict_to_plot = dict(
            sorted(
                metrics_by_group.items(), key=lambda x: x[1]["recall"], reverse=True
            )[:top_k]
        )

        worst_dict = dict(
            sorted(metrics_by_group.items(), key=lambda x: x[1]["recall"])
        )
        # Pick 10 best by recall (FN is low, TP is high relative to actual fraud)
        # Sorting by 'recall' descending gets the best (highest recall)
        best_dict = dict(
            sorted(metrics_by_group.items(), key=lambda x: x[1]["recall"], reverse=True)
        )

        results[subgroup_col] = {
            "worst_10_by_recall": worst_dict,
            "best_10_by_recall": best_dict,
        }

        results_to_plot[subgroup_col] = {
            "worst_10_by_recall": worst_dict_to_plot,
            "best_10_by_recall": best_dict_to_plot,
        }

        print(
            f"Finished calculating metrics by {subgroup_col}. Found {len(metrics_by_group)} groups with fraud.",
            file=sys.stderr,
        )

        # Generate and save plots (using the calculated metrics)
        plot_subgroup_performance(
            experiment_dir, subgroup_col, results_to_plot[subgroup_col]
        )

    out = experiment_dir / "subgroup_performance.json"  # Renamed output file
    try:
        with open(out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved subgroup performance to {out}", file=sys.stderr)
    except Exception as e:
        print(
            f"{Fore.RED}Error writing subgroup performance to {out}: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )


def plot_subgroup_performance(experiment_dir, subgroup, performance_data):
    """
    Generates and saves the stacked bar plot for fraud samples and detected frauds
    for worst/best groups by recall.
    """
    print(f"Generating subgroup performance plots for {subgroup}...", file=sys.stderr)
    for perf_type in ["worst_10_by_recall", "best_10_by_recall"]:  # Use updated keys
        data = performance_data.get(perf_type, {})
        if not data:
            print(
                f"{Fore.YELLOW}No data for {perf_type} in subgroup {subgroup}. Skipping plot.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            continue
        group_names = list(data.keys())

        if subgroup == "fiscal_year":
            group_names = list(sorted(group_names))

        if not group_names:
            print(
                f"{Fore.YELLOW}No groups found for {perf_type} in subgroup {subgroup}. Skipping plot.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            continue

        try:
            n_actual_fraud = [data[name]["n_actual_fraud"] for name in group_names]
            n_fraud_detected = [
                data[name]["n_fraud_detected"] for name in group_names
            ]  # This is TP
            n_false_negatives = [
                data[name]["n_false_negatives"] for name in group_names
            ]  # This is FN

            # Double check counts add up
            for i in range(len(group_names)):
                if n_fraud_detected[i] + n_false_negatives[i] != n_actual_fraud[i]:
                    print(
                        f"{Fore.YELLOW}Warning: Detected ({n_fraud_detected[i]}) + Undetected ({n_false_negatives[i]}) does not equal Actual Fraud ({n_actual_fraud[i]}) for group '{group_names[i]}' in {subgroup}. Check calculations.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )

            # Stacked Bar Plot for Actual Fraud vs. Detected/Undetected
            plt.figure(figsize=(12, 6))

            bar_width = 0.90

            wrapper = textwrap.TextWrapper(width=14)
            do_wrap = lambda x: "\n".join(wrapper.wrap(x))
            group_names_plot = [
                do_wrap(name) if len(name) > 14 else name for name in group_names
            ]

            index = np.arange(len(group_names))

            # Bars for Detected Fraud (TP) - bottom part of the stack
            bars_detected = plt.bar(
                index,
                n_fraud_detected,
                bar_width,
                label="Detected Fraud (TP)",
                color="#83d2d2",
            )  # Green color

            # Bars for Undetected Fraud (FN) - top part of the stack
            bars_undetected = plt.bar(
                index,
                n_false_negatives,
                bar_width,
                bottom=n_fraud_detected,
                label="Undetected Fraud (FN)",
                color="#d28383",
            )  # Red color

            # plt.xlabel(subgroup.title(), fontsize=18)
            # plt.ylabel("Number of Samples", fontsize=18)
            # plt.title(
            #     f"{perf_type.replace('_',' ').title()} - Fraud Detection by {subgroup.title()}",
            #     fontsize=14,
            # )
            plt.xticks(
                index,
                group_names_plot,
                rotation=90 if subgroup == "cik" else 45,
                ha="center",
                fontsize=22,
            )  # Rotate x-axis labels for readability
            plt.yticks(fontsize=22)
            plt.legend(fontsize=22)

            # Add value labels on top of each section of the bars
            def add_label(bars):
                for bar in bars:
                    height = bar.get_height()
                    # Only add label if height > 0 to avoid clutter

                    # Position the text in the middle of the bar segment
                    xval = bar.get_x() + bar.get_width() / 2
                    yval = bar.get_y() + height / 2

                    min_height = (
                        11 if subgroup == "sic" else 0 if subgroup == "cik" else 5
                    )

                    if height > min_height:
                        plt.text(
                            xval,
                            yval,
                            str(int(height)),
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=24,
                            weight="bold",
                        )

            add_label(bars_detected)
            add_label(bars_undetected)

            plt.tight_layout()
            plot_file = (
                experiment_dir
                / f"{subgroup}_{perf_type}_fraud_detection_stacked_bar.png"
            )
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved plot to {plot_file}", file=sys.stderr)
        except Exception as e:
            print(
                f"{Fore.RED}Error generating plot for {perf_type} subgroup {subgroup}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )


def compute_performance_by_token_count(experiment_dir: Path, n_bins=10):
    """
    Analyzes test set predictions and computes performance metrics
    (TPR, FPR, TNR, FNR) as a function of MDA token count.
    Combines data across all folds and generates a faceted plot.
    """
    print(f"Computing performance by token count for {experiment_dir}", file=sys.stderr)
    all_dfs = []
    for fold_num in [1, 2, 3, 4, 5]:
        sub_dir = experiment_dir / f"fold_{fold_num}"
        if not sub_dir.exists():
            continue
        test_files = list(sub_dir.rglob("test_predictions.csv"))
        if not test_files:
            continue

        test_file = sub_dir / "test_predictions.csv"
        if not test_file.exists() and test_files:
            test_file = test_files[0]

        if test_file.exists():
            try:
                df = _load_and_standardize_predictions_df(test_file)
                if df is None:
                    continue

                # Ensure cik_original is present and integer type for lookup
                if "cik_original" not in df.columns:
                    print(
                        f"{Fore.YELLOW}Skipping {test_file} for token count analysis: 'cik_original' column missing.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    continue
                df["cik_original"] = df["cik_original"].astype(int)  # Ensure it's int
                df["quarter"] = df["quarter"].astype(str)  # Ensure quarter is string

                all_dfs.append(df)
            except Exception as e:
                print(
                    f"{Fore.RED}Error reading or processing {test_file} for token count analysis: {e}{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                continue

    if not all_dfs:
        print(
            f"{Fore.YELLOW}No valid test_predictions.csv files found across folds in {experiment_dir} for token count analysis. Skipping.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(
        f"Combined data for token count analysis from {len(all_dfs)} folds ({len(combined_df)} samples).",
        file=sys.stderr,
    )

    try:

        def _format_mda_dict_key(row):
            cik_original = row["cik_original"]
            if pd.notna(cik_original):
                company_name = CIK_TO_COMPANY_NAME.get(
                    int(cik_original), f"Unknown_CIK_{int(cik_original)}"
                )
                return f"{company_name}_{row['quarter']}"
            else:
                return None

        combined_df["mda_dict_key"] = combined_df.apply(_format_mda_dict_key, axis=1)
    except Exception as e:
        print(
            f"{Fore.RED}Error creating MDA dictionary keys: {e}. Cannot perform token count analysis.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    # Map token counts
    combined_df["token_count"] = (
        combined_df["mda_dict_key"]
        .map(TOKEN_PER_SUMMARIZED_MDA_DICT)
        .fillna(-1)
        .astype(int)
    )  # Use -1 for missing tokens initially
    # Drop rows where token count mapping failed or was not found (-1 sentinel)
    df_with_tokens = combined_df[combined_df["token_count"] != -1].copy()

    # --- Debugging Print 1 ---
    print(
        f"DEBUG: After mapping token counts and dropping NaNs: {len(df_with_tokens)} samples remaining.",
        file=sys.stderr,
    )
    if not df_with_tokens.empty:
        print(
            f"DEBUG: Sample data with token counts:\n{df_with_tokens.head()}",
            file=sys.stderr,
        )
        print(
            f"DEBUG: Token count distribution:\n{df_with_tokens['token_count'].describe()}",
            file=sys.stderr,
        )
    # --- End Debugging Print 1 ---

    if df_with_tokens.empty:
        print(
            f"{Fore.YELLOW}No samples with valid token counts found in {experiment_dir}. Skipping token count analysis.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    try:
        unique_token_counts = df_with_tokens["token_count"].unique()
        if len(unique_token_counts) < 2:
            print(
                f"{Fore.YELLOW}Only {len(unique_token_counts)} unique token counts found. Cannot create bins for token count analysis.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            return
        elif len(unique_token_counts) < n_bins:
            print(
                f"{Fore.YELLOW}Fewer unique token counts ({len(unique_token_counts)}) than bins ({n_bins}) requested. Using fewer bins.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            n_bins = len(unique_token_counts)

        # Use pd.qcut to create quantile bins
        df_with_tokens["token_bin"], bins = pd.qcut(
            df_with_tokens["token_count"],
            q=n_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        actual_n_bins = df_with_tokens["token_bin"].nunique()
        if actual_n_bins < n_bins:
            print(
                f"{Fore.YELLOW}Reduced number of token count bins to {actual_n_bins} due to duplicates in quantiles.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            n_bins = actual_n_bins

        # Create descriptive bin labels (e.g., "0-100", "101-500", etc.)
        bin_ranges = []
        for i in range(len(bins) - 1):
            # just take the min
            bin_ranges.append(f"{int(bins[i])}")

        # Map bin index to descriptive label
        bin_label_map = {i: label for i, label in enumerate(bin_ranges)}
        df_with_tokens["token_bin_label"] = df_with_tokens["token_bin"].map(
            bin_label_map
        )

        # Also get the average token count for potential ordering on x-axis, or use bin index
        bin_avg_token_count = (
            df_with_tokens.groupby("token_bin")["token_count"].mean().to_dict()
        )

        # --- Debugging Print 2 ---
        print(f"DEBUG: After creating {n_bins} token bins:", file=sys.stderr)
        print(f"DEBUG: Bin ranges: {bin_ranges}", file=sys.stderr)
        print(
            f"DEBUG: Samples per bin:\n{df_with_tokens['token_bin_label'].value_counts().sort_index()}",
            file=sys.stderr,
        )
        # print(f"DEBUG: Sample data with bins:\n{df_with_tokens.head()}", file=sys.stderr) # Can be verbose
        # --- End Debugging Print 2 ---

    except Exception as e:
        print(
            f"{Fore.RED}Error creating token count bins: {e}. Cannot perform token count analysis.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    # Calculate metrics per bin
    performance_data_for_plot = []  # List to store data for plotting DataFrame
    results_for_json = {}  # Dict to store results for JSON output

    if "token_bin" in df_with_tokens.columns:
        grouped_by_bin = df_with_tokens.groupby("token_bin")

        # --- Debugging Print 3 ---
        print(
            f"DEBUG: Calculating metrics for {len(grouped_by_bin)} token bins...",
            file=sys.stderr,
        )
        # --- End Debugging Print 3 ---

        for bin_idx, group in grouped_by_bin:
            bin_label = bin_label_map.get(bin_idx, f"Bin {bin_idx}")
            avg_tokens = bin_avg_token_count.get(bin_idx, 0)

            y_true = group["y_true_id"]
            y_pred = group["y_pred_id"]
            n_samples = len(group)
            n_actual_fraud = int(y_true.sum())

            if n_samples == 0:
                print(
                    f"DEBUG: Skipping empty bin {bin_label}", file=sys.stderr
                )  # Should ideally not happen with qcut/drop
                continue

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            # Calculate rates, handling division by zero
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

            # Store results for JSON output
            results_for_json[bin_label] = {
                "avg_token_count": float(avg_tokens),
                "n_samples": int(n_samples),
                "n_actual_fraud": int(n_actual_fraud),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "tnr": float(tnr),
                "fnr": float(fnr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "recall": float(tpr),
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "f1_score": 0.0
                if (tp + fp) == 0
                else float(  # Calculate F1 from TP, FP, FN
                    (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                ),
            }

            # Prepare data for plotting DataFrame (long format)
            performance_data_for_plot.append(
                {
                    "token_bin_label": bin_label,
                    "avg_token_count": float(avg_tokens),
                    "Metric": "True Positive Rate (Recall)",
                    "Rate": float(tpr),
                }
            )
            performance_data_for_plot.append(
                {
                    "token_bin_label": bin_label,
                    "avg_token_count": float(avg_tokens),
                    "Metric": "False Positive Rate",
                    "Rate": float(fpr),
                }
            )
            performance_data_for_plot.append(
                {
                    "token_bin_label": bin_label,
                    "avg_token_count": float(avg_tokens),
                    "Metric": "True Negative Rate",
                    "Rate": float(tnr),
                }
            )
            performance_data_for_plot.append(
                {
                    "token_bin_label": bin_label,
                    "avg_token_count": float(avg_tokens),
                    "Metric": "False Negative Rate",
                    "Rate": float(fnr),
                }
            )

        # --- Debugging Print 4 ---
        print(
            f"DEBUG: Prepared {len(performance_data_for_plot)} data points for plotting.",
            file=sys.stderr,
        )
        if performance_data_for_plot:
            print(
                f"DEBUG: Sample plotting data:\n{performance_data_for_plot[:8]}",
                file=sys.stderr,
            )  # Print first few entries
        # --- End Debugging Print 4 ---

        # Sort JSON results by average token count for consistency
        results_for_json_sorted = dict(
            sorted(
                results_for_json.items(), key=lambda item: item[1]["avg_token_count"]
            )
        )
        results_output = {"performance_by_token_bin": results_for_json_sorted}

        # Save results to JSON
        results_file = experiment_dir / "performance_by_token_count.json"
        try:
            with open(results_file, "w") as f:
                json.dump(results_output, f, indent=4)
            print(
                f"Saved performance by token count to {results_file}", file=sys.stderr
            )
        except Exception as e:
            print(
                f"{Fore.RED}Error writing performance by token count to {results_file}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )

        # Generate and save faceted plot using Seaborn
        try:
            if performance_data_for_plot:
                plot_df = pd.DataFrame(performance_data_for_plot)

                # --- Debugging Print 5 ---
                print(
                    f"DEBUG: DataFrame for plotting created. Shape: {plot_df.shape}",
                    file=sys.stderr,
                )
                print(
                    f"DEBUG: Sample plot DataFrame:\n{plot_df.head()}", file=sys.stderr
                )
                print(
                    f"DEBUG: Plot DataFrame columns: {plot_df.columns.tolist()}",
                    file=sys.stderr,
                )
                print(
                    f"DEBUG: Unique Metrics in plot_df: {plot_df['Metric'].unique().tolist()}",
                    file=sys.stderr,
                )
                print(
                    f"DEBUG: Unique Bin Labels in plot_df: {plot_df['token_bin_label'].unique().tolist()}",
                    file=sys.stderr,
                )
                # --- End Debugging Print 5 ---

                # Ensure the order of bin labels on the x-axis matches the average token count
                # Create a categorical type with the correct order
                # Use the keys from the sorted JSON results for ordered bin labels
                ordered_bins_labels = list(results_for_json_sorted.keys())

                if not plot_df["token_bin_label"].isin(ordered_bins_labels).all():
                    print(
                        f"{Fore.YELLOW}Warning: Some token bin labels in plot_df are not in ordered_bins_labels. Check bin mapping.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    # Filter plot_df to only include labels in ordered_bins_labels to avoid errors with categorical
                    plot_df = plot_df[
                        plot_df["token_bin_label"].isin(ordered_bins_labels)
                    ].copy()

                # Ensure there's still data after filtering
                if plot_df.empty:
                    print(
                        f"{Fore.YELLOW}Plotting DataFrame is empty after filtering by ordered bin labels. Skipping plot.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    plt.close()  # Close any open figure from FacetGrid attempt
                    return

                plot_df["token_bin_label"] = pd.Categorical(
                    plot_df["token_bin_label"],
                    categories=ordered_bins_labels,
                    ordered=True,
                )

                # Define the order of metrics for consistent plotting
                metric_order = [
                    "True Positive Rate (Recall)",
                    "False Positive Rate",
                    "True Negative Rate",
                    "False Negative Rate",
                ]
                # Ensure all metrics intended for plotting are actually in the DataFrame
                available_metrics = plot_df["Metric"].unique().tolist()
                metric_order_filtered = [
                    m for m in metric_order if m in available_metrics
                ]

                if not metric_order_filtered:
                    print(
                        f"{Fore.YELLOW}No plotable metrics found in the DataFrame. Skipping plot.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    plt.close()
                    return

                plot_df["Metric"] = pd.Categorical(
                    plot_df["Metric"], categories=metric_order_filtered, ordered=True
                )

                # Create the faceted plot
                # Check if we have enough unique values in the 'col' variable ('Metric') to create a grid
                if (
                    len(plot_df["Metric"].unique()) > 0
                    and len(plot_df["token_bin_label"].unique()) > 0
                ):

                    sns.set(style="whitegrid")
                    g = sns.FacetGrid(
                        plot_df,
                        col="Metric",
                        col_wrap=2,
                        height=4,
                        aspect=1.5,
                        sharey=False,
                    )
                    g.map_dataframe(
                        sns.lineplot, "token_bin_label", "Rate", marker="o", ci="sd"
                    )

                    g.fig.suptitle(
                        "Performance Rates vs. MDA Token Count", fontsize=16, y=1.03
                    )
                    # g.set_axis_labels("MDA Token Count Bin Range", "Rate", fontsize=10)
                    # Set seaborn style

                    # Rotate x-axis labels for readability
                    for ax in g.axes.flatten():
                        for label in ax.get_xticklabels():
                            label.set_rotation(45)

                    # Set large font for tickand axis labels
                    g.set_xticklabels(fontsize=14)
                    g.set_yticklabels(fontsize=14)

                    plt.tight_layout()
                    plot_file = (
                        experiment_dir / "performance_by_token_count_faceted_plot.png"
                    )
                    plt.savefig(plot_file)
                    plt.close()
                    print(
                        f"Saved performance by token count faceted plot to {plot_file}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"{Fore.YELLOW}Not enough unique metrics or bins for faceted plotting for {experiment_dir}. Skipping plot.{Style.RESET_ALL}",
                        file=sys.stderr,
                    )
                    plt.close()  # Ensure plot is closed even if not saved

            else:
                print(
                    f"{Fore.YELLOW}Plotting DataFrame is empty. No performance data by token count to plot for {experiment_dir}.{Style.RESET_ALL}",
                    file=sys.stderr,
                )

        except Exception as e:
            print(
                f"{Fore.RED}Error generating performance by token count faceted plot: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc(
                file=sys.stderr
            )  # Print full traceback for plotting errors
            plt.close()  # Ensure plot is closed on error

    else:
        print(
            f"{Fore.YELLOW}Token binning failed for {experiment_dir}. Skipping performance calculation by bin.{Style.RESET_ALL}",
            file=sys.stderr,
        )


def process_all_experiments_dir(root_dir: Path):
    """
    Recursively find all experiment directories under root_dir
    and compute the average metrics and subgroup/token performance for each experiment.
    """
    print(f"Scanning for experiment directories under {root_dir}", file=sys.stderr)

    all_experiment_dirs = []
    for dir in root_dir.rglob("*"):
        if dir.is_dir() and "fold_1" in [d.name for d in dir.iterdir()]:
            all_experiment_dirs.append(dir)
    print(
        f"Found {len(all_experiment_dirs)} potential experiment directories.",
        file=sys.stderr,
    )

    def process_single_dir(exp_dir):
        print(
            f"\n{Fore.BLUE}--- Processing Experiment: {exp_dir} ---{Style.RESET_ALL}",
            file=sys.stderr,
        )
        try:
            # 1. Compute average overall metrics (test and validation)
            compute_auc_single_experiment_best_validation_metrics(exp_dir)
            compute_auc_single_experiment_best_test_metrics(exp_dir)

            # 2. Analyze prediction files (distributions, unseen metrics)
            analyze_prediction_files(exp_dir)

            # 3. Compute subgroup performance (CIK, Year, SIC)
            # Check if RCMA before running subgroup/token analysis
            if "rcma" in str(exp_dir).lower():
                print(
                    f"{Fore.YELLOW}Skipping subgroup and token count analysis for RCMA experiment.{Style.RESET_ALL}",
                    file=sys.stderr,
                )
            else:
                compute_subgroup_performance(exp_dir)

                # 4. Compute performance by MDA token count
                compute_performance_by_token_count(exp_dir)

                analyze_misstatement_performance(exp_dir)

        except Exception as e:
            print(
                f"{Fore.RED}An unexpected error occurred while processing experiment {exp_dir}: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc(
                file=sys.stderr
            )  # Print traceback for unexpected errors

    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(process_single_dir)(exp_dir)
        for exp_dir in tqdm(all_experiment_dirs, desc="Processing experiments")
    )

    print(
        f"\n{Fore.GREEN}--- Finished processing all experiments ---{Style.RESET_ALL}",
        file=sys.stderr,
    )


# ======================================================================
# START: NEW FUNCTIONS FOR MISSTATEMENT ANALYSIS
# ======================================================================


def plot_misstatement_performance(experiment_dir, performance_results):
    """Generates and saves plots for misstatement performance analysis."""
    print(
        f"Generating misstatement performance plots for {experiment_dir}",
        file=sys.stderr,
    )

    plot_data = []
    # Ensure misstatement types are in the same order as LIST_MISTATEMENT_TYPE_FOR_TRAINING
    # for consistent plotting order
    for m_type in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
        metrics = performance_results.get("per_misstatement", {}).get(m_type)
        if metrics and metrics.get("num_fraud_cases", 0) > 0:
            plot_data.append({"misstatement_type": m_type, **metrics})

    if not plot_data:
        print(
            f"{Fore.YELLOW}No misstatement data to plot for {experiment_dir}.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    df = pd.DataFrame(
        plot_data
    )  # Do not sort here, rely on LIST_MISTATEMENT_TYPE_FOR_TRAINING order for first two plots

    wrapper = textwrap.TextWrapper(
        width=20, break_long_words=False, break_on_hyphens=False
    )
    df["misstatement_label"] = df["misstatement_type"].apply(
        lambda x: "\n".join(wrapper.wrap(x.replace("mis_", " ").title()))
    )

    # Reorder DataFrame based on the custom labels list
    ordered_labels = [
        "\n".join(wrapper.wrap(x.replace("mis_", " ").title()))
        for x in LIST_MISTATEMENT_TYPE_FOR_TRAINING
    ]
    df["misstatement_label"] = pd.Categorical(
        df["misstatement_label"], categories=ordered_labels, ordered=True
    )
    df_sorted_for_plots = df.sort_values(
        "misstatement_label"
    )  # Sort by the categorical order

    # Plot 1: Recall Rate per Misstatement Type
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=df_sorted_for_plots, x="misstatement_label", y="recall", palette="viridis"
    )
    plt.title("Recall Rate per Misstatement Type", fontsize=18, weight="bold")
    plt.ylabel("Recall Rate", fontsize=14)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(experiment_dir / "misstatement_recall_performance.png")
    plt.close()

    # Plot 2: AUC Score per Misstatement Type
    plt.figure(figsize=(14, 8))
    # Sort by AUC for this specific plot for better visualization
    sns.barplot(
        data=df_sorted_for_plots.sort_values(
            "auc", ascending=False
        ),  # Sort for this specific plot
        x="misstatement_label",
        y="auc",
        palette="plasma",
    )
    plt.title("AUC Score per Misstatement Type", fontsize=18, weight="bold")
    plt.ylabel("AUC Score", fontsize=14)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(experiment_dir / "misstatement_auc_performance.png")
    plt.close()

    # Plot 3: Stacked Bar Chart for Detected vs. Undetected Fraud
    plt.figure(figsize=(14, 9))
    # Use df_sorted_for_plots to maintain consistent x-axis order with other plots
    df_plot_stacked = df_sorted_for_plots[
        ["misstatement_label", "num_detected_cases", "num_undetected_cases", "auc"]
    ].set_index("misstatement_label")

    ax = df_plot_stacked[["num_detected_cases", "num_undetected_cases"]].plot(
        kind="bar", stacked=True, color=["#83d2d2", "#d28383"], ax=plt.gca(), width=0.8
    )
    #
    plt.ylabel("Number of Fraud Cases", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=20)

    # Add AUC values on top of each bar stack
    for i, p in enumerate(ax.patches):
        # We need to get the total height of the stacked bar for positioning
        # The 'patches' list contains bars for 'num_detected_cases' then 'num_undetected_cases'
        # So, for the i-th misstatement_label, the total height is p (for detected) + next p (for undetected)
        if i < len(df_plot_stacked):  # This is for 'num_detected_cases' bars
            total_height = (
                p.get_height() + df_plot_stacked["num_undetected_cases"].iloc[i]
            )
            auc_value = df_plot_stacked["auc"].iloc[i]

            # Place text above the total height, slightly offset
            ax.text(
                p.get_x() + p.get_width() / 2.0,
                total_height
                + (ax.get_ylim()[1] * 0.02),  # 2% of y-axis height as offset
                f"AUC: {auc_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=16,
                color="black",
                weight="bold",
            )

    plt.yticks(fontsize=20)
    plt.legend(["Detected Fraud (TP)", "Undetected Fraud (FN)"], fontsize=18)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(experiment_dir / "misstatement_detection_composition.png")
    plt.close()

    print(f"Saved misstatement performance plots to {experiment_dir}", file=sys.stderr)


def analyze_misstatement_performance(experiment_dir: Path):
    """
    Aggregates test predictions across all folds to analyze performance per misstatement type.
    """
    print(
        f"Analyzing performance per misstatement type for {experiment_dir}",
        file=sys.stderr,
    )

    all_preds_dfs = []
    all_truth_dfs = []
    dataset_version = next(
        (v for v in ["company_isolated_splitting", "time_splitting", "time_splitting"] if v in str(experiment_dir)), "company_isolated_splitting"
    )

    for fold_num in range(1, 6):
        sub_dir = experiment_dir / f"fold_{fold_num}"
        if not sub_dir.exists():
            continue

        pred_file_candidates = list(sub_dir.rglob("test_predictions.csv"))
        pred_file = None
        # Prioritize the one directly in sub_dir
        if (sub_dir / "test_predictions.csv").exists():
            pred_file = sub_dir / "test_predictions.csv"
        elif pred_file_candidates:
            pred_file = pred_file_candidates[0]  # Take the first one if not in root

        if not pred_file:  # If still no file found
            print(
                f"{Fore.YELLOW}Skipping fold {fold_num}: test_predictions.csv not found.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            continue

        try:
            pred_df = _load_and_standardize_predictions_df(pred_file)
            if pred_df is None:
                print(
                    f"{Fore.YELLOW}Failed to load and standardize predictions for fold {fold_num}. Skipping.{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                continue

            _, test_path = load_cross_validation_path(
                {"dataset_version": dataset_version, "fold_id": fold_num}
            )

            # Ensure only columns from LIST_MISTATEMENT_TYPE_FOR_TRAINING are loaded and they exist
            existing_misstatement_cols = [
                col
                for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING
                if col in pd.read_csv(test_path, nrows=1).columns
            ]
            if not existing_misstatement_cols:
                print(
                    f"{Fore.YELLOW}No misstatement type columns found in {test_path}. Skipping misstatement analysis for this fold.{Style.RESET_ALL}",
                    file=sys.stderr,
                )
                continue

            truth_df = pd.read_csv(test_path, usecols=existing_misstatement_cols)

            if len(pred_df) == len(truth_df):
                all_preds_dfs.append(pred_df)
                all_truth_dfs.append(truth_df)
            else:
                print(
                    f"{Fore.RED}Row count mismatch in fold {fold_num} between predictions ({len(pred_df)}) and truth ({len(truth_df)}). Skipping fold.{Style.RESET_ALL}",
                    file=sys.stderr,
                )

        except Exception as e:
            print(
                f"{Fore.RED}Error processing fold {fold_num} for misstatement analysis: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )

    if not all_preds_dfs:
        print(
            f"{Fore.YELLOW}No valid data for misstatement analysis in {experiment_dir}.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        return

    combined_preds = pd.concat(all_preds_dfs, ignore_index=True)
    combined_truth = pd.concat(all_truth_dfs, ignore_index=True)
    combined_df = pd.concat([combined_preds, combined_truth], axis=1)

    performance_results = {}

    # Calculate overall false positives and non-fraud instances once
    non_fraud_df = combined_df[combined_df["y_true_id"] == 0].copy()
    total_false_positives = int(
        ((combined_df["y_true_id"] == 0) & (combined_df["y_pred_id"] == 1)).sum()
    )

    for (
        misstatement_col
    ) in (
        LIST_MISTATEMENT_TYPE_FOR_TRAINING
    ):  # Iterate through the full list, even if column wasn't found in a specific fold
        # Filter for rows that ARE this misstatement type (value is 1)
        # And ensure y_true_id is 1 (actual fraud) for the misstatement performance part
        if misstatement_col not in combined_df.columns:
            print(
                f"{Fore.YELLOW}Misstatement column '{misstatement_col}' not found in combined dataframe. Skipping.{Style.RESET_ALL}",
                file=sys.stderr,
            )
            performance_results[misstatement_col] = {
                "num_fraud_cases": 0,
                "num_detected_cases": 0,
                "num_undetected_cases": 0,
                "recall": 0.0,
                "auc": -1.0,
            }
            continue

        subset_df_fraud_only = combined_df[
            (combined_df[misstatement_col] == 1) & (combined_df["y_true_id"] == 1)
        ].copy()

        num_fraud_cases = len(subset_df_fraud_only)

        if num_fraud_cases == 0:
            performance_results[misstatement_col] = {
                "num_fraud_cases": 0,
                "num_detected_cases": 0,
                "num_undetected_cases": 0,
                "recall": 0.0,
                "auc": -1.0,
            }
            continue

        detected_cases = int(subset_df_fraud_only["y_pred_id"].sum())
        undetected_cases = num_fraud_cases - detected_cases

        # Recall is only on the fraud instances of this misstatement type
        recall = recall_score(
            subset_df_fraud_only["y_true_id"],
            subset_df_fraud_only["y_pred_id"],
            zero_division=0,
        )

        auc = -1.0
        # For AUC, we need both positive (fraud for this misstatement type) AND negative (non-fraud overall) cases
        # to calculate a meaningful ROC AUC curve.
        if "fraud_probability" in combined_df.columns:
            # Create the specific AUC evaluation set: fraud cases of this type + all non-fraud cases
            auc_eval_df = pd.concat(
                [subset_df_fraud_only, non_fraud_df], ignore_index=True
            )

            # Ensure there are at least two unique true labels for AUC calculation
            if auc_eval_df["y_true_id"].nunique() > 1:
                try:
                    auc = roc_auc_score(
                        auc_eval_df["y_true_id"], auc_eval_df["fraud_probability"]
                    )
                except ValueError:
                    # This happens if there's only one class after concat for some reason, or other issues
                    auc = -1.0
                    print(
                        f"{Fore.YELLOW}Warning: AUC calculation failed for misstatement type '{misstatement_col}'. Data might be singular. {Style.RESET_ALL}",
                        file=sys.stderr,
                    )
            else:
                auc = -1.0
                print(
                    f"{Fore.YELLOW}Warning: Only one class found in AUC evaluation set for misstatement type '{misstatement_col}'. AUC cannot be calculated. {Style.RESET_ALL}",
                    file=sys.stderr,
                )

        performance_results[misstatement_col] = {
            "num_fraud_cases": num_fraud_cases,
            "num_detected_cases": detected_cases,
            "num_undetected_cases": undetected_cases,
            "recall": float(recall),
            "auc": float(auc),
        }

    if performance_results:
        # Filter out misstatement types with 0 fraud cases before sorting for most/least detected
        # and ensure there's at least one valid entry.
        valid_results = {
            k: v for k, v in performance_results.items() if v["num_fraud_cases"] > 0
        }

        most_detected = ("N/A", {"recall": 0})
        least_detected = ("N/A", {"recall": 0})

        if valid_results:
            sorted_by_recall = sorted(
                valid_results.items(),
                key=lambda item: item[1]["recall"],
                reverse=True,
            )
            most_detected = sorted_by_recall[0]
            least_detected = sorted_by_recall[-1]

        final_report = {
            "summary": {
                "most_detected_misstatement": {
                    "type": most_detected[0],
                    "recall": most_detected[1].get("recall"),
                },
                "least_detected_misstatement": {
                    "type": least_detected[0],
                    "recall": least_detected[1].get("recall"),
                },
                "total_samples_in_test_set": len(combined_df),
                "total_fraud_samples_in_test_set": int(combined_df["y_true_id"].sum()),
                "total_false_positives_in_test_set": total_false_positives,
            },
            "per_misstatement": performance_results,
        }

        out_file = experiment_dir / "misstatement_performance.json"
        try:
            with open(out_file, "w") as f:
                json.dump(final_report, f, indent=4)
            print(
                f"Saved misstatement performance analysis to {out_file}",
                file=sys.stderr,
            )
            plot_misstatement_performance(experiment_dir, final_report)
        except Exception as e:
            print(
                f"{Fore.RED}Error writing misstatement performance file or plots: {e}{Style.RESET_ALL}",
                file=sys.stderr,
            )


# ======================================================================
# END: NEW FUNCTIONS
# ======================================================================


if __name__ == "__main__":
    # Define the root directory to scan for experiments
    # This should be the parent directory containing your experiment folders
    # Example: If experiments are in /path/to/results/exp1, /path/to/results/exp2, etc.
    # then ROOT_DIR should be /path/to/results/
    # Adjust this Path according to your directory structure
    # ROOT_DIR = Path(__file__).parent.parent.parent.parent / "results" # Example assuming 'results' is up 4 levels
    # If running this script from the root of your research project, maybe:
    # ROOT_DIR = Path("./results") # Assuming a 'results' folder at the project root

    # Or specify a hardcoded path for testing
    # Example:
    exp_root_dir = ROOT_DIR / "experiments_aaai"

    # Check if the root directory exists
    if not exp_root_dir.exists():
        print(
            f"{Fore.RED}Error: Root directory for experiments not found at {exp_root_dir}{Style.RESET_ALL}"
        )
        sys.exit(1)  # Exit script if the root directory doesn't exist

    print(
        f"Processing experiment results starting from root directory: {exp_root_dir}",
        file=sys.stderr,
    )
    process_all_experiments_dir(exp_root_dir)
