import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
from scipy.stats import wilcoxon
import logging
from datetime import datetime
import json
import sys
import re # Import the regex module

# Assume ROOT_DIR is correctly defined and accessible from researchpkg.anomaly_detection.config
from researchpkg.anomaly_detection.config import ROOT_DIR

# --- Configuration ---
# Le répertoire racine de vos expériences. Ce sera le point de départ de la recherche récursive.
# EX: Si vos dossiers d'expérience sont:
# /some_base_path/llm_softmax_fraud_V4_fin/Fino_vbalal/fold_1/20240101_100000/test_metrics.json
# Alors EXP_ROOT_DIR doit être /some_base_path/
EXP_ROOT_DIR = ROOT_DIR /"experiments_aaai"

WILCOXON_OUTPUT_DIR = Path("./wilcoxon_analysis_v4_agnostic_robust")
LOG_FILE = WILCOXON_OUTPUT_DIR / "wilcoxon_analysis_v4_agnostic_robust.log"
SIGNIFICANCE_LEVEL = 0.05

# For filtering. This pattern must be present in one of the folder names identified as part of the config.
TARGET_DATASET_VERSION_PATTERN = 'v4' 

# Configure logging
WILCOXON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout) # Use sys.stdout for console output
    ]
)

logging.info(f"Starting Wilcoxon analysis. Logs will be saved to: {LOG_FILE}")
logging.info(f"Analysis results will be saved in: {WILCOXON_OUTPUT_DIR}")
logging.info(f"Searching for experiments containing '{TARGET_DATASET_VERSION_PATTERN}' in their config paths.")

# Regex to find a fold folder (e.g., 'fold_1', 'fold_2', etc.)
FOLD_FOLDER_REGEX = re.compile(r'fold_(\d+)')


def parse_path_for_metrics(metrics_file_path: Path, exp_root_path: Path) -> dict | None:
    """
    Parses a test_metrics.json file path to extract experiment configuration details
    agnostically, by first identifying the 'fold_ID' folder and then its parents.
    
    Expected structure (flexibly):
    ... / FIRST_LEVEL_FOLDER / SECOND_LEVEL_FOLDER / fold_ID / TIMESTAMP / test_metrics.json
    """
    try:
        # Get the parts of the absolute path
        abs_parts = metrics_file_path.absolute().parts
        
        
        # Find the index of the 'fold_X' folder
        fold_idx = -1
        fold_id = -1
        for i, part in enumerate(abs_parts):
            match = FOLD_FOLDER_REGEX.match(part)
            if match:
                fold_idx = i
                fold_id = int(match.group(1))
                break
        
        if fold_idx == -1:
            logging.warning(f"Could not find a 'fold_ID' folder in path: {metrics_file_path}. Skipping.")
            return None

        # Check if there are enough preceding folders for the config (at least 2: FIRST_LEVEL, SECOND_LEVEL)
        if fold_idx < 2:
            logging.warning(f"Path '{metrics_file_path}' has insufficient preceding folders before 'fold_{fold_id}' for config. Skipping.")
            return None

        # Extract the two folders preceding the fold folder
        second_level_folder = abs_parts[fold_idx - 1] # e.g., 'Fino_vbalal'
        first_level_folder = abs_parts[fold_idx - 2]  # e.g., 'llm_softmax_fraud_V4_fin'

        # Check if this path matches the target dataset version pattern
        if TARGET_DATASET_VERSION_PATTERN not in first_level_folder and \
           TARGET_DATASET_VERSION_PATTERN not in second_level_folder:
            logging.info(f"Skipping '{metrics_file_path}': Does not contain '{TARGET_DATASET_VERSION_PATTERN}'.")
            return None

        # The unique key for each experiment/model configuration will be the combination
        # of the first two folder names (relative to the context of the fold).
        model_config_combo_name = f"{first_level_folder}/{second_level_folder}"
        model_config_combo_name=model_config_combo_name.replace("llm_softmax_fraud_classifier_dataset_","")
        return {
            "model_config_combo_name": model_config_combo_name, # Unique identifier for comparison
            "fold_id": fold_id,
            "metrics_path": str(metrics_file_path)
        }
    except Exception as e:
        
        logging.info(f"Error parsing path '{metrics_file_path}': {e}. Skipping.", exc_info=False)
        return None

# --- Re-using the standardize_metrics function (identical to previous version) ---
def standardize_metrics(content):
    """Standardize metric names and calculate derived metrics if needed."""
    if "f1" in content and "f1_fraud_optimized" not in content:
        content["f1_fraud_optimized"] = content["f1"]
        if "precision" in content and "precision_fraud" not in content:
            content["precision_fraud"] = content["precision"]
        if "recall" in content and "recall_fraud" not in content:
            content["recall_fraud"] = content["recall"]

    if "confusion_matrix" in content and isinstance(content["confusion_matrix"], dict):
        cm_keys = ["true_positives", "false_positives", "true_negatives", "false_negatives"]
        if all(k in content["confusion_matrix"] for k in cm_keys):
            tp = content["confusion_matrix"]["true_positives"]
            fp = content["confusion_matrix"]["false_positives"]
            tn = content["confusion_matrix"]["true_negatives"]
            fn = content["confusion_matrix"]["false_negatives"]

            f1_positive = ((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
            f1_negative = ((2 * tn) / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0)

            content["macro_f1"] = (f1_positive + f1_negative) / 2.0

            support_positive = tp + fn
            support_negative = tn + fp
            total_support = support_positive + support_negative

            if total_support > 0:
                weighted_f1 = (support_positive * f1_positive + support_negative * f1_negative) / total_support
                content["weighted_f1"] = weighted_f1
            else:
                content["weighted_f1"] = 0.0
        else:
            content["macro_f1"] = -1
            content["weighted_f1"] = -1
    elif "macro_f1" not in content:
        content["macro_f1"] = -1
        content["weighted_f1"] = -1

    auc_score = content.get("auc_score", content.get("auc", -1.0))
    f1_score_fraud = content.get("f1_fraud_optimized", content.get("f1_fraud", content.get("f1_score", -1.0)))
    accuracy = content.get("accuracy", -1.0)
    precision = content.get("precision_fraud", content.get("precision", -1.0))
    recall = content.get("recall_fraud", content.get("recall", -1.0))
    macro_f1 = content.get("macro_f1", -1.0)
    weighted_f1 = content.get("weighted_f1", -1.0)
    threshold = content.get("best_threshold", content.get("threshold_used", content.get("threshold", -1.0)))

    return {
        "auc_score": auc_score,
        "f1_score_fraud": f1_score_fraud,
        "accuracy": accuracy,
        "precision_fraud": precision,
        "recall_fraud": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "threshold": threshold,
    }


def collect_per_fold_auc_scores(exp_root_dir: Path) -> pd.DataFrame:
    """
    Collects AUC scores for each fold from 'test_metrics.json' files,
    filtering by target dataset version pattern.
    """
    logging.info(f"Collecting per-fold AUC scores from '{exp_root_dir}' with dataset version pattern '{TARGET_DATASET_VERSION_PATTERN}'...")
    all_fold_aucs = []
    metrics_files = list(exp_root_dir.rglob("test_metrics.json"))
    logging.info(f"Found {len(metrics_files)} 'test_metrics.json' files in '{exp_root_dir}'.")
    for metrics_file in tqdm(metrics_files, desc="Processing metrics files"):
        # We pass exp_root_dir to parse_path_for_metrics but it's not strictly needed
        # by the new robust parsing, as it works on absolute paths.
        # However, keeping it makes the filtering based on relative position to root more explicit if needed later.
        parsed_info = parse_path_for_metrics(metrics_file, exp_root_dir) 
        
        if not parsed_info:
            continue # Skipped due to parsing error or filter miss

        try:
            with open(metrics_file, 'r') as f:
                content = json.load(f)
            
            standardized_metrics = standardize_metrics(content)
            fold_auc = standardized_metrics.get("auc_score")

            if fold_auc is not None:
                all_fold_aucs.append({
                    "model_config_combo_name": parsed_info['model_config_combo_name'],
                    "fold": parsed_info['fold_id'],
                    "auc": fold_auc,
                    "metrics_file_path": parsed_info['metrics_path']
                })
            else:
                logging.warning(f"AUC not found in '{metrics_file}' after standardization. Skipping.")

        except Exception as e:
            logging.info(f"Error reading or parsing '{metrics_file}': {e}. Skipping.")

    df_per_fold_auc = pd.DataFrame(all_fold_aucs)
    logging.info(f"Successfully collected {len(df_per_fold_auc)} per-fold AUC scores matching criteria.")
    return df_per_fold_auc


def run_wilcoxon_tests(df_per_fold_auc: pd.DataFrame, output_dir: Path):
    """
    Identifies the best model_config_combo_name based on average AUC from per-fold data
    and performs Wilcoxon signed-rank tests against all other model_config_combo_names.
    """
    output_file_name = f"wilcoxon_test_results_AUC_agnostic_V4.csv" 
    wilcoxon_results_file = output_dir / output_file_name

    if df_per_fold_auc.empty:
        logging.info("No per-fold AUC data available for Wilcoxon tests. Aborting.")
        return

    # Calculate average AUC for each unique model_config_combo_name
    avg_auc_per_model_config = df_per_fold_auc.groupby('model_config_combo_name')['auc'].mean().sort_values(ascending=False)
    
    if avg_auc_per_model_config.empty:
        logging.info("No unique model configurations found after grouping per-fold AUC data. Aborting.")
        return

    # Identify the best model configuration
    best_model_config_name = avg_auc_per_model_config.index[0]
    best_model_avg_auc = avg_auc_per_model_config.iloc[0]
    
    logging.info(f"\nBest model configuration (average AUC): '{best_model_config_name}' (Avg AUC: {best_model_avg_auc:.4f})")

    # Extract AUCs for the best model configuration for comparison
    best_config_auc_per_fold = df_per_fold_auc[df_per_fold_auc['model_config_combo_name'] == best_model_config_name].set_index('fold')['auc'].sort_index()

    test_results = []
    model_configs_to_compare = [mc for mc in avg_auc_per_model_config.index.unique() if mc != best_model_config_name]

    num_comparisons = len(model_configs_to_compare)
    corrected_alpha = SIGNIFICANCE_LEVEL / num_comparisons if num_comparisons > 0 else SIGNIFICANCE_LEVEL
    logging.info(f"\nPerforming {num_comparisons} comparisons against the best model configuration '{best_model_config_name}'.")
    logging.info(f"Bonferroni corrected alpha level: {corrected_alpha:.4f}")

    for current_model_config_name in model_configs_to_compare:
        current_config_auc_per_fold = df_per_fold_auc[df_per_fold_auc['model_config_combo_name'] == current_model_config_name].set_index('fold')['auc'].sort_index()

        common_folds = best_config_auc_per_fold.index.intersection(current_config_auc_per_fold.index)
        if len(common_folds) < 2:
            logging.warning(f"Not enough common folds (n={len(common_folds)}) for comparison between '{best_model_config_name}' and '{current_model_config_name}'. Skipping.")
            continue
        
        diffs = best_config_auc_per_fold.loc[common_folds] - current_config_auc_per_fold.loc[common_folds]
        
        if diffs.empty:
            logging.warning(f"No valid differences for comparison between '{best_model_config_name}' and '{current_model_config_name}'. Skipping.")
            continue
        
        if diffs.abs().sum() == 0:
             stat, p_value = 0.0, 1.0
        else:
            stat, p_value = wilcoxon(diffs, method="exact")

        is_significant = "Yes" if p_value < corrected_alpha else "No"
        
        avg_diff = diffs.mean()
        direction = "Superior" if avg_diff > 0 else ("Inferior" if avg_diff < 0 else "Similar")

        test_results.append({
            "Reference_Model_Config": best_model_config_name,
            "Compared_Model_Config": current_model_config_name,
            "Avg_AUC_Reference": best_model_avg_auc,
            "Avg_AUC_Compared": avg_auc_per_model_config.loc[current_model_config_name],
            "Wilcoxon_Statistic": stat,
            "P_Value": p_value,
            "Bonferroni_Corrected_Alpha": corrected_alpha,
            "Is_Significant_After_Correction": is_significant,
            "Direction_of_Difference_vs_Reference": direction
        })
        logging.info(f"  - Comparing '{best_model_config_name}' vs '{current_model_config_name}': P-value = {p_value:.4f} (Corrected alpha={corrected_alpha:.4f}), Significant={is_significant}, Direction={direction}")

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(wilcoxon_results_file, index=False)
    logging.info(f"\nWilcoxon test results saved to: {wilcoxon_results_file}")
    logging.info("Analysis complete.")


if __name__ == "__main__":
    # IMPORTANT: Ensure ROOT_DIR is correctly configured in researchpkg.anomaly_detection.config
    # EXP_ROOT_DIR should be the TOP-LEVEL directory that contains all your
    # first-level experiment folders (like 'llm_softmax_fraud_V4_fin', 'tree_models_V4_fin', etc.).
    # Assuming ROOT_DIR is the root of your 'researchpkg', and experiment data is directly under it.
    
    EXP_ROOT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    if not EXP_ROOT_DIR.exists():
        logging.info(f"Error: Experiment root directory not found at {EXP_ROOT_DIR}. Please check the path and ROOT_DIR configuration.")
        sys.exit(1)

    # Collect AUCs for each fold for all relevant experiments
    # This will now filter for experiments matching the TARGET_DATASET_VERSION_PATTERN (e.g., '_V4_')
    # and use the combined folder names as unique identifiers.
    df_per_fold_auc = collect_per_fold_auc_scores(EXP_ROOT_DIR)

    # Run the Wilcoxon tests using the collected per-fold data
    run_wilcoxon_tests(df_per_fold_auc, WILCOXON_OUTPUT_DIR)