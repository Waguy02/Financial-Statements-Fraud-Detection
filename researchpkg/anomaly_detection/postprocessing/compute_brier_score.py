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
from sklearn.metrics import brier_score_loss # Import Brier Score

# Assume ROOT_DIR is correctly defined and accessible from researchpkg.anomaly_detection.config
from researchpkg.anomaly_detection.config import ROOT_DIR

# --- Configuration ---
EXP_ROOT_DIR = ROOT_DIR/"experiments_aaai" # Adjust this if your first-level experiment folders are under a subdirectory of ROOT_DIR
                        # E.g., if structure is ROOT_DIR/experiments_aaai/llm_.../ Then EXP_ROOT_DIR = ROOT_DIR / "experiments_aaai"

WILCOXON_OUTPUT_DIR = Path("./wilcoxon_analysis_brier_score") # New output directory for Brier Score analysis
LOG_FILE = WILCOXON_OUTPUT_DIR / "wilcoxon_analysis_brier_score.log"
SIGNIFICANCE_LEVEL = 0.05

# Pattern to identify v4 experiments in folder names
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

logging.info(f"Starting Wilcoxon analysis based on Brier Score. Logs will be saved to: {LOG_FILE}")
logging.info(f"Analysis results will be saved in: {WILCOXON_OUTPUT_DIR}")
logging.info(f"Searching for experiments containing '{TARGET_DATASET_VERSION_PATTERN}' in their config paths.")

# Regex to find a fold folder (e.g., 'fold_1', 'fold_2', etc.)
FOLD_FOLDER_REGEX = re.compile(r'fold_(\d+)')


def parse_path_for_predictions(predictions_file_path: Path, exp_root_path: Path) -> dict | None:
    """
    Parses a test_predictions.csv file path to extract experiment configuration details
    agnostically, by first identifying the 'fold_ID' folder and then its parents.
    
    Expected structure (flexibly):
    ... / FIRST_LEVEL_FOLDER / SECOND_LEVEL_FOLDER / fold_ID / TIMESTAMP / test_predictions.csv
    """
    try:
        abs_parts = predictions_file_path.absolute().parts
        
        fold_idx = -1
        fold_id = -1
        for i, part in enumerate(abs_parts):
            match = FOLD_FOLDER_REGEX.match(part)
            if match:
                fold_idx = i
                fold_id = int(match.group(1))
                break
        
        if fold_idx == -1:
            logging.warning(f"Could not find a 'fold_ID' folder in path: {predictions_file_path}. Skipping.")
            return None

        if fold_idx < 2:
            logging.warning(f"Path '{predictions_file_path}' has insufficient preceding folders before 'fold_{fold_id}' for config. Skipping.")
            return None

        second_level_folder = abs_parts[fold_idx - 1]
        first_level_folder = abs_parts[fold_idx - 2]

        if TARGET_DATASET_VERSION_PATTERN not in first_level_folder and \
           TARGET_DATASET_VERSION_PATTERN not in second_level_folder:
            logging.info(f"Skipping '{predictions_file_path}': Does not contain '{TARGET_DATASET_VERSION_PATTERN}'.")
            return None

        model_config_combo_name = f"{first_level_folder}/{second_level_folder}"
        model_config_combo_name=model_config_combo_name.replace("llm_softmax_fraud_classifier_dataset_","")
        return {
            "model_config_combo_name": model_config_combo_name,
            "fold_id": fold_id,
            "predictions_path": str(predictions_file_path)
        }
    except Exception as e:
        logging.error(f"Error parsing path '{predictions_file_path}': {e}. Skipping.", exc_info=False)
        return None


def collect_per_fold_brier_scores(exp_root_dir: Path) -> pd.DataFrame:
    """
    Collects Brier Scores for each fold from 'test_predictions.csv' files,
    filtering by target dataset version pattern.
    """
    logging.info(f"Collecting per-fold Brier Scores from '{exp_root_dir}' with dataset version pattern '{TARGET_DATASET_VERSION_PATTERN}'...")
    all_fold_brier_scores = []

    # Find all test_predictions.csv files recursively
    predictions_files = list(exp_root_dir.rglob("test_predictions.csv"))
    logging.info(f"Found {len(predictions_files)} 'test_predictions.csv' files in '{exp_root_dir}'.")

    for predictions_file in tqdm(predictions_files, desc="Processing prediction files"):
        parsed_info = parse_path_for_predictions(predictions_file, exp_root_dir)
        
        if not parsed_info:
            continue
        
        try:
            df_predictions = pd.read_csv(predictions_file)
            
            # Ensure required columns exist
            if 'y_true_id' not in df_predictions.columns or 'fraud_probability' not in df_predictions.columns:
                print(f"Required columns 'y_true_id' or 'fraud_probability' missing in '{predictions_file}'. Skipping.")
                continue

            # Calculate Brier Score for the current fold
            brier_score = brier_score_loss(df_predictions['y_true_id'], df_predictions['fraud_probability'])

            all_fold_brier_scores.append({
                "model_config_combo_name": parsed_info['model_config_combo_name'],
                "fold": parsed_info['fold_id'],
                "brier_score": brier_score,
                "predictions_file_path": parsed_info['predictions_path']
            })

        except Exception as e:
            logging.error(f"Error reading or processing '{predictions_file}': {e}. Skipping.")

    df_per_fold_brier = pd.DataFrame(all_fold_brier_scores)
    logging.info(f"Successfully collected {len(df_per_fold_brier)} per-fold Brier Scores matching criteria.")
    return df_per_fold_brier


def run_wilcoxon_tests_brier_score(df_per_fold_brier: pd.DataFrame, output_dir: Path):
    """
    Identifies the best model_config_combo_name based on average Brier Score (lowest)
    and performs Wilcoxon signed-rank tests against all other model_config_combo_names.
    """
    output_file_name = f"wilcoxon_test_results_BrierScore_agnostic_V4.csv" 
    wilcoxon_results_file = output_dir / output_file_name

    if df_per_fold_brier.empty:
        logging.info("No per-fold Brier Score data available for Wilcoxon tests. Aborting.")
        return

    # Calculate average Brier Score for each unique model_config_combo_name
    # # Sort ascending for Brier Score, as lower is better
    # avg_brier_per_model_config = df_per_fold_brier.groupby('model_config_combo_name')['brier_score'].mean()
    

    # best_model_config_combo_name = "v4_mda_summarized_undersample_full_summary/Fino1-8B_Last_32_Layers_lora_8_8_bs_8_dp0.05_fixed_head_lr_1e-4"

    # if avg_brier_per_model_config.empty:
    #     logging.info("No unique model configurations found after grouping per-fold Brier Score data. Aborting.")
    #     return

    # # Identify the best model configuration (lowest Brier Score)
    # best_model_config_name = best_model_config_combo_name
    # best_model_avg_brier = avg_brier_per_model_config.loc[best_model_config_name] 
    

    # Calculate average Brier Score for each unique model_config_combo_name
    # Sort ascending for Brier Score, as lower is better
    avg_brier_per_model_config = df_per_fold_brier.groupby('model_config_combo_name')['brier_score'].mean().sort_values(ascending=True)

    if avg_brier_per_model_config.empty:
        logging.info("No unique model configurations found after grouping per-fold Brier Score data. Aborting.")
        return

    # Identify the best model configuration (lowest Brier Score)
    best_model_config_name = avg_brier_per_model_config.index[0]
    best_model_avg_brier = avg_brier_per_model_config.iloc[0]

    

    # Extract Brier Scores for the best model configuration for comparison
    best_config_brier_per_fold = df_per_fold_brier[df_per_fold_brier['model_config_combo_name'] == best_model_config_name].set_index('fold')['brier_score'].sort_index()

    test_results = []
    model_configs_to_compare = [mc for mc in avg_brier_per_model_config.index.unique() if mc != best_model_config_name]

    num_comparisons = len(model_configs_to_compare)
    corrected_alpha = SIGNIFICANCE_LEVEL / num_comparisons if num_comparisons > 0 else SIGNIFICANCE_LEVEL
    logging.info(f"\nPerforming {num_comparisons} comparisons against the best model configuration '{best_model_config_name}'.")
    logging.info(f"Bonferroni corrected alpha level: {corrected_alpha:.4f}")

    for current_model_config_name in model_configs_to_compare:
        current_config_brier_per_fold = df_per_fold_brier[df_per_fold_brier['model_config_combo_name'] == current_model_config_name].set_index('fold')['brier_score'].sort_index()

        common_folds = best_config_brier_per_fold.index.intersection(current_config_brier_per_fold.index)
        if len(common_folds) < 2:
            logging.warning(f"Not enough common folds (n={len(common_folds)}) for comparison between '{best_model_config_name}' and '{current_model_config_name}'. Skipping.")
            continue
        
        # Calculate differences: Brier_best - Brier_current.
        # If best is truly better (lower Brier), this difference will be positive.
        diffs = best_config_brier_per_fold.loc[common_folds] - current_config_brier_per_fold.loc[common_folds]
        
        if diffs.empty:
            logging.warning(f"No valid differences for comparison between '{best_model_config_name}' and '{current_model_config_name}'. Skipping.")
            continue
        
        if diffs.abs().sum() == 0:
             stat, p_value = 0.0, 1.0
        else:
            # Note: For Brier Score (where lower is better), if you want positive difference for "better",
            # the better model's score should be subtracted from the other.
            # Here: Brier_best (lower) - Brier_current (higher) -> negative difference.
            # To make it consistent with "Superior" meaning best model is better:
            # We want to test if differences (Current_Brier - Best_Brier) are significantly > 0.
            # This is equivalent to testing if (Best_Brier - Current_Brier) are significantly < 0.
            # The Wilcoxon test in scipy.stats.wilcoxon tests if the median difference is non-zero.
            # Direction will indicate if best_model_brier - current_model_brier is > 0 (best is worse) or < 0 (best is better)
            # A negative average diff here means the best model has a lower Brier score.
            stat, p_value = wilcoxon(diffs)

        is_significant = "Yes" if p_value < corrected_alpha else "No"
        
        avg_diff = diffs.mean()
        # Interpretation for Brier Score: Negative avg_diff means best_brier < current_brier, so best is better.
        direction = "Superior" if avg_diff < 0 else ("Inferior" if avg_diff > 0 else "Similar")

        test_results.append({
            "Reference_Model_Config": best_model_config_name,
            "Compared_Model_Config": current_model_config_name,
            "Avg_Brier_Reference": best_model_avg_brier,
            "Avg_Brier_Compared": avg_brier_per_model_config.loc[current_model_config_name],
            "Wilcoxon_Statistic": stat,
            "P_Value": p_value,
            "Bonferroni_Corrected_Alpha": corrected_alpha,
            "Is_Significant_After_Correction": is_significant,
            "Direction_of_Difference_vs_Reference": direction # 'Superior' if Reference is better (lower Brier)
        })
        logging.info(f"  - Comparing '{best_model_config_name}' vs '{current_model_config_name}': P-value = {p_value:.4f} (Corrected alpha={corrected_alpha:.4f}), Significant={is_significant}, Direction={direction}")

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(wilcoxon_results_file, index=False)
    logging.info(f"\nWilcoxon test results saved to: {wilcoxon_results_file}")
    logging.info("Analysis complete.")


if __name__ == "__main__":
    # IMPORTANT: EXP_ROOT_DIR should be the TOP-LEVEL directory that contains all your
    # first-level experiment folders (like 'llm_softmax_fraud_V4_fin', 'tree_models_V4_fin', etc.).
    # Assuming ROOT_DIR is the root of your 'researchpkg', and experiment data is directly under it.
    
    EXP_ROOT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    
    if not EXP_ROOT_DIR.exists():
        logging.error(f"Error: Experiment root directory not found at {EXP_ROOT_DIR}. Please check the path and ROOT_DIR configuration.")
        sys.exit(1)

    # Collect Brier Scores for each fold for all relevant experiments
    # This will now filter for experiments matching the TARGET_DATASET_VERSION_PATTERN (e.g., '_V4_')
    # and use the combined folder names as unique identifiers.
    df_per_fold_brier = collect_per_fold_brier_scores(EXP_ROOT_DIR)

    # Run the Wilcoxon tests using the collected per-fold Brier Scores
    run_wilcoxon_tests_brier_score(df_per_fold_brier, WILCOXON_OUTPUT_DIR)