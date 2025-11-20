import re
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
from sklearn.metrics import roc_auc_score

# Assume ROOT_DIR is correctly defined and accessible from researchpkg.anomaly_detection.config
from researchpkg.anomaly_detection.config import ROOT_DIR
# --- Configuration ---
EXP_ROOT_DIR = ROOT_DIR/"experiments_aaai"
WILCOXON_OUTPUT_DIR = Path("./wilcoxon_analysis_auc_bootstrap")
LOG_FILE = WILCOXON_OUTPUT_DIR / "wilcoxon_analysis_auc_bootstrap.log"
SIGNIFICANCE_LEVEL = 0.05
TARGET_DATASET_VERSION_PATTERN = 'v4'
FOLD_FOLDER_REGEX = re.compile(r'fold_(\d+)')

# Configure logging
WILCOXON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"Starting Wilcoxon analysis based on bootstrapped AUC. Logs will be saved to: {LOG_FILE}")
logging.info(f"Analysis results will be saved in: {WILCOXON_OUTPUT_DIR}")
logging.info(f"Searching for experiments containing '{TARGET_DATASET_VERSION_PATTERN}' in their config paths.")

def parse_path_for_predictions(predictions_file_path: Path, exp_root_path: Path) -> dict | None:
    """
    Parses a test_predictions.csv file path to extract experiment configuration details
    agnostically, by first identifying the 'fold_ID' folder and then its parents.
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
        model_config_combo_name = model_config_combo_name.replace("llm_softmax_fraud_classifier_dataset_","")
        return {
            "model_config_combo_name": model_config_combo_name,
            "fold_id": fold_id,
            "predictions_path": str(predictions_file_path)
        }
    except Exception as e:
        logging.error(f"Error parsing path '{predictions_file_path}': {e}. Skipping.", exc_info=False)
        return None

def collect_per_fold_bootstrap_aucs(exp_root_dir: Path, n_bootstrap_samples=1000) -> pd.DataFrame:
    """
    Collects bootstrapped AUCs for each fold from 'test_predictions.csv' files.
    """
    logging.info(f"Collecting per-fold bootstrapped AUCs from '{exp_root_dir}'...")
    all_bootstrap_aucs = []
    predictions_files = list(exp_root_dir.rglob("test_predictions.csv"))
    logging.info(f"Found {len(predictions_files)} 'test_predictions.csv' files.")
    
    from collections import defaultdict
    fold_model_data = defaultdict(lambda: defaultdict(list))
    
    for predictions_file in predictions_files:
        parsed_info = parse_path_for_predictions(predictions_file, exp_root_dir)
        if not parsed_info:
            continue
        fold_id = parsed_info['fold_id']
        model_name = parsed_info['model_config_combo_name']

        try:
            df_predictions = pd.read_csv(predictions_file)
            if 'y_true_id' not in df_predictions.columns or 'fraud_probability' not in df_predictions.columns:
                logging.warning(f"Missing required columns in '{predictions_file}'. Skipping.")
                continue
                
            y_true = df_predictions['y_true_id'].values
            y_proba = df_predictions['fraud_probability'].values
            
            fold_model_data[fold_id][model_name] = {
                'y_true': y_true,
                'y_proba': y_proba,
                'predictions_file': str(predictions_file)
            }
        except Exception as e:
            logging.error(f"Error processing '{predictions_file}': {e}. Skipping.")
    
    # Process each fold
    for fold_id, model_data in fold_model_data.items():
        # Get y_true from any model in this fold (assuming all same)
        first_model = next(iter(model_data.values()))
        y_true = first_model['y_true']
        n_samples = len(y_true)
        
        for b in tqdm(range(n_bootstrap_samples), desc=f"Bootstrapping fold {fold_id}"):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            for model_name, model_info in model_data.items():
                y_proba = model_info['y_proba']
                y_proba_bootstrap = y_proba[indices]
                y_true_bootstrap = y_true[indices]
                
                try:
                    auc = roc_auc_score(y_true_bootstrap, y_proba_bootstrap)
                except ValueError as e:
                    logging.info(f"Could not compute AUC for model '{model_name}' in fold {fold_id}, bootstrap {b}: {e}. Using NaN.")
                    auc = np.nan
                
                all_bootstrap_aucs.append({
                    "model_config_combo_name": model_name,
                    "fold": fold_id,
                    "bootstrap_sample": b,
                    "auc": auc
                })

                
    
    df_bootstrap_aucs = pd.DataFrame(all_bootstrap_aucs)
    logging.info(f"Collected {len(df_bootstrap_aucs)} bootstrapped AUCs.")
    return df_bootstrap_aucs

def run_wilcoxon_tests_bootstrap_auc(df_bootstrap_aucs: pd.DataFrame, output_dir: Path):
    """
    Identifies the best model_config_combo_name based on average AUC (highest)
    and performs Wilcoxon signed-rank tests against all other model_config_combo_names
    using bootstrapped AUCs.
    """
    output_file_name = f"wilcoxon_test_results_AUC_bootstrap_agnostic_V4.csv"
    wilcoxon_results_file = output_dir / output_file_name
    if df_bootstrap_aucs.empty:
        logging.info("No bootstrapped AUC data available for Wilcoxon tests. Aborting.")
        return
    
    avg_auc_per_model_config = df_bootstrap_aucs.groupby('model_config_combo_name')['auc'].mean().sort_values(ascending=False)
    if avg_auc_per_model_config.empty:
        logging.info("No unique model configurations found after grouping bootstrapped AUC data. Aborting.")
        return
    
    best_model_config_name = avg_auc_per_model_config.index[0]
    best_model_avg_auc = avg_auc_per_model_config.iloc[0]
    
    test_results = []
    model_configs_to_compare = [mc for mc in avg_auc_per_model_config.index.unique() if mc != best_model_config_name]
    num_comparisons = len(model_configs_to_compare)
    corrected_alpha = SIGNIFICANCE_LEVEL / num_comparisons if num_comparisons > 0 else SIGNIFICANCE_LEVEL
    
    logging.info(f"\nPerforming {num_comparisons} comparisons against the best model configuration '{best_model_config_name}'.")
    logging.info(f"Bonferroni corrected alpha level: {corrected_alpha:.4f}")
    
    for current_model_config_name in model_configs_to_compare:
        best_aucs = []
        current_aucs = []
        
        for fold_id in df_bootstrap_aucs['fold'].unique():
            best_auc_df = df_bootstrap_aucs[(df_bootstrap_aucs['model_config_combo_name'] == best_model_config_name) &
                                             (df_bootstrap_aucs['fold'] == fold_id)]
            current_auc_df = df_bootstrap_aucs[(df_bootstrap_aucs['model_config_combo_name'] == current_model_config_name) &
                                              (df_bootstrap_aucs['fold'] == fold_id)]
            
            if best_auc_df.empty or current_auc_df.empty:
                logging.warning(f"No data for models '{best_model_config_name}' and '{current_model_config_name}' in fold {fold_id}. Skipping.")
                continue
                
            if len(best_auc_df) != len(current_auc_df):
                logging.warning(f"Mismatch in bootstrap samples between models '{best_model_config_name}' and '{current_model_config_name}' in fold {fold_id}. Skipping.")
                continue

            # Compute the average AUC for the current fold
            best_model_avg_auc = best_auc_df['auc'].mean()
            best_aucs.append(best_model_avg_auc)    
            # best_aucs.extend(best_auc_df['auc'].values)
            
            current_model_avg_auc = current_auc_df['auc'].mean()
            current_aucs.append(current_model_avg_auc)
            # current_aucs.extend(current_auc_df['auc'].values)
        
        if len(best_aucs) < 2:
            logging.warning(f"Not enough bootstrap samples for comparison between '{best_model_config_name}' and '{current_model_config_name}'. Skipping.")
            continue
        
        diffs = np.array(best_aucs) - np.array(current_aucs)
        if np.abs(diffs.sum()) == 0:
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = wilcoxon(diffs)
        
        is_significant = "Yes" if p_value <  SIGNIFICANCE_LEVEL else "No"
        avg_diff = np.mean(diffs)
        direction = "Superior" if avg_diff > 0 else ("Inferior" if avg_diff < 0 else "Similar")
        
        test_results.append({
            "Reference_Model_Config": best_model_config_name,
            "Compared_Model_Config": current_model_config_name,
            "Avg_AUC_Reference": best_model_avg_auc,
            "Avg_AUC_Compared": avg_auc_per_model_config.loc[current_model_config_name],
            "Wilcoxon_Statistic": stat,
            "P_Value": p_value,
            "Is_Significant": is_significant,
            "Avg_AUC_bootrap_Reference": np.mean(best_aucs),
            "Avg_AUC_bootrap_Compared": np.mean(current_aucs),
            "Direction_of_Difference_vs_Reference": direction
        })
        
        logging.info(f"  - Comparing '{best_model_config_name}' vs '{current_model_config_name}': P-value = {p_value:.4f}, Significant={is_significant}, Direction={direction}")
    
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(wilcoxon_results_file, index=False)
    logging.info(f"\nWilcoxon test results for bootstrapped AUCs saved to: {wilcoxon_results_file}")
    logging.info("Bootstrap AUC analysis complete.")

if __name__ == "__main__":
    EXP_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not EXP_ROOT_DIR.exists():
        logging.error(f"Error: Experiment root directory not found at {EXP_ROOT_DIR}. Please check the path and ROOT_DIR configuration.")
        sys.exit(1)
    
    # AUC bootstrap analysis
    df_bootstrap_aucs = collect_per_fold_bootstrap_aucs(EXP_ROOT_DIR)
    # Run the Wilcoxon tests using the collected bootstrapped AUCs
    run_wilcoxon_tests_bootstrap_auc(df_bootstrap_aucs, WILCOXON_OUTPUT_DIR)