import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import sys
from sklearn.metrics import roc_auc_score
import hashlib
# Import joblib for parallel processing
from joblib import Parallel, delayed

# Assume ROOT_DIR is correctly defined and accessible from researchpkg.anomaly_detection.config
# In a standalone script, you might define it like this:
# ROOT_DIR = Path("/path/to/your/project") 
from researchpkg.anomaly_detection.config import ROOT_DIR

# --- Configuration ---
EXP_ROOT_DIR = ROOT_DIR / "experiments_aaai"
OUTPUT_DIR = Path("./bootstrap_analysis_auc_empirical_pvalue")
LOG_FILE = OUTPUT_DIR / "bootstrap_analysis_auc_empirical_pvalue.log"
SIGNIFICANCE_LEVEL = 0.05
TARGET_DATASET_VERSION_PATTERN = 'v4'
FOLD_FOLDER_REGEX = re.compile(r'fold_(\d+)')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(f"Starting bootstrap AUC empirical p-value analysis. Logs will be saved to: {LOG_FILE}")


def stable_hash(s):
    """Generates a stable integer hash from a string."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def parse_path_for_predictions(predictions_file_path: Path) -> dict | None:
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
        if fold_idx == -1 or fold_idx < 2:
            logging.warning(f"Skipping path without proper fold folder: {predictions_file_path}")
            return None
        second_level_folder = abs_parts[fold_idx - 1]
        first_level_folder = abs_parts[fold_idx - 2]
        if TARGET_DATASET_VERSION_PATTERN not in first_level_folder and TARGET_DATASET_VERSION_PATTERN not in second_level_folder:
            logging.info(f"Skipping '{predictions_file_path}': does not contain '{TARGET_DATASET_VERSION_PATTERN}'")
            return None
        model_config_combo_name = f"{first_level_folder}/{second_level_folder}".replace("llm_softmax_fraud_classifier_dataset_", "")
        return {
            "model_config_combo_name": model_config_combo_name,
            "fold_id": fold_id,
            "predictions_path": str(predictions_file_path)
        }
    except Exception as e:
        logging.error(f"Error parsing path '{predictions_file_path}': {e}")
        return None


def collect_bootstrap_aucs(exp_root_dir: Path, n_bootstrap_samples=1000) -> pd.DataFrame:
    logging.info(f"Collecting bootstrap AUCs from '{exp_root_dir}' with {n_bootstrap_samples} samples per fold.")
    all_bootstrap_aucs = []
    
    predictions_files = list(exp_root_dir.rglob("test_predictions.csv"))
    logging.info(f"Found {len(predictions_files)} 'test_predictions.csv' files.")
    
    from collections import defaultdict
    fold_model_data = defaultdict(dict)
    
    for pred_file in predictions_files:
        info = parse_path_for_predictions(pred_file)
        if not info:
            continue
        fold_id = info['fold_id']
        model_name = info['model_config_combo_name']
        try:
            df = pd.read_csv(pred_file)
            if 'y_true_id' not in df.columns or 'fraud_probability' not in df.columns:
                logging.warning(f"Columns missing in {pred_file}. Skipping.")
                continue
    
            y_true = df['y_true_id'].values
            y_proba = df['fraud_probability'].values            
            fold_model_data[fold_id][model_name] = {'y_true': y_true, 'y_proba': y_proba}
        except Exception as e:
            logging.error(f"Error reading {pred_file}: {e}")

    # Use joblib to parallelize bootstrap sample generation across folds
    def process_fold(fold_id, models_data, n_bootstrap_samples):
        fold_aucs = []
        # Assume y_true is the same for all models in the fold
        y_true = next(iter(models_data.values()))['y_true']
        n_samples = len(y_true)
        
        for b in tqdm(range(n_bootstrap_samples)):
            # Generate one set of indices for all models in this bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            for model_name, data in models_data.items():
                try:
                    auc_val = roc_auc_score(y_true[idxs], data['y_proba'][idxs])
                except ValueError:
                    auc_val = np.nan
                
                fold_aucs.append({
                    "model_config_combo_name": model_name,
                    "model_hash": stable_hash(model_name),
                    "fold": fold_id,
                    "bootstrap_sample": b,
                    "auc": auc_val
                })
        return fold_aucs

    # The loop over folds can also be parallelized, though the main bottleneck is the comparison loop.
    # For simplicity and to focus on the user's request, we keep this part as is, but it's another optimization point.
    for fold_id, models_data in tqdm(fold_model_data.items(), desc="Bootstrapping folds"):
        # The inner loop over `n_bootstrap_samples` is the heavy part here
        # but parallelizing the outer loop over folds is simpler.
        all_bootstrap_aucs.extend(process_fold(fold_id, models_data, n_bootstrap_samples))

    df_bootstrap = pd.DataFrame(all_bootstrap_aucs)
    logging.info(f"Completed collecting bootstrapped AUCs: total {len(df_bootstrap)} rows.")
    return df_bootstrap

# --- New Helper Function for Parallel Execution ---
def _compare_model_pair(comp_model, best_model, df_indexed_aucs, folds, n_bootstrap, significance_level):
    """
    Performs bootstrap comparison for a single pair of models.
    This function is designed to be called in parallel by joblib.
    """
    best_model_hash = stable_hash(best_model)
    comp_model_hash = stable_hash(comp_model)
    diffs_all_bootstrap = []

    # Loop over bootstrap samples
    for b in range(n_bootstrap):
        diffs_per_fold = []
        for f in folds:
            try:
                # Use fast .loc lookup on the multi-index
                best_auc = df_indexed_aucs.loc[(best_model_hash, f, b), 'auc']
                comp_auc = df_indexed_aucs.loc[(comp_model_hash, f, b), 'auc']
                diffs_per_fold.append(best_auc - comp_auc)
            except KeyError:
                # This can happen if a model failed for a specific fold
                continue
        
        if diffs_per_fold:
            diffs_all_bootstrap.append(np.mean(diffs_per_fold))

    diffs_all_bootstrap = np.array(diffs_all_bootstrap)
    if diffs_all_bootstrap.size == 0:
        logging.warning(f"No valid bootstrap samples for comparison {best_model} vs {comp_model}. Skipping.")
        return None
    
    observed_mean_diff = diffs_all_bootstrap.mean()
    
    # Two-tailed bootstrap p-value estimation
    if observed_mean_diff > 0:
        p_val = np.mean(diffs_all_bootstrap <= 0) * 2
    else:
        p_val = np.mean(diffs_all_bootstrap >= 0) * 2
    
    p_val = min(p_val, 1.0)
    
    is_significant = "Yes" if p_val < significance_level else "No"
    direction = "Superior" if observed_mean_diff > 0 else ("Inferior" if observed_mean_diff < 0 else "Similar")
    
    return {
        "Reference_Model": best_model,
        "Compared_Model": comp_model,
        "Observed_Mean_Diff_AUC": observed_mean_diff,
        "Empirical_P_Value": p_val,
        "Is_Significant": is_significant,
        "Direction": direction
    }


# --- Optimized Main Analysis Function ---
def bootstrap_empirical_pvalue_analysis(df_bootstrap_aucs: pd.DataFrame, output_dir: Path):
    if df_bootstrap_aucs.empty:
        logging.error("No bootstrap AUC data found. Aborting.")
        return
    
    # Determine the best model based on mean AUC across all folds and bootstraps
    mean_auc = df_bootstrap_aucs.groupby('model_config_combo_name')['auc'].mean().sort_values(ascending=False)
    best_model = mean_auc.index[0]
    logging.info(f"Best model selected: {best_model} with mean AUC {mean_auc.iloc[0]:.4f}")

    other_models = [m for m in mean_auc.index if m != best_model]
    folds = df_bootstrap_aucs['fold'].unique()
    n_bootstrap = df_bootstrap_aucs['bootstrap_sample'].nunique()
    
    # --- PERFORMANCE OPTIMIZATION ---
    # Set a multi-index for extremely fast lookups. This is crucial for performance.
    logging.info("Optimizing DataFrame for fast lookups by setting multi-index...")
    df_indexed_aucs = df_bootstrap_aucs.set_index(['model_hash', 'fold', 'bootstrap_sample']).sort_index()

    # --- PARALLEL EXECUTION ---
    logging.info(f"Starting parallel comparison of {len(other_models)} models against the best model using all available CPU cores.")
    
    # Use joblib to run comparisons in parallel
    # verbose=10 provides progress updates every ~10% of tasks completed.
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_compare_model_pair)(
            comp_model,
            best_model,
            df_indexed_aucs,
            folds,
            n_bootstrap,
            SIGNIFICANCE_LEVEL
        )
        for comp_model in other_models
    )

    # Filter out any None results from failed comparisons
    results = [r for r in results if r is not None]
    
    if not results:
        logging.error("All model comparisons failed. No results to save.")
        return

    df_results = pd.DataFrame(results)
    output_file = output_dir / "bootstrap_empirical_pvalue_results.csv"
    df_results.to_csv(output_file, index=False)
    logging.info(f"Bootstrap empirical p-value results saved to: {output_file}")


if __name__ == "__main__":
    if not EXP_ROOT_DIR.exists():
        logging.error(f"Experiment root directory {EXP_ROOT_DIR} does not exist. Exiting.")
        sys.exit(1)

    # Note: With large datasets and many bootstrap samples, this can be memory intensive.
    # The `collect_bootstrap_aucs` can be further optimized if memory becomes an issue.
    df_bootstrap_aucs = collect_bootstrap_aucs(EXP_ROOT_DIR, n_bootstrap_samples=1000)
    bootstrap_empirical_pvalue_analysis(df_bootstrap_aucs, OUTPUT_DIR)