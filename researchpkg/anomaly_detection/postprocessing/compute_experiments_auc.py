import json
import sys
from pathlib import Path

import pandas as pd
from colorama import Fore, Style, init
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from researchpkg.anomaly_detection.config import ROOT_DIR

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)


def compute_auc_single_experiment(experiment_dir: Path):
    """ """

    # Find all the metrics jsons
    all_epochs_metrics_files = experiment_dir.rglob("metrics*.json")
    for json_file in tqdm(all_epochs_metrics_files, f"  Processing : {experiment_dir}"):
        print("File:", json_file)
        try:
            content = json.load(open(json_file, "r"))
            if "epoch" not in content:
                continue
            epoch_num = content["epoch"]
            predictions_file = experiment_dir / f"val_predictions_epoch_{epoch_num}.csv"

            df = pd.read_csv(predictions_file)
            if "true_labels" in df.columns:
                # true_label,probability,predicted_label,correct
                # rename columns
                df.rename(
                    columns={
                        "true_labels": "y_true_id",
                        "predicted_label": "y_pred_id",
                        "probability": "fraud_probability",
                    },
                    inplace=True,
                )

            ytrue = df["y_true_id"].tolist()
            ypred_prob = df["fraud_probability"].tolist()
            auc_score = roc_auc_score(ytrue, ypred_prob)

            content["auc_score"] = auc_score
            with open(json_file, "w") as f:
                json.dump(content, f, indent=4)
        except:
            print(f"Error with file: {json_file}")

    # now the test metics
    test_metrics_file = experiment_dir / "test_metrics.json"
    if test_metrics_file.exists():
        json_file = test_metrics_file
        content = json.load(open(json_file))
        content = json.load(open(test_metrics_file))
        predictions_file = experiment_dir / f"test_predictions.csv"

        df = pd.read_csv(predictions_file)

        if "true_labels" in df.columns:
            # true_label,probability,predicted_label,correct
            # rename columns
            df.rename(
                columns={
                    "true_labels": "y_true_id",
                    "predicted_label": "y_pred_id",
                    "probability": "fraud_probability",
                },
                inplace=True,
            )

        ytrue = df["y_true_id"].tolist()
        ypred_prob = df["fraud_probability"].tolist()
        auc_score = roc_auc_score(ytrue, ypred_prob)

        content["auc_score"] = auc_score
        with open(json_file, "w") as f:
            json.dump(content, f, indent=4)


def process_all_experiments_dir(root_dir: Path):
    """
    Recursively find all experiment directories under root_dir
    (those containing an 'experiment_config.yaml') and compute AUC for each.
    """
    # Ensure root_dir is a Path
    root_dir = Path(root_dir)

    # Find all subdirectories containing 'experiment_config.yaml'
    experiment_dirs = [p.parent for p in root_dir.rglob("experiment_config.yaml")]

    print(f"Found {len(experiment_dirs)} experiments.")

    for exp_dir in tqdm(
        experiment_dirs, "Computing auc scores of all experiments completed"
    ):
        try:
            compute_auc_single_experiment(exp_dir)
        except Exception as e:
            print(f"{Fore.RED}Error processing {exp_dir}: {e}{Style.RESET_ALL}")
            continue


if __name__ == "__main__":
    exp_root_dir = ROOT_DIR / "experiments_paper_emnlp"
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
