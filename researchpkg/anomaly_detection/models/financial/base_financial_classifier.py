import logging
from abc import ABC
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from researchpkg.anomaly_detection.config import SEED_TRAINING
from researchpkg.anomaly_detection.models.utils import (
    IdentityScaler,
    NumericalFeaturesType,
    load_numpy_dataset,
)
from researchpkg.utils import configure_logger, numpy_to_scalar, torch_to_scalar


class BaseFinancialClassifier(ABC):
    """
    Base class for financial classifiers.
    This class should be inherited by all financial classifier models.
    """

    BASE_EXPERIMENT_DIR = None

    def __init__(self, features_type: NumericalFeaturesType, config: dict):
        """
        Initialize the base financial classifier.

        Args:
            model_name (str): Name of the model.
            features_type (NumericalFeaturesType): Type of numerical features used.
            config (dict): Configuration dictionary containing model parameters.
        """
        self.features_type = features_type
        self.config = config
        self.fold_id = config["fold_id"]
        self.dataset_version = config["dataset_version"]
        self.experiment_name = self.parse_experiment_name()
        if self.config["standardize"]:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
        else:
            # Identical scaling
            self.scaler = IdentityScaler()

        self.setup_experiment_dir()  # Setup logging and directories

    def parse_experiment_name(self) -> str:
        """
        Parse the experiment name from the model name.

        Returns:
            str: Parsed experiment name.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def setup_experiment_dir(self):
        """Sets up the experiment directory and logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            self.parse_experiment_name() + "_" + self.config["dataset_version"]
        )

        base_experiment_dir = self.config.get(
            "base_experiment_dir", self.BASE_EXPERIMENT_DIR
        )
        base_experiment_dir = Path(base_experiment_dir)
        self.log_dir = (
            base_experiment_dir
            / experiment_name
            / f"fold_{self.config['fold_id']}"
            / timestamp
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiment.log"
        self.tensorboard_log_dir = self.log_dir / "lightning_logs"  # PL default

        configure_logger(logFile=self.log_file, logLevel=logging.INFO)
        logging.info(f"Experiment logs will be saved to: {self.log_dir}")
        logging.info(f"TensorBoard logs directory: {self.tensorboard_log_dir}")

        # Save config (optional, can be done in fit/train_and_evaluate)
        with open(self.log_dir / "experiment_config.yaml", "w") as f:
            yaml.dump(self.config, f, indent=2)

    def load_data(self, train_path, test_path, full_financial_path=None):
        """Loads and scales data, fits scaler."""
        dataset_version = getattr(self, "dataset_version")
        fold_id = getattr(self, "fold_id")
        if dataset_version is None:
            raise NotImplementedError(
                "Subclasses must implement load_data or define dataset_version"
            )

        # Load raw data first
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
        ) = load_numpy_dataset(
            dataset_version=dataset_version,
            features=self.features_type,
            fold_id=fold_id,
            train_path=train_path,
            test_path=test_path,
        )
        self.feature_cols = feature_cols

        # Combine features for scaling
        logging.info(f"Fitting Scaler: {self.scaler.__class__.__name__}")
        X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)

        self.scaler.fit(X_all)  # Fit the scaler on all data

        # # Transform each subset
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrames (optional, can work with numpy)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=feature_cols, index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            X_val_scaled, columns=feature_cols, index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=feature_cols, index=X_test.index
        )

        logging.info("Data scaling complete.")

        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

    def oversample_fraud_cases(self, X_train, y_train):
        """
        Oversamples the minority class (fraud cases) in the training data.

        Args:
            X_train (np.ndarray or pd.DataFrame): Training features.
            y_train (np.ndarray or pd.Series): Training labels.

        Returns:
            tuple: (X_train_oversampled, y_train_oversampled) as numpy arrays.
        """
        logging.info("Applying custom oversampling to training data...")
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        fraud_indices = np.where(y_train_np == 1)[0]
        non_fraud_indices = np.where(y_train_np == 0)[0]

        num_fraud = len(fraud_indices)
        num_non_fraud = len(non_fraud_indices)

        if num_fraud == 0:
            logging.warning("No fraud cases found. Oversampling cannot be applied.")
            return X_train_np, y_train_np
        if num_fraud >= num_non_fraud:
            logging.warning("Fraud cases >= non-fraud cases. Oversampling not needed.")
            return X_train_np, y_train_np

        # Calculate how many fraud samples to add
        num_to_add = num_non_fraud - num_fraud
        rng = np.random.default_rng(SEED_TRAINING)
        oversample_indices = rng.choice(fraud_indices, size=num_to_add, replace=True)

        X_oversampled = np.vstack((X_train_np, X_train_np[oversample_indices]))
        y_oversampled = np.hstack((y_train_np, y_train_np[oversample_indices]))

        logging.info(f"Original training size: {len(X_train_np)}")
        logging.info(f"Oversampled training size: {len(X_oversampled)}")
        logging.info(f"Fraud count after oversampling: {np.sum(y_oversampled)}")

        # Shuffle the oversampled data
        shuffle_indices = rng.permutation(len(X_oversampled))
        X_oversampled = X_oversampled[shuffle_indices]
        y_oversampled = y_oversampled[shuffle_indices]

        return X_oversampled, y_oversampled

    def setup_model(self):
        """
        Sets up the model architecture and training parameters.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load_model(self, model_path):
        """
        Loads a pre-trained model from the specified path.

        Args:
            model_path (str): Path to the saved model file.

        Returns:
            The loaded model.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plot_precision_recall_curve(self, X_test_scaled, y_test, save_path=None):
        """Plot precision-recall curve. Assumes X_test_scaled is already scaled."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")

        y_pred_proba, _ = self.predict(X_test_scaled)  # Use predict method

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision_sklearn = average_precision_score(y_test, y_pred_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(
            recall,
            precision,
            marker=".",
            label=f"MLP (AP = {avg_precision_sklearn:.3f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Precision-recall curve saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def save_experiment_results(self, metrics_val, metrics_test):
        """Save experiment results to YAML file."""
        if self.log_dir is None:
            logging.error(
                "Log directory not initialized. Cannot save experiment results."
            )
            return None
        output_path = self.log_dir / "experiment.yaml"

        # Extract best epoch from checkpoint path if possible
        best_epoch = -1
        if self.best_model_path and "epoch=" in str(self.best_model_path):
            try:
                best_epoch = int(self.best_model_path.split("epoch=")[1].split("-")[0])
            except Exception:
                logging.warning("Could not parse best epoch from checkpoint path.")

        # Get hyperparameters from the loaded model if available
        hparams_dict = {}
        if self.model and hasattr(self.model, "hparams"):
            hparams_dict = {
                k: v
                for k, v in self.model.hparams.items()
                if k not in ["input_dim", "pos_weight"]
            }  # Exclude dynamic ones
            hparams_dict[
                "hidden_dims"
            ] = self.hidden_dims  # Ensure correct format if loaded differently
            hparams_dict[
                "batch_size"
            ] = self.batch_size  # Add batch size if not in hparams
            hparams_dict["epochs"] = self.epochs  # Add max epochs
            hparams_dict["patience"] = self.patience  # Add patience
            hparams_dict["device"] = self.device_type

        experiment_dict = {
            "model_name": self.model_name,
            "hyperparameters": numpy_to_scalar(hparams_dict),
            "training_results": {
                "best_epoch": best_epoch,
                "best_model_checkpoint": str(self.best_model_path),
                # Best val loss is implicitly tracked by ModelCheckpoint
                # Training history is in TensorBoard logs
            },
            "evaluation_metrics_validation": torch_to_scalar(
                metrics_val
            ),  # Metrics from last validation epoch during fit
            "evaluation_metrics_test": torch_to_scalar(
                metrics_test
            ),  # Metrics from trainer.test()
            "feature_columns": self.feature_cols
            if self.feature_cols is not None
            else None,
        }

        with open(output_path, "w") as f:
            yaml.dump(experiment_dict, f, indent=4, default_flow_style=False)

        logging.info(f"Experiment results saved to {output_path}")
        return output_path

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit the model to the training data.

        Args:
            X_train (np.ndarray or pd.DataFrame): Training features.
            y_train (np.ndarray or pd.Series): Training labels.
            X_val (np.ndarray or pd.DataFrame): Validation features.
            y_val (np.ndarray or pd.Series): Validation labels.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate(self, X_test, y_test, subset="test"):
        """
        Evaluate the model on the test data.

        Args:
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): Test labels.

        Returns:
            dict: Evaluation metrics.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self, X_test):
        raise NotImplementedError("Should implementt the preidct methode")

    def make_predictions_csv(self, X_test, y_test, save_path):
        """
        Save predictions to a CSV file.

        Args:
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): Test labels.
            save_path (str): Path to save the predictions CSV.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")

        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        y_pred_proba, y_pred = self.predict(X_test)

        results_df = pd.DataFrame(
            {"y_true": y_test, "y_pred_proba": y_pred_proba, "y_pred": y_pred}
        )
        results_df.to_csv(save_path, index=False)
        logging.info(f"Predictions saved to {save_path}")

    def find_best_threshold(self, y_val, y_pred_proba):
        """
        Find the optimal decision threshold based on validation data.

        Args:
            X_val (np.ndarray or pd.DataFrame): Validation features.
            y_val (np.ndarray or pd.Series): Validation labels.

        Returns:
            float: Optimal decision threshold.
        """
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

        # Find the threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        logging.info(f"Optimal threshold found: {optimal_threshold:.4f}")
        return optimal_threshold
