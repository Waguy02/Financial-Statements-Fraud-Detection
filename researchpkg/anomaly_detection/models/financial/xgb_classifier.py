import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from researchpkg.anomaly_detection.config import SEED_TRAINING, XGB_EXPERIMENTS_DIR
from researchpkg.anomaly_detection.models.financial.base_financial_classifier import (
    BaseFinancialClassifier,
)
from researchpkg.anomaly_detection.models.utils import (
    NumericalFeaturesType,
    load_cross_validation_path,
)
from researchpkg.utils import numpy_to_scalar

XGB_EXPERIMENTS_DIR = XGB_EXPERIMENTS_DIR  # Ensure the directory is set correctly


class XGB_Classifier(BaseFinancialClassifier):
    """
    XGBoost classifier for financial anomaly detection.
    Inherits from BaseFinancialClassifier.
    """

    BASE_EXPERIMENT_DIR = XGB_EXPERIMENTS_DIR

    def __init__(self, features_type, config):
        # XGBoost specific parameters
        self.max_depth = int(config["max_depth"])
        self.num_leaves = config["num_leaves"]
        self.num_estimators = config["num_estimators"]
        self.learning_rate = config["learning_rate"]

        self.standardize = config.get("standardize", True)
        self.model = None
        self.best_model_path = None
        super().__init__(features_type=features_type, config=config)
        # Common parameters for tracking
        self.device_type = "cpu"  # Default to CPU, can be changed to GPU
        self.model_name = config.get("model_name", self.parse_experiment_name())

    def parse_experiment_name(self):
        """Parses the experiment name from the configuration."""
        experiment_name = (
            f"xgb_md.{self.max_depth}_nl.{self.num_leaves}_ne.{self.num_estimators}_lr.{self.learning_rate}_"
            f"scale.{self.standardize}_"
            f"{self.features_type.name.lower()}"
        )
        return experiment_name

    def setup_model(self):
        """Sets up the XGBoost model architecture with given hyperparameters."""
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.num_estimators,
            "seed": SEED_TRAINING,
            "verbosity": 1,
            "use_label_encoder": False,
            "tree_method": "hist",  # Efficient histogram-based method
            "max_leaves": self.num_leaves,  # For hist tree method
        }

        self.model = xgb.XGBClassifier(**params)
        logging.info(f"XGBoost model initialized with params: {params}")
        return self.model

    def fit(self, X_train, y_train, X_val, y_val):
        """Fit the XGBoost model to the training data."""
        if self.model is None:
            self.setup_model()

        logging.info("Starting XGBoost model training...")

        # Calculate scale_pos_weight
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        self.model.scale_pos_weight = scale_pos_weight
        logging.info(f"Using scale_pos_weight: {scale_pos_weight}")

        # Setup early stopping using validation data
        eval_set = [(X_train, y_train), (X_val, y_val)]

        # Train with early stopping
        self.model.fit(
            X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=True
        )

        # Get best iteration from early stopping
        best_iteration = (
            self.model.best_iteration
            if hasattr(self.model, "best_iteration")
            else self.model.n_estimators
        )

        # Save the model
        model_path = self.log_dir / "best_model.joblib"
        joblib.dump(self.model, model_path)
        self.best_model_path = model_path

        logging.info(f"Training completed. Best iteration: {best_iteration}")

        # Get validation metrics for return
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        # Find the best threshold based on validation set
        best_threshold = self.find_best_threshold(y_val, y_val_pred_proba)
        logging.info(f"Best threshold found: {best_threshold}")
        self.decision_threshold = best_threshold

        y_val_pred = (y_val_pred_proba >= self.decision_threshold).astype(int)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_f1 = f1_score(y_val, y_val_pred)

        metrics = {
            "val_auc": val_auc,
            "val_f1": val_f1,
            "best_iteration": best_iteration,
        }

        logging.info(f"Validation metrics: AUC={val_auc:.4f}, F1={val_f1:.4f}")
        return metrics

    def evaluate(self, X_test, y_test, subset="Test", save_metrics=False):
        """Evaluate the XGBoost model on test data."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        logging.info(f"Evaluating model on {subset} set...")

        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.decision_threshold).astype(int)

        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks([0, 1], ["Normal", "Fraud"])
        plt.yticks([0, 1], ["Normal", "Fraud"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2.0 else "black",
                )

        plt.tight_layout()
        cm_path = self.log_dir / f"{subset}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Collect all metrics
        metrics = {
            "auc_score": auc_score,
            "f1_score": f1,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "decision_threshold": self.decision_threshold,
        }
        metrics = numpy_to_scalar(metrics)

        # logging.info(f"{subset} metrics: AUC={auc_score:.4f}, F1={f1:.4f}")
        if save_metrics:
            # Save detailed metrics to a file
            metrics_path = self.log_dir / f"{subset}_metrics.json"
            import json

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
        if subset.lower() == "test":
            test_predictions_path = self.log_dir / "test_predictions.csv"
            import pandas as pd

            predictions_df = pd.DataFrame(
                {
                    "y_true_id": y_test,
                    "fraud_probability": y_pred_proba,
                    "y_pred_id": (y_pred),
                }
            )
            predictions_df.to_csv(test_predictions_path, index=False)
            logging.info(f"Test predictions saved to {test_predictions_path}")
        return metrics

    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.decision_threshold).astype(int)

        return y_pred_proba, y_pred

    def load_model(self, model_path):
        """Load a pretrained XGBoost model."""
        logging.info(f"Loading XGBoost model from {model_path}")
        self.model = joblib.load(model_path)
        self.best_model_path = model_path
        return self

    def save_model(self, filepath=None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")

        if filepath is None:
            filepath = self.log_dir / "best_model.joblib"

        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")

        scaler_path = self.log_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")

        self.best_model_path = filepath
        return str(filepath), str(scaler_path)


if __name__ == "__main__":

    CONFIG = {
        "fold_id": 2,
        "dataset_version": "company_isolated_splitting",
        "max_depth": 0,
        "num_estimators": 50,
        "num_leaves": 50,
        "decision_threshold": 0.5,
        "learning_rate": 0.05,
        "features_type": NumericalFeaturesType.EXTENDED_DECHOW,
        "standardize": True,
    }

    model = XGB_Classifier(features_type=CONFIG["features_type"], config=CONFIG)
    model.setup_model()

    train_path, test_path = load_cross_validation_path(CONFIG)
    data = model.load_data(train_path=train_path, test_path=test_path)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = data

    print("Training MLP model with PyTorch Lightning...")
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    print("Evaluating MLP model on test set...")
    metrics_test = model.evaluate(X_test_scaled, y_test, subset="Test")

    print(f"Test Metrics: {metrics_test}")
    model.save_experiment_results(metrics_val={}, metrics_test=metrics_test)
