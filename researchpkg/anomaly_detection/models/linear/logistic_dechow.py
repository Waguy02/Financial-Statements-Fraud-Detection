import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic behavior
from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_DECHOW,
    FINANCIALS_DIR_EXTENDED,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.utils import get_train_test_splitter
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_dechow import (
    DECHOW_FEATURES,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
)
from researchpkg.utils import configure_logger

DECHOW_FIN_PATH = FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv"
FINANCIALS_FIN_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"


class DechowDataset(Dataset):
    """PyTorch Dataset for Dechow financial data"""

    def __init__(self, features, labels):
        """
        Initialize the dataset.

        Args:
            features (np.ndarray): Financial features
            labels (np.ndarray): Binary fraud labels (0=Not Fraud, 1=Fraud)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PermutableUndersamplingDechowDataset(Dataset):
    """
    A dataset wrapper that dynamically undersamples the majority class
    and can be permuted at the start of each epoch.
    """

    def __init__(
        self,
        features,
        labels,
        negative_positive_ratio=1.0,
        seed=SEED_TRAINING,
    ):
        """
        Initialize the permutable undersampling dataset.

        Args:
            features (torch.Tensor): Feature tensors
            labels (torch.Tensor): Label tensors
            negative_positive_ratio (float): Ratio of negative (non-fraud) to positive (fraud) samples
            seed (int): Random seed for reproducibility
        """
        # Store original data
        self.orig_features = features
        self.orig_labels = labels
        self.negative_positive_ratio = negative_positive_ratio

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get fraud and non-fraud indices
        self.labels_np = (
            labels.squeeze().numpy()
            if hasattr(labels, "numpy")
            else np.array(labels).squeeze()
        )
        self.positive_indices = np.where(self.labels_np == 1)[0]
        self.negative_indices = np.where(self.labels_np == 0)[0]

        self.num_positive = len(self.positive_indices)

        if self.num_positive == 0:
            logging.warning(
                "No positive samples found for undersampling. Using all data."
            )
            self.features = self.orig_features
            self.labels = self.orig_labels
            return
        self.permute()

    def permute(self):
        """
        Permute the dataset by undersampling non-fraud cases and shuffling.
        Called at the beginning of each epoch for dynamic undersampling.
        """
        # Calculate number of negative samples to keep based on ratio
        num_negative_to_keep = int(self.num_positive * self.negative_positive_ratio)
        num_negative_to_keep = min(num_negative_to_keep, len(self.negative_indices))

        # Randomly select negative samples to keep
        selected_negative_indices = np.random.choice(
            self.negative_indices, size=num_negative_to_keep, replace=False
        )

        # Always keep all positive samples
        selected_indices = np.concatenate(
            [self.positive_indices, selected_negative_indices]
        )
        np.random.shuffle(selected_indices)  # Shuffle the combined indices

        # Get selected data
        self.features = self.orig_features[selected_indices]
        self.labels = self.orig_labels[selected_indices]

        logging.info(
            f"Undersampled dataset: {len(self.positive_indices)} fraud samples and {len(selected_negative_indices)} non-fraud samples"
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPDechowModel(nn.Module):
    """
    MLP model for Dechow financial fraud detection.
    Can be configured with multiple hidden layers or a simple logistic regression.
    """

    def __init__(self, input_dim, hidden_layers=None):
        """
        Initialize the MLP model.

        Args:
            input_dim (int): Number of input features
            hidden_layers (list, optional): List of hidden layer dimensions. If None or empty,
                                           a simple logistic regression is used.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # If no hidden layers specified, create a simple logistic regression
        if not hidden_layers:
            # Simple logistic regression layer with intercept
            self.layers.append(nn.Linear(input_dim, 1, bias=True))
        else:
            # Build MLP with specified hidden layers
            prev_dim = input_dim
            for h_dim in hidden_layers:
                self.layers.append(nn.Linear(prev_dim, h_dim))
                prev_dim = h_dim

            # Output layer
            self.layers.append(nn.Linear(prev_dim, 1))

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input features

        Returns:
            torch.Tensor: Model predictions
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))

        # Last layer without activation (for binary classification with BCEWithLogitsLoss)
        x = self.layers[-1](x)
        return x


class LogisticFraudClassifierDechow(pl.LightningModule):
    """
    Logistic regression model for fraud detection based on Dechow's methodology.
    This model uses logistic regression to classify financial data as fraudulent or not.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # Get configs
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.hidden_layers = config.get("hidden_layers", [])
        self.weight_decay = config.get("weight_decay", 0.0)
        self.undersample = config.get("undersample", False)
        self.oversample = config.get("oversample", False)

        # Set up experiment directory and logging
        self.setup_logging()

        # Initialize scaler for feature normalization

        if self.config.get("standardize", False):
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.include_extended_features = self.config.get(
            "include_extended_features", False
        )

        # Build model - wait until we know the feature dimensions
        self.model = None
        self.input_dim = None

        # For threshold optimization
        self.best_threshold = 0.5
        self.val_preds = []
        self.val_labels = []

    def setup_logging(self):
        """Set up logging directories and files"""
        # Set up experiment directory
        fold_id = self.config.get("fold_id", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct model type name
        if self.config.get("hidden_layers", []):
            dims_str = "_".join(map(str, self.config["hidden_layers"]))
            model_name = f"mlp_dechow_{dims_str}"
        else:
            model_name = "logistic_dechow"

        if self.config.get("include_extended_features", False):
            model_name += "_with_extended"

        if self.undersample:
            model_name += "_undersample"

        if self.oversample:
            model_name += "_oversample"

        if self.config.get("standardize", False):
            model_name += "_standardized"

        self.log_dir = (
            Path(
                EXPERIMENTS_DIR
                / f"dechow_fraud_classifier_{self.config.get('dataset_version', 'v4')}"
            )
            / f"{model_name}"
            / f"fold_{fold_id}"
            / f"{timestamp}"
        )

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"experiment.log"

        configure_logger(
            logFile=self.log_file,  # Ensure Path is converted to string
            logLevel=logging.INFO,
        )
        logging.info(f"Logging to {self.log_file}")

    def forward(self, x):
        return self.model(x)

    def _process_loaded_data(self, df):
        """Process and merge financial data with the dataset."""
        # Load financial data if not already loaded
        if not hasattr(self, "_full_dechow_features_df"):
            logging.info(f"Loading full dechow data from {DECHOW_FIN_PATH}")
            self._full_dechow_features_df = pd.read_csv(
                DECHOW_FIN_PATH, usecols=["cik", "year", "quarter"] + DECHOW_FEATURES
            ).drop_duplicates(subset=["cik", "year", "quarter"])

        if self.include_extended_features:
            if not hasattr(self, "_full_financials_df"):
                logging.info(f"Loading full financials data from {FINANCIALS_FIN_PATH}")
                self._full_financials_df = pd.read_csv(
                    FINANCIALS_FIN_PATH,
                    usecols=["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES,
                ).drop_duplicates(subset=["cik", "year", "quarter"])

        # Drop existing feature count columns if present
        df = df.drop(columns=EXTENDED_FINANCIAL_FEATURES_COUNT_COLS, errors="ignore")

        # Merge with financials data
        df = df.merge(
            self._full_dechow_features_df, on=["cik", "year", "quarter"], how="left"
        )

        if self.include_extended_features:
            # Merge with extended financials data if configured
            df = df.merge(
                self._full_financials_df,
                on=["cik", "year", "quarter"],
                how="left",
                suffixes=("", "_extended"),
            )

        return df

    def load_data(self, train_path=None, test_path=None):
        """
        Load train and test datasets.

        Args:
            train_path (Path or str): Path to training data CSV.
            test_path (Path or str): Path to test data CSV.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        if train_path is None or test_path is None:
            raise ValueError("Both train_path and test_path must be provided.")

        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)
        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # --- Debugging: Subset Data ---
        if self.config.get("debug", False):
            debug_fraction = self.config.get(
                "debug_fraction", 0.01
            )  # Allow configurable fraction
            logging.warning(
                f"--- DEBUG MODE: Using {debug_fraction*100:.1f}% of the data ---"
            )
            # Ensure some fraud cases are included if possible
            min_samples = 50  # Ensure at least a few samples
            train_df = train_df.groupby("is_fraud", group_keys=False).apply(
                lambda x: x.sample(
                    max(min_samples // 2, int(len(x) * debug_fraction)),
                    random_state=SEED_TRAINING,
                )
            )
            test_df = test_df.groupby("is_fraud", group_keys=False).apply(
                lambda x: x.sample(
                    max(min_samples // 2, int(len(x) * debug_fraction)),
                    random_state=SEED_TRAINING,
                )
            )
            # train_df = train_df.sample(frac=debug_fraction, random_state=SEED_TRAINING)
            # test_df = test_df.sample(frac=debug_fraction, random_state=SEED_TRAINING)

        # --- Preprocessing ---
        # Apply consistent preprocessing steps defined in the subclass
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        # --- Train/Validation Split ---
        if self.config.get("no_validation", False):
            logging.warning(
                "`no_validation` is True. Using full training data for training and a small dummy validation set."
            )
            # Create a tiny validation set just to satisfy the Trainer requirements
            val_df = train_df.sample(
                n=min(10, len(train_df)), random_state=SEED_TRAINING
            )

        else:
            val_split_size = self.config.get("validation_split_size", 0.1)
            splitter = get_train_test_splitter(self.config)
            train_df, val_df = splitter(
                train_df, test_size=val_split_size, seed=SEED_TRAINING
            )

        # --- Oversampling (Applied *after* split to avoid data leakage) ---
        if self.config.get("oversample", False):
            logging.info("Oversampling fraud cases in the training data...")
            train_df = self.oversample_fraud_cases(train_df)

        # --- Logging Final Sizes ---
        logging.info(
            f"Train data size: {len(train_df)} (Fraud: {train_df['is_fraud'].sum()})"
        )
        logging.info(
            f"Validation data size: {len(val_df)} (Fraud: {val_df['is_fraud'].sum()})"
        )
        logging.info(
            f"Test data size: {len(test_df)} (Fraud: {test_df['is_fraud'].sum()})"
        )

        return train_df, val_df, test_df

    def prepare_data(self):
        """
        PyTorch Lightning method to prepare data.
        This runs once at the beginning of training.
        """
        pass  # Data loading is done explicitly before training

    def setup(self, stage=None):
        """
        PyTorch Lightning method to set up data for each stage.
        This prepares data for training, validation, or testing.

        Args:
            stage (str, optional): 'fit' for training, 'validate' for validation, 'test' for testing.
        """
        if (
            not hasattr(self, "train_df")
            or not hasattr(self, "val_df")
            or not hasattr(self, "test_df")
        ):
            logging.warning("Data not loaded yet. Call load_data() before setup().")
            return

        features = DECHOW_FEATURES.copy()
        if self.include_extended_features:
            features += EXTENDED_FINANCIAL_FEATURES

        # Extract features and target
        X_train = self.train_df[features].fillna(0).values
        y_train = self.train_df["is_fraud"].values

        X_val = self.val_df[features].fillna(0).values
        y_val = self.val_df["is_fraud"].values

        X_test = self.test_df[features].fillna(0).values
        y_test = self.test_df["is_fraud"].values

        # Initialize scaler and transform data
        if self.scaler is not None:
            if stage == "fit" or stage is None:
                self.scaler.fit(np.vstack((X_train, X_val, X_test)))

            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test

        # Store CIK and other metadata for evaluation
        self.train_meta = self.train_df[["cik", "year", "quarter"]].copy()
        self.val_meta = self.val_df[["cik", "year", "quarter"]].copy()
        self.test_meta = self.test_df[["cik", "year", "quarter"]].copy()

        # Create PyTorch datasets
        self.train_dataset = DechowDataset(X_train_scaled, y_train)
        self.val_dataset = DechowDataset(X_val_scaled, y_val)
        self.test_dataset = DechowDataset(X_test_scaled, y_test)

        # Initialize the model
        if self.model is None:
            self.input_dim = X_train_scaled.shape[1]
            logging.info(f"Initializing model with input dimension: {self.input_dim}")
            self.model = MLPDechowModel(
                input_dim=self.input_dim,
                hidden_layers=self.config.get("hidden_layers", []),
            )
            if torch.cuda.is_available():
                logging.info("Using GPU for training")
                self.model = self.model.cuda()

            # Log model architecture
            hidden_layers = self.config.get("hidden_layers", [])
            if hidden_layers:
                logging.info(
                    f"MLP architecture: {self.input_dim} -> {' -> '.join(map(str, hidden_layers))} -> 1"
                )
            else:
                logging.info(f"Logistic regression: {self.input_dim} -> 1")

    def oversample_fraud_cases(self, df):
        """
        Oversample minority class (fraud) to balance the dataset.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Oversampled dataframe
        """
        fraud_df = df[df["is_fraud"] == 1]
        non_fraud_df = df[df["is_fraud"] == 0]

        num_fraud = len(fraud_df)
        num_non_fraud = len(non_fraud_df)

        logging.info(
            f"Original class distribution: Fraud={num_fraud}, Non-fraud={num_non_fraud}"
        )

        if num_fraud == 0:
            logging.warning("No fraud cases found. Cannot oversample.")
            return df

        # Oversample fraud cases to match non-fraud count
        oversampled_fraud = fraud_df.sample(
            n=num_non_fraud, replace=True, random_state=SEED_TRAINING
        )

        # Combine and shuffle
        oversampled_df = pd.concat([oversampled_fraud, non_fraud_df], ignore_index=True)
        oversampled_df = oversampled_df.sample(
            frac=1, random_state=SEED_TRAINING
        ).reset_index(drop=True)

        logging.info(
            f"After oversampling: Fraud={len(oversampled_df[oversampled_df['is_fraud'] == 1])}, Non-fraud={len(oversampled_df[oversampled_df['is_fraud'] == 0])}"
        )

        return oversampled_df

    def train_dataloader(self):
        """
        Create the training dataloader with dynamic undersampling support.

        Returns:
            DataLoader: PyTorch DataLoader for training data
        """
        if self.undersample:
            logging.info("Using dynamic undersampling for training data at each epoch")

            # Create a permutable dataset that will be resampled at each epoch
            fraud_indices = np.where(self.train_df["is_fraud"] == 1)[0]
            non_fraud_indices = np.where(self.train_df["is_fraud"] == 0)[0]

            if len(fraud_indices) == 0:
                logging.warning("No fraud samples found. Cannot undersample.")
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.config.get("num_workers", 4),
                )

            # Create the permutable undersampling dataset
            self.permutable_train_dataset = PermutableUndersamplingDechowDataset(
                features=self.train_dataset.features,
                labels=self.train_dataset.labels,
                negative_positive_ratio=1.0,  # 1:1 ratio by default
                seed=SEED_TRAINING,
            )

            return DataLoader(
                self.permutable_train_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No need to shuffle as the dataset is already shuffled in permute()
                num_workers=self.config.get("num_workers", 4),
            )

        # Default dataloader without undersampling
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
        )

    def on_train_epoch_start(self):
        """
        Called at the beginning of each training epoch.
        Permute the dataset if using dynamic undersampling.
        """
        if self.undersample and hasattr(self, "permutable_train_dataset"):
            logging.info(
                f"Epoch {self.current_epoch}: Permuting training dataset for undersampling."
            )
            self.permutable_train_dataset.permute()

    def val_dataloader(self):
        """
        Create the validation dataloader.

        Returns:
            DataLoader: PyTorch DataLoader for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
        )

    def test_dataloader(self):
        """
        Create the test dataloader.

        Returns:
            DataLoader: PyTorch DataLoader for test data
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
        )

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            dict: Optimizer configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        """
        Training step for a batch of data.

        Args:
            batch (tuple): Tuple of (features, labels)
            batch_idx (int): Batch index

        Returns:
            dict: Training loss and metrics
        """
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        preds = torch.sigmoid(logits)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "labels": y}

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a batch of data.

        Args:
            batch (tuple): Tuple of (features, labels)
            batch_idx (int): Batch index

        Returns:
            dict: Validation loss and metrics
        """
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        # Calculate predictions
        preds = torch.sigmoid(logits)

        # Store predictions and labels for epoch-end processing
        self.val_preds.append(preds.cpu().detach())
        self.val_labels.append(y.cpu().detach())

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "preds": preds, "labels": y}

    def on_validation_epoch_end(self):
        """
        Process the outputs of all validation steps at the end of the epoch.

        Args:
            outputs (list): List of outputs from each validation step
        """
        # Collect predictions and labels
        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        # Reset stored predictions and labels
        self.val_preds = []
        self.val_labels = []

        # Calculate metrics
        auc = roc_auc_score(labels, preds)

        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(labels, preds)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
        )

        # Store the optimal threshold
        self.best_threshold = optimal_threshold

        # Calculate metrics using optimal threshold
        preds_binary = (preds >= optimal_threshold).astype(int)
        accuracy = accuracy_score(labels, preds_binary)
        precision_val = precision_score(labels, preds_binary, zero_division=0)
        recall_val = recall_score(labels, preds_binary, zero_division=0)
        f1 = f1_score(labels, preds_binary, zero_division=0)

        # Log metrics
        self.log("val_auc", auc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision_val, on_epoch=True)
        self.log("val_recall", recall_val, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True)
        self.log("best_threshold", optimal_threshold, on_epoch=True)

        # Plot PR curve
        try:
            self.plot_pr_curve(
                precision, recall, optimal_threshold, f1_scores[optimal_idx]
            )
        except Exception as e:
            logging.error(f"Error plotting PR curve: {e}")

        # Log detailed metrics
        logging.info(f"Epoch {self.current_epoch} - Validation metrics:")
        logging.info(f"  AUC: {auc:.4f}")
        logging.info(f"  Best threshold: {optimal_threshold:.4f}")
        logging.info(f"  F1 Score: {f1:.4f}")
        logging.info(f"  Precision: {precision_val:.4f}")
        logging.info(f"  Recall: {recall_val:.4f}")
        logging.info(f"  Accuracy: {accuracy:.4f}")

        # Save validation predictions
        val_results = pd.DataFrame(
            {
                "cik": self.val_meta["cik"].values,
                "year": self.val_meta["year"].values,
                "quarter": self.val_meta["quarter"].values,
                "fraud_probability": preds.flatten(),
                "predicted_fraud": preds_binary.flatten(),
                "actual_fraud": labels.flatten(),
            }
        )

        # Save validation results to CSV
        epoch_results_path = (
            self.log_dir / f"val_predictions_epoch_{self.current_epoch}.csv"
        )
        val_results.to_csv(epoch_results_path, index=False)

        # metrics json
        metrics_dict = {
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision_val),
            "recall": float(recall_val),
            "accuracy": float(accuracy),
            "threshold": float(optimal_threshold),
            "num_val_samples": int(len(labels)),
            "num_fraud_samples_val": int(np.sum(labels)),
        }

        metrics_path = self.log_dir / f"metrics_epoch_{self.current_epoch}.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)

    def test_step(self, batch, batch_idx):
        """
        Test step for a batch of data.

        Args:
            batch (tuple): Tuple of (features, labels)
            batch_idx (int): Batch index

        Returns:
            dict: Test predictions and labels
        """
        x, y = batch
        logits = self(x)
        preds = torch.sigmoid(logits)

        out = {"preds": preds, "labels": y}
        self.test_outputs.append(out)
        return out

    def on_test_epoch_start(self):
        """
        Called at the start of the test epoch.
        Reset stored predictions and labels for test step.
        """
        self.test_outputs = []

    def on_test_epoch_end(self):
        """
        Process the outputs of all test steps at the end of the epoch.

        Args:
            outputs (list): List of outputs from each test step
        """
        outputs = self.test_outputs
        # Collect predictions and labels
        preds = (
            torch.cat([output["preds"] for output in outputs]).cpu().detach().numpy()
        )
        labels = (
            torch.cat([output["labels"] for output in outputs]).cpu().detach().numpy()
        )

        # Use optimal threshold from validation
        threshold = self.best_threshold
        preds_binary = (preds >= threshold).astype(int)

        # Calculate metrics
        auc = roc_auc_score(labels, preds)
        accuracy = accuracy_score(labels, preds_binary)
        precision_val = precision_score(labels, preds_binary, zero_division=0)
        recall_val = recall_score(labels, preds_binary, zero_division=0)
        f1 = f1_score(labels, preds_binary, zero_division=0)

        # Get classification report
        report = classification_report(
            labels, preds_binary, target_names=["Not Fraud", "Fraud"], zero_division=0
        )

        # Log metrics
        self.log("test_auc", auc)
        self.log("test_f1", f1)
        self.log("test_precision", precision_val)
        self.log("test_recall", recall_val)
        self.log("test_accuracy", accuracy)

        # Log detailed results
        logging.info(f"Test Results (threshold={threshold:.4f}):")
        logging.info(f"  AUC: {auc:.4f}")
        logging.info(f"  F1 Score: {f1:.4f}")
        logging.info(f"  Precision: {precision_val:.4f}")
        logging.info(f"  Recall: {recall_val:.4f}")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"\nClassification Report:\n{report}")

        # Save test predictions
        test_results = pd.DataFrame(
            {
                "cik": self.test_meta["cik"].values,
                "year": self.test_meta["year"].values,
                "quarter": self.test_meta["quarter"].values,
                "fraud_probability": preds.flatten(),
                "predicted_fraud": preds_binary.flatten(),
                "actual_fraud": labels.flatten(),
            }
        )

        # Save test results to CSV
        results_path = self.log_dir / "test_predictions.csv"
        test_results.to_csv(results_path, index=False)
        logging.info(f"Test predictions saved to {results_path}")

        # Save metrics to JSON
        import json

        metrics = {
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision_val),
            "recall": float(recall_val),
            "accuracy": float(accuracy),
            "threshold": float(threshold),
            "num_test_samples": int(len(labels)),
            "num_fraud_samples_test": int(np.sum(labels)),
        }

        metrics_path = self.log_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Test metrics saved to {metrics_path}")

    def plot_pr_curve(self, precision, recall, threshold, f1):
        """
        Plot precision-recall curve and save to the log directory.

        Args:
            precision (np.ndarray): Precision values
            recall (np.ndarray): Recall values
            threshold (float): Optimal threshold
            f1 (float): Best F1 score
        """
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker=".")
        plt.scatter(
            [
                recall[
                    np.where(precision * recall == f1 * (precision + recall) / 2)[0][0]
                ]
            ],
            [
                precision[
                    np.where(precision * recall == f1 * (precision + recall) / 2)[0][0]
                ]
            ],
            s=100,
            marker="o",
            color="red",
            label=f"Optimal (F1={f1:.4f}, Thresh={threshold:.4f})",
        )

        plt.title(f"Precision-Recall Curve - Epoch {self.current_epoch}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.legend()

        # Save plot
        plot_path = self.log_dir / f"pr_curve_epoch_{self.current_epoch}.png"
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"PR curve saved to {plot_path}")

    def fit(self, train_path=None, test_path=None, trainer_kwargs=None):
        """
        Train the model using PyTorch Lightning Trainer.

        Args:
            train_path (Path or str, optional): Path to training data CSV
            test_path (Path or str, optional): Path to test data CSV
            trainer_kwargs (dict, optional): Additional arguments for PyTorch Lightning Trainer

        Returns:
            self: Trained model
        """
        # Load data if paths provided
        if train_path is not None and test_path is not None:
            self.train_df, self.val_df, self.test_df = self.load_data(
                train_path, test_path
            )
            logging.info(
                "Data loaded successfully. Sizes : "
                f"Train: {len(self.train_df)}, (Fraud: {self.train_df['is_fraud'].sum()}) "
                f"Validation: {len(self.val_df)}, (Fraud: {self.val_df['is_fraud'].sum()}) "
                f"Test: {len(self.test_df)}, (Fraud: {self.test_df['is_fraud'].sum()})"
            )

        elif (
            not hasattr(self, "train_df")
            or not hasattr(self, "val_df")
            or not hasattr(self, "test_df")
        ):
            raise ValueError(
                "No data provided. Either call load_data() first or provide paths."
            )

        # Prepare data
        self.setup(stage="fit")

        # Default trainer arguments
        default_trainer_kwargs = {
            "max_epochs": self.config.get("num_epochs", 10),
            "default_root_dir": self.log_dir,
            "num_sanity_val_steps": 0,
            "logger": pl.loggers.TensorBoardLogger(save_dir=self.log_dir),
            "callbacks": [
                pl.callbacks.ModelCheckpoint(
                    monitor="val_auc",
                    mode="max",
                    save_top_k=1,
                    dirpath=self.log_dir / "checkpoints",
                    filename="best_model-{epoch:02d}-{val_auc:.4f}",
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=self.config.get("early_stopping_patience", 10),
                    min_delta=0.001,
                ),
                pl.callbacks.LearningRateMonitor("epoch"),
            ],
        }

        # Update with provided trainer arguments
        if trainer_kwargs:
            default_trainer_kwargs.update(trainer_kwargs)

        # Initialize trainer
        trainer = pl.Trainer(**default_trainer_kwargs)

        # Train model
        trainer.fit(self)

        # Test model using best model
        trainer.test(ckpt_path="best")

        return self


def train_and_evaluate_dechow_model(config):
    """
    Train and evaluate a Dechow logistic regression model.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (model, metrics_dict, test_predictions_df)
    """
    # Create model
    model = LogisticFraudClassifierDechow(config)

    from researchpkg.anomaly_detection.models.utils import load_cross_validation_path

    train_path, test_path = load_cross_validation_path(config)

    # Train and evaluate model
    model.fit(train_path=train_path, test_path=test_path)

    # Get test metrics
    metrics_file = model.log_dir / "test_metrics.json"
    if metrics_file.exists():
        import json

        with open(metrics_file, "r") as f:
            metrics_dict = json.load(f)
    else:
        metrics_dict = {}

    # Get test predictions
    predictions_file = model.log_dir / "test_predictions.csv"
    if predictions_file.exists():
        test_predictions_df = pd.read_csv(predictions_file)
    else:
        test_predictions_df = pd.DataFrame()

    return model, metrics_dict, test_predictions_df


if __name__ == "__main__":
    # Example configuration
    CONFIG = {
        "hidden_layers": [],
        "learning_rate": 0.01,
        "batch_size": 8,
        "num_epochs": 20,
        "undersample": True,
        "weight_decay": 0.01,
        "early_stopping_patience": 100,
        "fold_id": 1,
        "dataset_version": "company_isolated_splitting",
        "auto_continue": int(os.environ.get("AUTO_CONTINUE", 0)),
    }

    # Train and evaluate model
    model, metrics, predictions = train_and_evaluate_dechow_model(CONFIG)

    # Print results
    print(f"Test metrics: {metrics}")
    print(f"Test predictions shape: {predictions.shape}")
