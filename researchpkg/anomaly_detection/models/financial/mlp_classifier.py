"""
Base MLP Classifier for Fraud Detection using PyTorch Lightning
---------------------------------------------------------------
Base implementation of a Multi-Layer Perceptron (MLP) binary classification model
for fraud detection using PyTorch Lightning. Includes Min-Max feature scaling.
"""

import json
import logging
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl  # Import PyTorch Lightning
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import average_precision_score  # Can use torchmetrics
from sklearn.metrics import (  # accuracy_score, # Calculated internally if needed or via torchmetrics; f1_score, # Use torchmetrics; precision_score, # Use torchmetrics; recall_score, # Use torchmetrics; roc_auc_score, # Use torchmetrics
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset  # Added Dataset
from torchmetrics import Precision  # Added Precision
from torchmetrics import Recall  # Added Recall
from torchmetrics import AUROC, AveragePrecision, F1Score  # Use torchmetrics
from tqdm import tqdm

from researchpkg.anomaly_detection.config import MLP_EXPERIMENTS_DIR, SEED_TRAINING
from researchpkg.anomaly_detection.models.financial.base_financial_classifier import (
    BaseFinancialClassifier,
)
from researchpkg.anomaly_detection.models.utils import (
    NumericalFeaturesType,
    load_cross_validation_path,
)
from researchpkg.utils import numpy_to_scalar, torch_to_scalar

pl.seed_everything(SEED_TRAINING, workers=True)
from pathlib import PosixPath

from torch.serialization import add_safe_globals

add_safe_globals([PosixPath])

# Keep MLP definition separate or move inside LightningModule if preferred
class MLP(nn.Module):
    """Simple MLP model definition."""

    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.5):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LightningMLP(pl.LightningModule):
    """PyTorch Lightning module for the MLP classifier."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_rate,
        learning_rate,
        pos_weight,
        decision_threshold,
    ):
        super().__init__()
        self.save_hyperparameters()  # Automatically logs hyperparameters

        self.model = MLP(input_dim, hidden_dims, 1, dropout_rate)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )  # Pass pos_weight here

        # --- Standard Binary Metrics (using the specified threshold) ---
        self.train_f1 = F1Score(task="binary", threshold=decision_threshold)
        self.val_f1 = F1Score(task="binary", threshold=decision_threshold)
        self.test_f1 = F1Score(task="binary", threshold=decision_threshold)

        self.val_precision = Precision(task="binary", threshold=decision_threshold)
        self.test_precision = Precision(task="binary", threshold=decision_threshold)

        self.val_recall = Recall(task="binary", threshold=decision_threshold)
        self.test_recall = Recall(task="binary", threshold=decision_threshold)

        # --- Weighted and Macro F1 (using the specified threshold) ---
        self.train_f1_weighted = F1Score(
            task="multiclass", num_classes=2, average="weighted"
        )
        self.val_f1_weighted = F1Score(
            task="multiclass", num_classes=2, average="weighted"
        )
        self.test_f1_weighted = F1Score(
            task="multiclass", num_classes=2, average="weighted"
        )

        self.train_f1_macro = F1Score(task="multiclass", num_classes=2, average="macro")
        self.val_f1_macro = F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_f1_macro = F1Score(task="multiclass", num_classes=2, average="macro")

        # --- Threshold-Independent Metrics (AUC, AP) ---
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        self.val_ap = AveragePrecision(task="binary")
        self.test_ap = AveragePrecision(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Optional: Update training metrics (can slow down training)
        # preds_proba = torch.sigmoid(logits)
        # y_int = y.int()
        # self.train_f1.update(preds_proba, y_int)
        # self.train_f1_weighted.update(preds_proba, y_int)
        # self.train_f1_macro.update(preds_proba, y_int)
        # self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_f1_weighted', self.train_f1_weighted, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('train_f1_macro', self.train_f1_macro, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds_proba = torch.sigmoid(logits)
        y_int = y.int()  # Ensure labels are integer type for torchmetrics

        # Log validation loss
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Update and log validation metrics
        self.val_f1.update(preds_proba, y_int)
        self.val_f1_weighted.update(preds_proba, y_int)
        self.val_f1_macro.update(preds_proba, y_int)
        self.val_precision.update(preds_proba, y_int)
        self.val_recall.update(preds_proba, y_int)
        self.val_auc.update(preds_proba, y_int)
        self.val_ap.update(preds_proba, y_int)

        self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "val_f1_weighted",
            self.val_f1_weighted,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_f1_macro", self.val_f1_macro, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_precision",
            self.val_precision,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recall", self.val_recall, on_epoch=True, prog_bar=True, logger=True
        )

        self.log("val_ap", self.val_ap, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds_proba = torch.sigmoid(logits)
        y_int = y.int()  # Ensure labels are integer type for torchmetrics

        # Log test loss
        self.log("test_loss", loss, on_epoch=True, logger=True)

        # Update and log test metrics
        self.test_f1.update(preds_proba, y_int)
        self.test_f1_weighted.update(preds_proba, y_int)
        self.test_f1_macro.update(preds_proba, y_int)
        self.test_precision.update(preds_proba, y_int)
        self.test_recall.update(preds_proba, y_int)
        self.test_auc.update(preds_proba, y_int)
        self.test_ap.update(preds_proba, y_int)

        self.log("test_f1", self.test_f1, on_epoch=True, logger=True)
        self.log("test_f1_weighted", self.test_f1_weighted, on_epoch=True, logger=True)
        self.log("test_f1_macro", self.test_f1_macro, on_epoch=True, logger=True)
        self.log("test_precision", self.test_precision, on_epoch=True, logger=True)
        self.log("test_recall", self.test_recall, on_epoch=True, logger=True)
        self.log("test_auc", self.test_auc, on_epoch=True, logger=True)
        self.log("test_ap", self.test_ap, on_epoch=True, logger=True)

        # Return predictions and labels for confusion matrix etc. outside Lightning
        return {"preds_proba": preds_proba, "labels": y}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=100,
            cooldown=5,
            verbose=True,
            min_lr=1e-6,
            eps=1e-08,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "monitor": "val_auc",
            "interval": "epoch",
        }
        return [optimizer], [scheduler_dict]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Used by trainer.predict()
        x, _ = batch  # Assuming prediction dataloader might have dummy labels
        logits = self(x)
        preds_proba = torch.sigmoid(logits)
        preds_binary = (preds_proba > self.hparams.decision_threshold).int()
        return preds_proba, preds_binary


class PermutableUndersamplingTensorDataset(Dataset):
    """
    A PyTorch Dataset wrapper that performs undersampling and allows permutation
    of non-fraud samples for each epoch.
    """

    def __init__(self, features, labels, random_state=None):
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        self.all_features = features
        self.all_labels = labels
        self.random_state = random_state if random_state is not None else SEED_TRAINING

        is_fraud = (labels.squeeze() == 1).numpy()
        self.fraud_indices = np.where(is_fraud)[0]
        self.non_fraud_indices = np.where(~is_fraud)[0]
        self.num_fraud = len(self.fraud_indices)

        if self.num_fraud == 0:
            logging.warning("No fraud cases found in the dataset for undersampling.")
            self.current_indices = np.arange(len(features))  # Use all data if no fraud
        elif self.num_fraud >= len(self.non_fraud_indices):
            logging.warning(
                "Fraud cases >= non-fraud cases. Undersampling will not reduce data."
            )
            self.current_indices = np.arange(len(features))  # Use all data
        else:
            self.permute()  # Initial permutation

    def permute(self):
        """Resamples non-fraud indices and shuffles with fraud indices."""
        if self.num_fraud == 0 or self.num_fraud >= len(self.non_fraud_indices):
            # No permutation needed if no fraud or fraud >= non-fraud
            self.current_indices = np.arange(len(self.all_features))
            return

        rng = np.random.default_rng(
            self.random_state
        )  # Use Generator for better practice
        selected_non_fraud_indices = rng.choice(
            self.non_fraud_indices,
            size=self.num_fraud,
            replace=False,
        )
        self.current_indices = np.concatenate(
            [self.fraud_indices, selected_non_fraud_indices]
        )
        rng.shuffle(self.current_indices)
        logging.info(
            f"Undersampling permutation: {self.num_fraud} fraud + {len(selected_non_fraud_indices)} non-fraud = {len(self.current_indices)} total samples."
        )
        # Update random state for next permutation if desired, or keep fixed
        # self.random_state += 1

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        original_idx = self.current_indices[idx]
        return self.all_features[original_idx], self.all_labels[original_idx]


class UndersamplingPermutationCallback(Callback):
    """PyTorch Lightning Callback to permute the undersampling dataset each epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        # Access the dataset via the dataloader
        # This assumes the train_dataloader is the first one if multiple are present
        train_dataloader = trainer.train_dataloader
        if hasattr(train_dataloader, "dataset") and isinstance(
            train_dataloader.dataset, PermutableUndersamplingTensorDataset
        ):
            logging.info(
                f"Permuting undersampling dataset for epoch {trainer.current_epoch}"
            )
            train_dataloader.dataset.permute()
        elif (
            hasattr(train_dataloader, "loaders")
            and hasattr(train_dataloader.loaders, "dataset")
            and isinstance(
                train_dataloader.loaders.dataset, PermutableUndersamplingTensorDataset
            )
        ):
            # Handle CombinedLoader case if necessary
            logging.info(
                f"Permuting undersampling dataset for epoch {trainer.current_epoch} (CombinedLoader)"
            )
            train_dataloader.loaders.dataset.permute()


class MLP_Classifier(BaseFinancialClassifier):
    """
    Base MLP classifier wrapper using PyTorch Lightning.
    Handles data loading, scaling, trainer setup, prediction, and saving.
    """

    BASE_EXPERIMENT_DIR = MLP_EXPERIMENTS_DIR  # Default experiments directory

    def __init__(self, features_type, config):
        config["hidden_dims"] = list(
            config["hidden_dims"]
        )  # Ensure hidden_dims is a list

        self.hidden_dims = config["hidden_dims"]
        self.dropout_rate = config["dropout_rate"]
        self.fold_id = config["fold_id"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.decision_threshold = config["decision_threshold"]
        self.oversample = config.get("oversample", False)
        self.undersample = config.get("undersample", False)
        self.standardize = config.get("standardize", True)
        self.trainer = None
        self.model = None
        self.best_model_path = None
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(features_type=features_type, config=config)
        self.model_name = config.get("model_name", self.parse_experiment_name())

    def parse_experiment_name(self):

        """Parses threturn super().parse_experiment_name()e experiment name from the configuration."""
        hiddens_dims = self.config["hidden_dims"]
        hidden_dims_str = "_".join(map(str, hiddens_dims))

        experiment_name = f"mlp_hd.{hidden_dims_str}_lr.{self.config['dropout_rate']}_lr_{self.config['learning_rate']}_"
        f"scale.{self.standardize}_"
        f"{self.features_type.name.lower()}"
        return experiment_name

    def _prepare_dataloader(
        self, X, y, shuffle=False, use_permutable_dataset=False, num_workers=0
    ):
        """Creates PyTorch DataLoader, optionally using PermutableUndersamplingTensorDataset."""
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = X
        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y

        if use_permutable_dataset:
            logging.info(
                "Creating DataLoader with PermutableUndersamplingTensorDataset."
            )
            dataset = PermutableUndersamplingTensorDataset(
                X_np, y_np, random_state=SEED_TRAINING
            )
            # Shuffle is handled internally by permutation, so set shuffle=False for DataLoader
            shuffle = False
        else:
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
            y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )  # Added num_workers
        return dataloader

    def fit(self, X_train_scaled, y_train, X_val_scaled, y_val):
        """Configures and runs the PyTorch Lightning Trainer."""
        logging.info("Configuring PyTorch Lightning Trainer")

        # --- Handle Sampling BEFORE creating DataLoader ---
        X_train_processed = X_train_scaled
        y_train_processed = y_train
        use_permutable = False

        if self.oversample:
            X_train_processed, y_train_processed = self.oversample_fraud_cases(
                X_train_scaled, y_train
            )
        elif self.undersample:
            # Undersampling is handled by the PermutableUndersamplingTensorDataset
            use_permutable = True
            # Pass the original scaled data to the permutable dataset
            X_train_processed = X_train_scaled
            y_train_processed = y_train

        # --- Create DataLoaders ---
        train_loader = self._prepare_dataloader(
            X_train_processed,
            y_train_processed,
            shuffle=(not use_permutable),
            num_workers=0,
            use_permutable_dataset=use_permutable,
        )
        val_loader = self._prepare_dataloader(
            X_val_scaled, y_val, shuffle=False, num_workers=0
        )

        input_dim = X_train_scaled.shape[1]  # Input dim based on original scaled data

        # Calculate pos_weight based on the potentially resampled training data
        y_train_for_weight = (
            y_train_processed.values
            if isinstance(y_train_processed, pd.Series)
            else y_train_processed
        )
        pos_count = np.sum(y_train_for_weight)
        neg_count = len(y_train_for_weight) - pos_count
        pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
        logging.info(
            f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_val:.4f} (based on {'resampled' if self.oversample or self.undersample else 'original'} train data)"
        )

        # Instantiate the LightningModule
        self.model = LightningMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            pos_weight=pos_weight_val,
            decision_threshold=self.decision_threshold,
        )

        # --- Configure Callbacks ---
        callbacks = []
        early_stop_callback = EarlyStopping(
            monitor="val_auc",
            patience=self.patience,
            verbose=True,
            mode="max",
        )
        callbacks.append(early_stop_callback)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_auc",
            dirpath=self.log_dir / "checkpoints",
            filename=f"{self.model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=1,
            mode="max",
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Add undersampling permutation callback if needed
        if self.undersample:
            callbacks.append(UndersamplingPermutationCallback())

        # Configure Logger
        tb_logger = TensorBoardLogger(
            save_dir=self.tensorboard_log_dir,
            name=self.model_name,
            version="",  # Use empty version to avoid subfolder
        )

        # Instantiate Trainer
        self.trainer = pl.Trainer(
            logger=tb_logger,
            callbacks=callbacks,  # Pass list of callbacks
            max_epochs=self.epochs,
            accelerator="auto",  # Automatically use GPU if available
            devices="auto",
            deterministic=True,
            log_every_n_steps=5,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
        )

        print("Fit MLP model with PyTorch Lightning...")
        self.trainer.fit(self.model, train_loader, val_loader)
        logging.info("Training finished.")

        self.best_model_path = checkpoint_callback.best_model_path
        logging.info(f"Best model saved at: {self.best_model_path}")

        if self.best_model_path:
            logging.info(f"Loading best model weights from {self.best_model_path}")
            # Reload model using the checkpoint path
            self.model = LightningMLP.load_from_checkpoint(
                checkpoint_path=self.best_model_path
            )
        else:
            logging.warning(
                "Could not find best model path. Using model from last epoch."
            )

        return self.trainer.callback_metrics

    def evaluate(self, X_test_scaled, y_test, subset="Test", save_metrics=False):
        """Evaluates the model using trainer.test() and calculates confusion matrix."""
        if self.trainer is None or self.model is None:
            # If trainer wasn't used (e.g., loaded model), we might need manual evaluation
            logging.warning("Trainer not available, attempting manual evaluation.")
            # Fallback to manual evaluation if needed, or ensure trainer is available
            # For now, assume trainer is available after fit or model is loaded correctly for predict
            if self.model is None:
                raise ValueError("Model has not been trained or loaded properly.")

        logging.info(f"Evaluating model on {subset} data...")
        test_loader = self._prepare_dataloader(X_test_scaled, y_test, shuffle=False)

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        # Use trainer.test() if available and model was trained with it
        test_results = [
            {"test_loss": -1, "test_f1": -1, "test_auc": -1, "test_ap": -1}
        ]  # Default
        if self.trainer:
            try:
                # Ensure the model's metrics are reset before test run if reusing trainer
                # self.model.test_f1.reset() # etc. - This is usually handled by Lightning internally
                test_results = self.trainer.test(
                    self.model, dataloaders=test_loader, verbose=True
                )
                # test_results is a list containing a dict of metrics logged in test_step
                logging.info(f"{subset} Metrics (from Trainer.test): {test_results}")
            except Exception as e:
                logging.warning(
                    f"trainer.test() failed: {e}. Proceeding with manual prediction."
                )
                test_results = [{}]  # Ensure test_results[0] exists
                # Proceed to manual prediction below

        # Get predictions manually for detailed report regardless of trainer.test() success
        logging.info(f"Generating predictions for {subset} set for detailed report...")
        all_preds_proba = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting {subset}"):
                inputs, labels = batch
                inputs = inputs.to(self.model.device)
                logits = self.model(inputs)
                preds_proba = torch.sigmoid(logits).cpu().numpy()
                all_preds_proba.extend(preds_proba.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        all_labels = np.array(all_labels)
        all_preds_proba = np.array(all_preds_proba)
        y_pred_binary = (all_preds_proba > self.decision_threshold).astype(int)

        # --- Calculate and Plot Confusion Matrix ---
        cm = confusion_matrix(all_labels, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        figure_path = self.log_dir / f"{subset}_confusion_matrix.png"
        plt.savefig(figure_path)
        logging.info(f"{subset} Confusion Matrix saved to {figure_path}")
        plt.close()

        # --- Calculate and Plot ROC Curve ---
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_preds_proba)
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
        roc_disp.plot()
        figure_path = self.log_dir / f"{subset}_roc_curve.png"
        plt.savefig(figure_path)
        logging.info(f"{subset} ROC curve saved to {figure_path}")
        plt.close()

        # --- Calculate Best Threshold (Youden's J) ---
        # best_threshold_roc = 0.5
        # if len(thresholds_roc) > 1 and len(np.unique(all_labels)) > 1:
        #     youden_j = tpr - fpr
        #     # Handle cases where thresholds_roc might not align perfectly with fpr/tpr length
        #     valid_indices = np.arange(min(len(tpr), len(fpr), len(thresholds_roc)))
        #     best_idx = youden_j[valid_indices].argmax()
        #     best_threshold_roc = thresholds_roc[valid_indices[best_idx]]
        # logging.info(
        #     f"Best decision threshold from ROC (Youden's J): {best_threshold_roc:.4f}"
        # )

        # Calculate threshold that maximizes the f1 score
        y_pred_proba = all_preds_proba
        y_true = all_labels
        best_threshold_roc = self.find_best_threshold(y_true, y_pred_proba)
        logging.info(
            f"Best decision threshold from ROC (F1 maximization): {best_threshold_roc:.4f}"
        )

        # --- Classification Report ---
        target_names = ["Non Fraud", "Fraud"]
        report = classification_report(
            all_labels, y_pred_binary, target_names=target_names, zero_division=0
        )
        logging.info(f"{subset} Classification Report:\n{report}")

        # Combine metrics - prioritize trainer.test() results as they use torchmetrics correctly
        final_metrics = {}
        if test_results and isinstance(test_results, list) and test_results[0]:
            # Convert tensor values in results to float if necessary
            final_metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in test_results[0].items()
            }

        # --- Manual Calculation (Optional - Good for verification or if trainer.test fails) ---
        # Calculate metrics manually using sklearn for comparison/fallback
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        manual_f1 = f1_score(all_labels, y_pred_binary, zero_division=0)
        manual_precision = precision_score(all_labels, y_pred_binary, zero_division=0)
        manual_recall = recall_score(all_labels, y_pred_binary, zero_division=0)
        manual_auc = (
            roc_auc_score(all_labels, all_preds_proba)
            if len(np.unique(all_labels)) > 1
            else 0.0
        )
        manual_ap = average_precision_score(all_labels, all_preds_proba)

        # Add manual metrics (prefixed) to the results dict for comparison/logging
        final_metrics["f1"] = manual_f1
        final_metrics["precision"] = manual_precision
        final_metrics["recall"] = manual_recall
        final_metrics["auc"] = manual_auc
        final_metrics["manual_test_ap"] = manual_ap
        final_metrics["best_threshold_roc"] = best_threshold_roc
        final_metrics["used_threshold"] = best_threshold_roc
        final_metrics["confusion_matrix"] = cm.tolist()
        final_metrics["auc_score"] = manual_auc
        logging.info(f"Final {subset} Metrics (Combined): {final_metrics}")

        final_metrics = numpy_to_scalar(final_metrics)
        if save_metrics:
            metrics_path = self.log_dir / f"{subset}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(final_metrics, f, indent=4)
            logging.info(f"{subset} metrics saved to {metrics_path}")

        if subset.lower() == "test":
            test_predictions_path = self.log_dir / "test_predictions.csv"
            import pandas as pd

            predictions_df = pd.DataFrame(
                {
                    "y_true_id": y_test,
                    "fraud_probability": y_pred_proba,
                    "y_pred_id": (y_pred_binary),
                }
            )
            predictions_df.to_csv(test_predictions_path, index=False)
            logging.info(f"Test predictions saved to {test_predictions_path}")

        return final_metrics

    def predict(self, X_scaled):
        """Make predictions on new data. Assumes input data X_scaled is already scaled."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        if self.scaler is None:
            raise ValueError(
                "Scaler has not been fitted. Train or load the model first."
            )

        self.model.eval()
        self.model.to(self.device_type)  # Ensure model is on correct device

        # Convert scaled numpy data to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device_type)

        # Create a dummy DataLoader for prediction
        # No labels needed, use a simple TensorDataset
        predict_dataset = TensorDataset(
            X_tensor, torch.zeros(len(X_tensor))
        )  # Dummy labels
        predict_loader = DataLoader(
            predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Use trainer.predict if trainer is available, otherwise manual loop
        if self.trainer:
            logging.info("Using trainer.predict for predictions.")
            predictions = self.trainer.predict(self.model, dataloaders=predict_loader)
            # predictions is a list of batches, concatenate them
            y_pred_proba = (
                torch.cat([p[0] for p in predictions]).cpu().numpy().flatten()
            )
            y_pred_binary = (
                torch.cat([p[1] for p in predictions]).cpu().numpy().flatten()
            )
        else:
            # Manual prediction loop if trainer isn't available (e.g., after loading)
            logging.info("Using manual loop for predictions.")
            all_preds_proba = []
            all_preds_binary = []
            with torch.no_grad():
                for batch in tqdm(predict_loader, desc="Predicting"):
                    inputs, _ = batch  # Ignore dummy labels
                    inputs = inputs.to(self.device_type)
                    logits = self.model(inputs)
                    preds_proba = torch.sigmoid(logits)
                    preds_binary = (preds_proba > self.decision_threshold).int()
                    all_preds_proba.append(preds_proba.cpu())
                    all_preds_binary.append(preds_binary.cpu())
            y_pred_proba = torch.cat(all_preds_proba).numpy().flatten()
            y_pred_binary = torch.cat(all_preds_binary).numpy().flatten()

        return y_pred_proba, y_pred_binary

    def save_model(self, filepath=None):
        """Save the scaler and log the checkpoint path."""
        if self.scaler is None:
            raise ValueError("Scaler not available to save")
        if self.best_model_path is None and self.trainer is not None:
            # If training just finished but best_model_path wasn't set (e.g., error), try last checkpoint
            last_ckpt_path = self.trainer.checkpoint_callback.last_model_path
            if last_ckpt_path:
                self.best_model_path = last_ckpt_path
                logging.warning(
                    f"Best model path not found, using last checkpoint: {self.best_model_path}"
                )
            else:
                raise ValueError("No model checkpoint path found to save.")
        elif self.best_model_path is None:
            raise ValueError(
                "Model has not been trained or loaded, cannot determine checkpoint path."
            )

        scaler_path = self.log_dir / "scaler.pkl"

        # Model is saved by ModelCheckpoint callback. Log the path.
        logging.info(
            f"PyTorch Lightning model checkpoint saved at: {self.best_model_path}"
        )

        # Save scaler separately
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logging.info(f"Scaler saved to {scaler_path}")

        return str(self.best_model_path), str(scaler_path)

    def load_model(self, model_filepath, scaler_filepath):
        """Load a Lightning model checkpoint and the scaler."""
        logging.info(f"Loading model checkpoint from {model_filepath}")
        logging.info(f"Loading scaler from {scaler_filepath}")

        # Load Lightning model from checkpoint
        # We need the class definition (LightningMLP) to load
        try:
            self.model = LightningMLP.load_from_checkpoint(model_filepath)
            # Load hyperparameters if needed (already done by load_from_checkpoint if saved)
            self.hidden_dims = self.model.hparams.hidden_dims
            self.dropout_rate = self.model.hparams.dropout_rate
            self.learning_rate = self.model.hparams.learning_rate
            self.decision_threshold = self.model.hparams.decision_threshold
            # Note: input_dim and pos_weight might be needed if not in hparams
            self.model.eval()  # Set to evaluation mode
            self.best_model_path = model_filepath  # Store path
        except Exception as e:
            logging.error(f"Error loading Lightning checkpoint: {e}")
            logging.error(
                "Ensure the LightningMLP class definition is available and hyperparameters match."
            )
            raise

        # Load scaler
        with open(scaler_filepath, "rb") as f:
            self.scaler = pickle.load(f)
        try:
            self.feature_cols = self.scaler.feature_names_in_
        except AttributeError:
            logging.warning("Could not retrieve feature names from loaded scaler.")
            self.feature_cols = None

        logging.info("Model and scaler loaded successfully.")
        # Note: Trainer is not recreated here, only the model module.
        # If you need trainer functions like predict, you might need to re-instantiate it.
        return self

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
        if self.best_model_path and "epoch=" in self.best_model_path:
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
            hparams_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"

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


def train_and_evaluate_mlp_model(config: dict):
    """
    Base function to train and evaluate an MLP fraud classifier using PyTorch Lightning.
    """

    # Initialize the wrapper class, passing all necessary params
    model_name = config["model_name"]
    model = MLP_Classifier(**config)

    dataset_version = config["dataset_version"]
    logging.info(
        f"Starting {model_name} training and evaluation for dataset {dataset_version}"
    )

    # Seed Everyting
    # Seed after model intiilaiation
    logging.info(f"Setting random seed to {SEED_TRAINING}")
    np.random.seed(SEED_TRAINING)
    random.seed(SEED_TRAINING)
    torch.manual_seed(SEED_TRAINING)

    train_path, test_path = load_cross_validation_path(config)

    # Load and scale data using the wrapper's method
    (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
    ) = model.load_data(train_path, test_path)
    logging.info(
        f"Data loaded and scaled successfully with {X_train_scaled.shape[1]} features of type {model.features_type.name}"
    )

    # Train the model
    final_val_metrics = model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
    logging.info(f"Final Validation Metrics (from last epoch): {final_val_metrics}")

    # Evaluate the best model on the test set
    logging.info("Evaluating best model on Test set...")
    metrics_test = model.evaluate(X_test_scaled, y_test, subset="Test")

    # Plot precision-recall curve
    # wrapper.plot_precision_recall_curve(
    #     X_test_scaled,
    #     y_test,
    #     save_path=wrapper.log_dir / f"precision_recall_curve.png",
    # )

    # Save the experiment results
    model.save_experiment_results(final_val_metrics, metrics_test)

    # Save the scaler (model checkpoint is saved by Lightning callback)
    model.save_model()

    logging.info(f"{model_name} training and evaluation completed")
    return model, metrics_test


if __name__ == "__main__":

    CONFIG = {
        "model_name": "mlp_classifier",
        "features_type": NumericalFeaturesType.EXTENDED,
        "hidden_dims": [64, 32],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 10,
        "standardize": True,
        "patience": 5,
        "decision_threshold": 0.5,
        "fold_id": 2,
        "dataset_version": "time_splitting",
        "oversample": False,
    }

    model = MLP_Classifier(features_type=CONFIG["features_type"], config=CONFIG)

    train_path, test_path = load_cross_validation_path(CONFIG)
    data = model.load_data(train_path=train_path, test_path=test_path)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = data

    print("Training MLP model with PyTorch Lightning...")
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    print("Evaluating MLP model on test set...")
    metrics_test = model.evaluate(X_test_scaled, y_test, subset="Test")

    print(f"Test Metrics: {metrics_test}")
    model.save_experiment_results(metrics_val={}, metrics_test=metrics_test)

    # Save test predictions
    test_file = "/tmp/test_predictions_mlp_classifier.csv"
    model.make_predictions_csv(
        X_test_scaled,
        y_test,
        test_file,
    )
