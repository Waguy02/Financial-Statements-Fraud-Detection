# rcma_trainable_sbert_02_modified.py
"""
RCMA-inspired classifier for fraud detection combining financial data with MDA text
"""

import hashlib
import json
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve  # Added for Youden's J
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, AveragePrecision, F1Score, Precision, Recall
from tqdm import tqdm
from transformers import AutoTokenizer

from researchpkg.anomaly_detection.config import RCMA_EXPERIMENTS_DIR, SEED_TRAINING
from researchpkg.anomaly_detection.models.rcma.utils_rcma import load_rcma_dataset
from researchpkg.anomaly_detection.models.utils import (
    load_cross_validation_path,
    load_sic_industry_title_index,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
)
from researchpkg.utils import configure_logger, numpy_to_scalar, torch_to_scalar

pl.seed_everything(SEED_TRAINING, workers=True)
MAX_MDA_LENGTH = 512


class RcmaFocalLoss(nn.Module):
    def __init__(self, beta=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        logging.info(f"Initialized PaperFocalLoss with beta={beta}, gamma={gamma}")

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        log_p = F.logsigmoid(logits)
        loss_pos = -self.beta * ((1 - p) ** self.gamma) * log_p
        log_one_minus_p = F.logsigmoid(-logits)
        loss_neg = -(1 - self.beta) * (p**self.gamma) * log_one_minus_p
        loss = targets * loss_pos + (1 - targets) * loss_neg
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MDAFinancialDataset(Dataset):
    def __init__(
        self,
        grouped_financial_features,
        labels,
        mda_texts,
        tokenizer_for_trainable_sbert,
        max_length=MAX_MDA_LENGTH,
        precomputed_sbert_embeddings=None,
    ):
        self.grouped_financial_features = grouped_financial_features
        self.labels = labels
        self.mda_texts = mda_texts
        self.tokenizer_for_trainable_sbert = tokenizer_for_trainable_sbert
        self.max_length = max_length
        self.precomputed_sbert_embeddings = precomputed_sbert_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fin_features_grouped = self.grouped_financial_features[idx]
        label = self.labels[idx]
        item = {"financial_features_grouped": fin_features_grouped, "labels": label}
        if self.precomputed_sbert_embeddings is not None:
            item["sbert_embedding"] = self.precomputed_sbert_embeddings[idx]
            item["use_precomputed"] = True
        else:
            mda_text = self.mda_texts[idx] if idx < len(self.mda_texts) else ""
            if self.tokenizer_for_trainable_sbert is None:
                raise ValueError(
                    "Tokenizer is None, but SBERT embeddings are not precomputed."
                )
            encoded_mda = self.tokenizer_for_trainable_sbert(
                mda_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = encoded_mda["input_ids"].squeeze(0)
            item["attention_mask"] = encoded_mda["attention_mask"].squeeze(0)
            item["use_precomputed"] = False
        return item


class RCMAInspiredNet(nn.Module):
    def __init__(
        self,
        num_original_financial_features,
        num_financial_groups,
        sbert_model_name,
        sbert_output_dim,
        trainable_sbert_layers=0,
        financial_embedding_dim=128,
        text_embedding_dim=128,
        mlp_hidden_dims_paper=[256, 128],
        dropout_rate=0.3,
        max_mda_length_for_trainable_sbert=MAX_MDA_LENGTH,
        target_modules=None,
    ):
        super().__init__()
        self.sbert_model_name = sbert_model_name
        self.trainable_sbert_layers = trainable_sbert_layers
        self.sbert_output_dim = sbert_output_dim
        self.max_mda_length_for_trainable_sbert = max_mda_length_for_trainable_sbert
        self.num_original_financial_features = num_original_financial_features
        self.num_financial_groups = num_financial_groups
        self.target_modules = target_modules or []
        self.financial_group_query_u = nn.Parameter(
            torch.Tensor(self.num_original_financial_features)
        )
        nn.init.xavier_uniform_(self.financial_group_query_u.unsqueeze(0))

        fin_mlp_layers = []
        last_dim = self.num_original_financial_features
        for h_dim in mlp_hidden_dims_paper:
            fin_mlp_layers.extend(
                [nn.Linear(last_dim, h_dim), nn.Tanh(), nn.Dropout(dropout_rate)]
            )
            last_dim = h_dim
        fin_mlp_layers.append(nn.Linear(last_dim, financial_embedding_dim))
        fin_mlp_layers.append(nn.Tanh())
        self.financial_mlp_after_attention = nn.Sequential(*fin_mlp_layers)

        self.sbert_model = None
        if self.trainable_sbert_layers > 0:
            logging.info(f"Loading trainable SentenceTransformer: {sbert_model_name}")
            self.sbert_model = SentenceTransformer(
                sbert_model_name, trust_remote_code=True
            )
            self.sbert_model.max_seq_length = self.max_mda_length_for_trainable_sbert

            # Add LoRA adapter to train the model.
            from peft import LoraConfig, TaskType

            print("Model state dict:", self.sbert_model.state_dict().keys())

            logging.info("Adding Lora Adapter to SentenceTransformer model.")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                target_modules=self.target_modules,
                r=8,
                lora_alpha=8,
                lora_dropout=0,
            )
            self.sbert_model.add_adapter(peft_config)

        text_proj_mlp_layers = []
        last_dim_text = self.sbert_output_dim
        for h_dim in mlp_hidden_dims_paper:
            text_proj_mlp_layers.extend(
                [nn.Linear(last_dim_text, h_dim), nn.Tanh(), nn.Dropout(dropout_rate)]
            )
            last_dim_text = h_dim
        text_proj_mlp_layers.append(nn.Linear(last_dim_text, text_embedding_dim))
        text_proj_mlp_layers.append(nn.Tanh())
        self.text_projection_mlp = nn.Sequential(*text_proj_mlp_layers)

        self.ma_scorer_financial = nn.Linear(financial_embedding_dim, 1)
        self.ma_scorer_text = nn.Linear(text_embedding_dim, 1)
        assert (
            financial_embedding_dim == text_embedding_dim
        ), "Mf and Mt must have same dim for fusion."
        self.modal_feature_dim = financial_embedding_dim

        classifier_mlp_layers = []
        last_dim_classifier = self.modal_feature_dim
        for h_dim in mlp_hidden_dims_paper:
            classifier_mlp_layers.extend(
                [
                    nn.Linear(last_dim_classifier, h_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate),
                ]
            )
            last_dim_classifier = h_dim
        classifier_mlp_layers.append(nn.Linear(last_dim_classifier, 1))
        self.classifier_mlp = nn.Sequential(*classifier_mlp_layers)

    def forward(
        self,
        financial_features_grouped,
        input_ids=None,
        attention_mask=None,
        sbert_embedding=None,
    ):
        financial_features_transposed = financial_features_grouped.transpose(1, 2)
        scores_g = torch.einsum(
            "bgi,i->bg", financial_features_transposed, self.financial_group_query_u
        )
        alpha_f_group_weights = F.softmax(scores_g, dim=1)
        m_f = torch.einsum(
            "bg,bgi->bi", alpha_f_group_weights, financial_features_transposed
        )
        Mf = self.financial_mlp_after_attention(m_f)

        batch_size = Mf.size(0)
        raw_text_embed = None
        if sbert_embedding is not None:
            raw_text_embed = sbert_embedding.to(self.text_projection_mlp[0].weight.device, 
                                                dtype=torch.bfloat16)
        elif self.sbert_model is not None and input_ids is not None:
            sbert_output = self.sbert_model(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            raw_text_embed = sbert_output["sentence_embedding"]
        else:
            raw_text_embed = torch.zeros(
                batch_size, self.sbert_output_dim, device=Mf.device, dtype=Mf.dtype
            )
        Mt = self.text_projection_mlp(raw_text_embed)

        # Ablation: zero out financial embeddings
        # Mf = torch.zeros_like(Mf)  # Ablation: zero out financial embeddings
        score_mf = self.ma_scorer_financial(Mf)
        
        # Ablation: zero out text embeddings
        score_mt = self.ma_scorer_text(Mt)

        ma_scores = torch.cat([score_mf, score_mt], dim=1)
        ma_attention_weights = F.softmax(ma_scores, dim=1)
        weight_af = ma_attention_weights[:, 0].unsqueeze(1)
        weight_at = ma_attention_weights[:, 1].unsqueeze(1)
        Mm_fused = weight_af * Mf + weight_at * Mt
        logits = self.classifier_mlp(Mm_fused)
        return logits, Mf, Mt


class LightningRCMAClassifier(pl.LightningModule):
    def __init__(
        self,
        num_original_financial_features,
        num_financial_groups,
        sbert_model_name,
        sbert_output_dim,
        trainable_sbert_layers,
        financial_embedding_dim,
        text_embedding_dim,
        mlp_hidden_dims_paper,
        dropout_rate,
        learning_rate,
        pos_weight_beta,
        focal_gamma,
        consistency_loss_weight,
        max_mda_length_for_trainable_sbert,
        target_modules=None,
        # decision_threshold removed
    ):
        super().__init__()
        # Note: decision_threshold removed from save_hyperparameters if it was implicitly part of **kwargs
        self.save_hyperparameters(
            ignore=["decision_threshold"]
        )  # Explicitly ignore if it might come via kwargs

        self.model = RCMAInspiredNet(
            num_original_financial_features=num_original_financial_features,
            num_financial_groups=num_financial_groups,
            sbert_model_name=sbert_model_name,
            sbert_output_dim=sbert_output_dim,
            trainable_sbert_layers=trainable_sbert_layers,
            financial_embedding_dim=financial_embedding_dim,
            text_embedding_dim=text_embedding_dim,
            mlp_hidden_dims_paper=mlp_hidden_dims_paper,
            dropout_rate=dropout_rate,
            max_mda_length_for_trainable_sbert=max_mda_length_for_trainable_sbert,
            target_modules=target_modules,
        )
        self.focal_loss_fn = RcmaFocalLoss(
            beta=self.hparams.pos_weight_beta, gamma=self.hparams.focal_gamma
        )
        self.consistency_loss_fn = nn.CosineSimilarity(dim=1)

        # Metrics will use default threshold (0.5) or no threshold (AUC, AP)
        # Optimized F1 will be calculated and logged separately
        metrics_config_no_thresh = {"task": "binary"}
        for stage in ["val", "test"]:
            setattr(
                self,
                f"{stage}_f1_at_0.5",
                F1Score(**metrics_config_no_thresh, threshold=0.5),
            )  # Explicitly 0.5
            setattr(
                self,
                f"{stage}_precision_at_0.5",
                Precision(**metrics_config_no_thresh, threshold=0.5),
            )
            setattr(
                self,
                f"{stage}_recall_at_0.5",
                Recall(**metrics_config_no_thresh, threshold=0.5),
            )
            setattr(self, f"{stage}_auc", AUROC(**metrics_config_no_thresh))
            setattr(self, f"{stage}_ap", AveragePrecision(**metrics_config_no_thresh))

        self.validation_step_outputs = []

    def forward(
        self,
        financial_features_grouped,
        input_ids=None,
        attention_mask=None,
        sbert_embedding=None,
    ):
        return self.model(
            financial_features_grouped, input_ids, attention_mask, sbert_embedding
        )

    def _common_step(self, batch, batch_idx, stage="train"):
        financial_features_grouped = batch["financial_features_grouped"]
        y_true = batch["labels"]
        sbert_embedding, input_ids, attention_mask = None, None, None
        use_precomputed_batch = batch["use_precomputed"][0].item()

        if use_precomputed_batch:
            sbert_embedding = batch["sbert_embedding"]
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        logits, Mf, Mt = self.model(
            financial_features_grouped, input_ids, attention_mask, sbert_embedding
        )
        focal_loss = self.focal_loss_fn(logits, y_true)
        consistency_loss = -self.consistency_loss_fn(Mf, Mt).mean()
        total_loss = (
            focal_loss + self.hparams.consistency_loss_weight * consistency_loss
        )

        self.log(
            f"{stage}_focal_loss",
            focal_loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_consistency_loss",
            consistency_loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        preds_proba = torch.sigmoid(logits)
        y_int = y_true.int()

        if stage != "train":
            # Log metrics at 0.5 threshold using torchmetrics
            getattr(self, f"{stage}_f1_at_0.5").update(preds_proba, y_int)
            getattr(self, f"{stage}_precision_at_0.5").update(preds_proba, y_int)
            getattr(self, f"{stage}_recall_at_0.5").update(preds_proba, y_int)
            getattr(self, f"{stage}_auc").update(preds_proba, y_int)
            getattr(self, f"{stage}_ap").update(preds_proba, y_int)

        # For validation, collect outputs for threshold optimization
        if stage == "val":
            self.validation_step_outputs.append(
                {"preds_proba": preds_proba.detach(), "labels": y_true.detach()}
            )

        # For test, also return probabilities and labels for external evaluation
        if stage == "test":
            return {"preds_proba": preds_proba.detach(), "labels": y_true.detach()}

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        # _common_step now handles appending to self.validation_step_outputs
        self._common_step(batch, batch_idx, stage="val")
        # Validation_step in PL should not return anything when manual optimization loop is used in on_validation_epoch_end

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def _log_and_reset_torchmetrics_epoch_end(self, stage="val"):
        metrics_to_log = {}
        for metric_name_suffix in [
            "f1_at_0.5",
            "precision_at_0.5",
            "recall_at_0.5",
            "auc",
            "ap",
        ]:
            metric_obj = getattr(self, f"{stage}_{metric_name_suffix}")
            try:
                computed_metric = metric_obj.compute()
                full_metric_name = f"{stage}_{metric_name_suffix}"
                self.log(
                    full_metric_name,
                    computed_metric,
                    on_epoch=True,
                    prog_bar=(metric_name_suffix == "auc"),  # Changed to auc
                    logger=True,
                    sync_dist=True,
                )
                metrics_to_log[full_metric_name] = (
                    computed_metric.item()
                    if isinstance(computed_metric, torch.Tensor)
                    else computed_metric
                )
                metric_obj.reset()
            except Exception as e:
                logging.error(
                    f"Error logging/resetting {stage}_{metric_name_suffix}: {e}"
                )
        return metrics_to_log

    def on_validation_epoch_end(self):
        # Log standard metrics (e.g., at 0.5 threshold)
        epoch_metrics = self._log_and_reset_torchmetrics_epoch_end(stage="val")

        # Perform threshold optimization
        if not self.validation_step_outputs:
            logging.warning(
                "No outputs collected in validation_step_outputs. Skipping threshold optimization."
            )
            # Clear the list for the next epoch
            self.validation_step_outputs.clear()
            return

        all_val_probas = (
            torch.cat([out["preds_proba"] for out in self.validation_step_outputs])
            .squeeze()
            .cpu()
            .float()
            .numpy()
        )
        all_val_labels = (
            torch.cat([out["labels"] for out in self.validation_step_outputs])
            .squeeze()
            .cpu()
            .float()
            .numpy()
        )

        # Clear the list for the next epoch
        self.validation_step_outputs.clear()

        if len(all_val_probas) == 0 or len(all_val_labels) == 0:
            logging.warning(
                "Collected probabilities or labels are empty. Skipping threshold optimization."
            )
            return

        current_epoch = self.current_epoch
        log_dir_path = (
            Path(self.trainer.logger.log_dir)
            if self.trainer.logger and self.trainer.logger.log_dir
            else Path(".")
        )

        # Changed: Optimize threshold for ROC (Youden's J)
        best_val_threshold = find_optimal_roc_threshold(
            all_val_labels,
            all_val_probas,
            log_dir_path,
            prefix=f"epoch_{current_epoch}_val_",
        )

        # Calculate metrics at the optimized threshold
        y_pred_at_threshold = (all_val_probas >= best_val_threshold).astype(int)

        # Calculate all metrics with standardized names to match LLM classifier
        accuracy = accuracy_score(all_val_labels, y_pred_at_threshold)
        precision = precision_score(
            all_val_labels, y_pred_at_threshold, pos_label=1, zero_division=0
        )
        recall = recall_score(
            all_val_labels, y_pred_at_threshold, pos_label=1, zero_division=0
        )
        f1_final = f1_score(
            all_val_labels, y_pred_at_threshold, pos_label=1, zero_division=0
        )
        macro_f1 = f1_score(
            all_val_labels, y_pred_at_threshold, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            all_val_labels, y_pred_at_threshold, average="weighted", zero_division=0
        )
        auc_score = roc_auc_score(all_val_labels, all_val_probas)
        report = classification_report(
            all_val_labels,
            y_pred_at_threshold,
            target_names=["Not Fraud", "Fraud"],
            output_dict=True,
            zero_division=0,
        )

        # Log metrics with consistent names
        self.log(
            "val_f1_at_optimized_threshold",  # Renamed
            f1_final,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_best_threshold",
            best_val_threshold,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_precision_fraud",
            precision,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_recall_fraud",
            recall,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_macro_f1",
            macro_f1,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_weighted_f1",
            weighted_f1,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_auc_score",
            auc_score,
            on_epoch=True,
            prog_bar=True,  # Set to True for AUC
            logger=True,
            sync_dist=True,
        )

        # Create standardized metrics dictionary for JSON saving
        epoch_metrics = {
            "epoch": current_epoch,
            "accuracy": float(accuracy),
            "precision_fraud": float(precision),
            "recall_fraud": float(recall),
            "f1_fraud_at_optimized_threshold": float(f1_final),  # Renamed
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "auc_score": float(auc_score),
            "best_threshold": float(best_val_threshold),
            "report": report,
            "num_eval_samples": len(all_val_labels),
            "num_fraud_samples": int(all_val_labels.sum()),
        }

        # Save all epoch metrics to JSON
        if self.trainer and self.trainer.logger and self.trainer.logger.log_dir:
            metrics_path = log_dir_path.parent / f"metrics_epoch_{current_epoch}.json"

            # Update with trainer's callback metrics if any are missing
            for k, v in self.trainer.callback_metrics.items():
                # Only add if not already there and if not a torchmetric that's already processed
                if k not in epoch_metrics and not k.startswith("val_"):
                    epoch_metrics[k] = v.item() if isinstance(v, torch.Tensor) else v

            epoch_metrics.update({"step": self.global_step})
            try:
                with open(metrics_path, "w") as f:
                    json.dump(numpy_to_scalar(epoch_metrics), f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save epoch metrics to {metrics_path}: {e}")
        else:
            logging.warning(
                "Could not save epoch metrics: Trainer logger log_dir not found."
            )

    def on_test_epoch_end(
        self,
    ):  # This is for PL trainer.test(), final metrics are usually calculated outside
        self._log_and_reset_torchmetrics_epoch_end(stage="test")

    def configure_optimizers(self):
        return optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        financial_features_grouped = batch["financial_features_grouped"]
        sbert_embedding, input_ids, attention_mask = None, None, None
        use_precomputed_batch = batch["use_precomputed"][0].item()
        if use_precomputed_batch:
            sbert_embedding = batch["sbert_embedding"]
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        logits, _, _ = self.model(
            financial_features_grouped, input_ids, attention_mask, sbert_embedding
        )
        preds_proba = torch.sigmoid(logits)
        # Return only probabilities; thresholding happens in RCMAClassifier.evaluate
        return preds_proba


def find_best_auc_threshold(
    y_true, y_scores, log_dir, prefix="", min_threshold=0.05, max_threshold=0.95
):
    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get all possible thresholds from the scores
    all_possible_thresholds = np.sort(np.unique(y_scores))
    # Filter thresholds within the specified range
    valid_thresholds = all_possible_thresholds[
        (all_possible_thresholds >= min_threshold)
        & (all_possible_thresholds <= max_threshold)
    ]

    if len(valid_thresholds) == 0:  # If no thresholds in range
        # Fallback: Use default threshold
        logging.warning(
            f"No thresholds available for AUC optimization for {prefix}. Using default 0.5."
        )
        return 0.5, roc_auc_score(y_true, y_scores)

    # Calculate AUC for each threshold
    auc_scores = []
    for threshold in valid_thresholds:
        y_pred_binary = (y_scores >= threshold).astype(int)
        try:
            # Note: AUC is calculated on the binary predictions vs true labels
            # This actually gives a different metric - we're measuring how well the thresholded
            # predictions align with the true labels in terms of ranking ability
            current_auc = roc_auc_score(y_true, y_pred_binary)
            auc_scores.append(current_auc)
        except ValueError:
            # This can happen if predictions are all the same class
            auc_scores.append(0.0)

    auc_scores = np.array(auc_scores)

    # Visualize thresholds vs AUC
    fig = plt.figure(figsize=(10, 6))
    plt.plot(valid_thresholds, auc_scores, marker="o", linestyle="-", color="b")
    plt.title(f"AUC Score vs. Threshold {prefix}")
    plt.xlabel("Threshold")
    plt.ylabel("AUC Score")
    plt.grid()
    plt.savefig(log_dir / f"{prefix}auc_vs_threshold.png")
    plt.close(fig)

    # Also plot precision-recall curve for reference
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker=".", linestyle="-", color="r")
    plt.title(f"Precision-Recall Curve {prefix}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.savefig(log_dir / f"{prefix}precision_recall_curve.png")
    plt.close(fig)

    # Find the best threshold that maximizes AUC
    best_idx = np.argmax(auc_scores)
    best_auc = auc_scores[best_idx]
    best_threshold = valid_thresholds[best_idx]

    best_threshold = np.clip(
        best_threshold, 1e-6, 1 - 1e-6
    )  # Clip to avoid exact 0 or 1
    logging.info(
        f"{prefix}Optimized threshold: {best_threshold:.4f} -> AUC Score: {best_auc:.4f}"
    )

    return best_threshold, best_auc


def find_optimal_roc_threshold(y_true, y_scores, log_dir, prefix=""):
    """
    Finds the optimal threshold on the ROC curve using Youden's J statistic.
    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR.
    """
    if not isinstance(log_dir, Path):
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Ensure optimal_threshold is within reasonable bounds
    optimal_threshold = np.clip(optimal_threshold, 1e-6, 1 - 1e-6)

    logging.info(
        f"{prefix}Optimal ROC threshold (Youden's J): {optimal_threshold:.4f} "
        f"at TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}"
    )

    # Plot ROC curve with optimal point
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker=".", linestyle="-", color="b", label="ROC Curve")
    plt.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        marker="o",
        color="red",
        s=100,
        label=f"Optimal Threshold: {optimal_threshold:.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.title(f"ROC Curve with Optimal Threshold {prefix}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.legend()
    plt.savefig(log_dir / f"{prefix}roc_curve.png")
    plt.close(fig)

    return optimal_threshold


class RCMAClassifier:
    def __init__(
        self,
        model_alias_prefix="RCMA",
        sbert_model_name="all-MiniLM-L6-v2",
        sbert_output_dim=384,
        trainable_sbert_layers=0,
        num_original_financial_features=len(EXTENDED_FINANCIAL_FEATURES),
        num_financial_groups=7,
        financial_embedding_dim=64,
        text_embedding_dim=64,
        mlp_hidden_dims_paper=[128, 64],
        dropout_rate=0.3,
        learning_rate=1e-4,
        batch_size=32,
        val_batch_size=64,
        epochs=50,
        patience=100,
        # decision_threshold removed
        pos_weight_beta=0.25,
        focal_gamma=2.0,
        consistency_loss_weight=0.1,
        init_log_dir=True,
        oversample=False,
        dataset_version="company_isolated_splitting",
        fold_id=1,
        experiments_dir=None,
        max_mda_length=MAX_MDA_LENGTH,
        embedding_prefix_for_sbert="",
        **kwargs,  # To catch any stray decision_threshold from old configs
    ):
        self.model_alias = (
            f"{model_alias_prefix}_{sbert_model_name.split('/')[-1].replace('-', '_')}"
        )
        self.sbert_model_name = sbert_model_name
        self.sbert_output_dim = sbert_output_dim
        self.trainable_sbert_layers = trainable_sbert_layers
        self.num_original_financial_features = num_original_financial_features
        self.num_financial_groups = num_financial_groups
        self.financial_embedding_dim = financial_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        assert (
            self.financial_embedding_dim == self.text_embedding_dim
        ), "Mf and Mt must have same dim for fusion."
        self.mlp_hidden_dims_paper = mlp_hidden_dims_paper
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.patience = patience
        # self.decision_threshold removed
        self.pos_weight_beta = pos_weight_beta
        self.focal_gamma = focal_gamma
        self.consistency_loss_weight = consistency_loss_weight
        self.max_mda_length = max_mda_length
        self.embedding_prefix_for_sbert = embedding_prefix_for_sbert
        self.oversample = oversample
        self.dataset_version = dataset_version
        self.fold_id = fold_id
        self.base_experiments_dir = experiments_dir or Path(RCMA_EXPERIMENTS_DIR)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.accelerator = "gpu" if self.device_type == "cuda" else "cpu"
        self.devices = "auto" if self.device_type == "cuda" else "auto"

        self.lightning_model = None
        self.trainer = None
        self.log_dir = None
        self.tensorboard_log_dir = None
        self.best_model_path = None
        self.best_eval_threshold = 0.5  # Default, will be updated
        self.gradient_accumulation_steps =  kwargs.get("gradient_accumulation_steps", 1)
        self.tokenizer_for_trainable_sbert = (
            AutoTokenizer.from_pretrained(sbert_model_name, trust_remote_code=True)
            if trainable_sbert_layers > 0
            else None
        )
        if init_log_dir:
            self._setup_experiment_dir()
        logging.info(
            f"RCMAClassifier init. SBERT: {sbert_model_name}, OrigFinFeats: {num_original_financial_features}, FinGroups: {num_financial_groups}"
        )
        if "decision_threshold" in kwargs:
            logging.warning(
                "Ignoring 'decision_threshold' passed via kwargs as it's now dynamically optimized."
            )
        self.target_modules = kwargs.get(
            "target_modules",
            None)

    def _setup_experiment_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sampling_suffix = "_oversample" if self.oversample else ""
        safe_model_alias = self.model_alias.replace("/", "_").replace("\\", "_")
        experiment_name = (
            f"{safe_model_alias}_ds_{self.dataset_version}{sampling_suffix}"
            f"_hd_{'-'.join([str(x) for x in self.mlp_hidden_dims_paper])}"
        )
        self.log_dir = (
            self.base_experiments_dir
            / experiment_name
            / f"fold_{self.fold_id}"
            / timestamp
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "experiment.log"
        self.tensorboard_log_dir = (
            self.log_dir / "lightning_logs"
        )  # Changed from being a subdirectory of log_dir
        configure_logger(logFile=self.log_file, logLevel=logging.INFO)

        config_to_save = self.__dict__.copy()
        # Remove non-serializable or verbose attributes
        keys_to_remove = [
            "lightning_model",
            "trainer",
            "tokenizer_for_trainable_sbert",
            "base_experiments_dir",
            "log_file",
            "tensorboard_log_dir",
            "accelerator",
            "devices",  # trainer related
        ]
        for key_to_remove in keys_to_remove:
            config_to_save.pop(key_to_remove, None)

        # Ensure Path objects are converted to strings for YAML
        for key, value in config_to_save.items():
            if isinstance(value, Path):
                config_to_save[key] = str(value)

        with open(self.log_dir / "experiment_config.yaml", "w") as f:
            yaml.dump(numpy_to_scalar(config_to_save), f, indent=2, sort_keys=False)

    def precompute_sbert_embeddings(
        self, mda_texts, batch_size=32, dataset_type="unknown"
    ):
        # ... (no changes to this method)
        cache_dir_name = self.sbert_model_name.replace("/", "_").replace("\\", "_")
        cache_dir = Path(
            os.path.expanduser(f"~/.cache/mda_embeddings_rcma/{cache_dir_name}")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        texts_sample_for_hash = (
            "".join(mda_texts[: min(10, len(mda_texts))])
            + self.embedding_prefix_for_sbert
            + str(self.fold_id)
        )
        texts_hash = hashlib.md5(texts_sample_for_hash.encode()).hexdigest()[:10]
        cache_file_name = f"{self.dataset_version}_{dataset_type}_{texts_hash}_len{self.max_mda_length}.pt"
        cache_file = cache_dir / cache_file_name
        if cache_file.exists():
            try:
                logging.info(f"Loading SBERT embeddings from cache: {cache_file}")
                return torch.load(cache_file).float()
            except Exception as e:
                logging.warning(f"Cache load failed ({cache_file}): {e}. Recomputing.")
        sbert_inference_model = SentenceTransformer(
            self.sbert_model_name, device=self.device_type, trust_remote_code=True
        )
        sbert_inference_model.max_seq_length = self.max_mda_length
        all_embeddings = (
            sbert_inference_model.encode(
                mda_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device_type,
                prompt=self.embedding_prefix_for_sbert or None,
            )
            .cpu()
            .float()
        )
        try:
            torch.save(all_embeddings, cache_file)
            logging.info(f"Saved SBERT embeddings to cache: {cache_file}")
        except Exception as e:
            logging.warning(
                f"Failed to save SBERT embeddings to cache ({cache_file}): {e}"
            )
        return all_embeddings

    def load_data(
        self, train_path_str=None, test_path_str=None, full_financial_path_str=None
    ):
        # ... (no changes to this method, assuming it works as intended)
        dataset_splits = load_rcma_dataset(
            dataset_version=self.dataset_version,
            train_path=Path(train_path_str) if train_path_str else None,
            test_path=Path(test_path_str) if test_path_str else None,
            full_financial_path=Path(full_financial_path_str)
            if full_financial_path_str
            else None,
            fold_id=self.fold_id,
        )
        if dataset_splits["train"]["x_num"].ndim == 3:
            self.num_original_financial_features = dataset_splits["train"][
                "x_num"
            ].shape[1]
        else:
            self.num_original_financial_features = len(EXTENDED_FINANCIAL_FEATURES)
            logging.error(
                f"Train x_num is not 3D as expected! Shape: {dataset_splits['train']['x_num'].shape}. "
                f"Defaulting num_original_financial_features to {self.num_original_financial_features}"
            )
        logging.info(
            f"Data loaded via load_rcma_dataset. Inferred num_original_financial_features: {self.num_original_financial_features}"
        )
        return dataset_splits

    def _oversample_data_with_embeddings(
        self, X_grouped_np, y_np, mda_list, embeds_tensor
    ):
        # ... (no changes to this method)
        logging.info("Oversampling with grouped financial data and embeddings...")
        fraud_indices_orig = np.where(y_np == 1)[0]
        non_fraud_indices_orig = np.where(y_np == 0)[0]
        if len(fraud_indices_orig) == 0:
            return X_grouped_np, y_np, mda_list, embeds_tensor
        n_non_fraud = len(non_fraud_indices_orig)
        if n_non_fraud == 0:  # No non-fraud cases to match
            logging.warning(
                "No non-fraud cases found for oversampling. Returning original data."
            )
            return X_grouped_np, y_np, mda_list, embeds_tensor

        oversampled_fraud_indices = np.random.choice(
            fraud_indices_orig, size=n_non_fraud, replace=True
        )
        X_fraud_os = X_grouped_np[oversampled_fraud_indices]
        y_fraud_os = y_np[oversampled_fraud_indices]
        mda_fraud_os = [mda_list[i] for i in oversampled_fraud_indices]
        embeds_fraud_os = embeds_tensor[oversampled_fraud_indices]
        X_non_fraud = X_grouped_np[non_fraud_indices_orig]
        y_non_fraud = y_np[non_fraud_indices_orig]
        mda_non_fraud = [mda_list[i] for i in non_fraud_indices_orig]
        embeds_non_fraud = embeds_tensor[non_fraud_indices_orig]
        X_comb = np.concatenate([X_fraud_os, X_non_fraud], axis=0)
        y_comb = np.concatenate([y_fraud_os, y_non_fraud], axis=0)
        mda_comb = mda_fraud_os + mda_non_fraud
        embeds_comb = torch.cat([embeds_fraud_os, embeds_non_fraud], dim=0)
        perm = np.random.permutation(len(X_comb))
        return (
            X_comb[perm],
            y_comb[perm],
            [mda_comb[i] for i in perm],
            embeds_comb[perm],
        )

    def fit(self, dataset_splits):
        # ... (data loading and oversampling logic mostly the same) ...
        train_data = dataset_splits["train"]
        val_data = dataset_splits["val"]
        X_train_grouped, y_train_np, mda_train = (
            train_data["x_num"],
            train_data["y"].values,
            train_data["x_mda"],
        )
        X_val_grouped, y_val_np, mda_val = (
            val_data["x_num"],
            val_data["y"].values,
            val_data["x_mda"],
        )

        pre_train_embeds, pre_val_embeds = None, None
        if self.trainable_sbert_layers == 0:
            if mda_train:
                pre_train_embeds = self.precompute_sbert_embeddings(
                    mda_train, self.batch_size, "train"
                )
            if mda_val:
                pre_val_embeds = self.precompute_sbert_embeddings(
                    mda_val, self.val_batch_size, "val"
                )

        X_train_f, y_train_f, mda_train_f, train_embeds_f = (
            X_train_grouped,
            y_train_np,
            mda_train,
            pre_train_embeds,
        )
        if self.oversample:
            if self.trainable_sbert_layers == 0 and pre_train_embeds is not None:
                (
                    X_train_f,
                    y_train_f,
                    mda_train_f,
                    train_embeds_f,
                ) = self._oversample_data_with_embeddings(
                    X_train_grouped, y_train_np, mda_train, pre_train_embeds
                )
            else:  # Oversampling for trainable SBERT case
                logging.warning(
                    "Oversampling for trainable SBERT: applying to raw components before tokenization."
                )
                fraud_idx = np.where(y_train_np == 1)[0]
                non_fraud_idx = np.where(y_train_np == 0)[0]
                if len(fraud_idx) > 0 and len(non_fraud_idx) > 0:
                    oversampled_f_idx = np.random.choice(
                        fraud_idx, size=len(non_fraud_idx), replace=True
                    )
                    X_train_f = np.concatenate(
                        (
                            X_train_grouped[oversampled_f_idx],
                            X_train_grouped[non_fraud_idx],
                        ),
                        axis=0,
                    )
                    y_train_f = np.concatenate(
                        (y_train_np[oversampled_f_idx], y_train_np[non_fraud_idx]),
                        axis=0,
                    )
                    mda_train_f = [mda_train[i] for i in oversampled_f_idx] + [
                        mda_train[i] for i in non_fraud_idx
                    ]

                    perm = np.random.permutation(len(y_train_f))
                    X_train_f, y_train_f = X_train_f[perm], y_train_f[perm]
                    mda_train_f = [mda_train_f[i] for i in perm]
                    # train_embeds_f will be None here, as SBERT is trainable
                else:
                    logging.warning(
                        "Not enough samples in one class for oversampling when SBERT is trainable."
                    )

        train_ds = MDAFinancialDataset(
            torch.tensor(X_train_f, dtype=torch.float32),
            torch.tensor(y_train_f, dtype=torch.float32).unsqueeze(1),
            mda_train_f,
            self.tokenizer_for_trainable_sbert,
            self.max_mda_length,
            train_embeds_f,
        )
        val_ds = MDAFinancialDataset(
            torch.tensor(X_val_grouped, dtype=torch.float32),
            torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1),
            mda_val,
            self.tokenizer_for_trainable_sbert,
            self.max_mda_length,
            pre_val_embeds,
        )

        train_loader_workers = min(
            os.cpu_count() // 2 if os.cpu_count() else 0, 4
        )  # Limit workers
        val_loader_workers = min(os.cpu_count() // 2 if os.cpu_count() else 0, 2)

        train_loader = DataLoader(
            train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=train_loader_workers,
            persistent_workers=train_loader_workers > 0,
            pin_memory=True if self.device_type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_ds,
            self.val_batch_size,
            num_workers=val_loader_workers,
            persistent_workers=val_loader_workers > 0,
            pin_memory=True if self.device_type == "cuda" else False,
        )

        self.lightning_model = LightningRCMAClassifier(
            num_original_financial_features=self.num_original_financial_features,
            num_financial_groups=self.num_financial_groups,
            sbert_model_name=self.sbert_model_name,
            sbert_output_dim=self.sbert_output_dim,
            trainable_sbert_layers=self.trainable_sbert_layers,
            financial_embedding_dim=self.financial_embedding_dim,
            text_embedding_dim=self.text_embedding_dim,
            mlp_hidden_dims_paper=self.mlp_hidden_dims_paper,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            pos_weight_beta=self.pos_weight_beta,
            focal_gamma=self.focal_gamma,
            consistency_loss_weight=self.consistency_loss_weight,
            max_mda_length_for_trainable_sbert=self.max_mda_length,
            target_modules=self.target_modules,
            # decision_threshold removed from args
        )
        from pytorch_lightning.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            RichProgressBar,
        )
        from pytorch_lightning.loggers import TensorBoardLogger

        callbacks = [
            EarlyStopping(
                monitor="val_auc_score",  # Changed monitor to AUC
                patience=self.patience,
                mode="max",
                verbose=True,
            ),
            ModelCheckpoint(
                monitor="val_auc_score",  # Changed monitor to AUC
                dirpath=self.log_dir / "checkpoints",
                filename=f"{self.model_alias}-{{epoch:02d}}-{{val_auc_score:.4f}}",  # Filename reflects AUC
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
            RichProgressBar(),
        ]
        tb_logger = TensorBoardLogger(
            save_dir=str(self.log_dir), name="lightning_logs", version=""
        )  # Ensure logs are in self.log_dir/lightning_logs

        self.trainer = pl.Trainer(
            logger=tb_logger,
            callbacks=callbacks,
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            precision="bf16-true",
            log_every_n_steps=max(
                1, len(train_loader) // 10 if len(train_loader) > 0 else 1
            ),  # Added check for len(train_loader)
            num_sanity_val_steps=0,
            accumulate_grad_batches= self.gradient_accumulation_steps,
        )
        self.trainer.fit(self.lightning_model, train_loader, val_loader)
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path

        if self.best_model_path and Path(self.best_model_path).exists():
            logging.info(f"Loading best model from: {self.best_model_path}")
            self.lightning_model = LightningRCMAClassifier.load_from_checkpoint(
                self.best_model_path,
                # Map location needed if training on GPU and loading on CPU or vice-versa for inference without trainer
                map_location=torch.device(self.device_type),
            )
            # Determine the best threshold from this checkpoint's validation phase
            self.best_eval_threshold = self._get_threshold_for_best_checkpoint()
        else:
            logging.warning(
                "No best model path found after training. Using last model state and default threshold."
            )

        return self.trainer.callback_metrics

    def _get_threshold_for_best_checkpoint(self):
        if not self.best_model_path or not Path(self.best_model_path).exists():
            logging.warning(
                "Best model path not set or invalid. Cannot determine best threshold. Using default 0.5."
            )
            return 0.5
        try:
            filename = Path(self.best_model_path).name
            epoch_str = filename.split("-")[1].split("=")[
                1
            ]  # Assumes "model-epoch=EE-metric=VV.ckpt"
            best_epoch = int(epoch_str)

            metric_file = (
                self.log_dir / f"metrics_epoch_{best_epoch}.json"
            )  # Corrected path to parent/metrics_epoch.json
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    metrics = json.load(f)
                # Changed: Retrieve best_threshold and auc_score from JSON
                threshold = metrics["best_threshold"]
                val_auc_opt = metrics["auc_score"]
                logging.info(
                    f"Using threshold {threshold:.4f} from best epoch {best_epoch} (Val AUC Score: {val_auc_opt:.4f}) for test evaluation."
                )
                return threshold
            else:
                logging.warning(
                    f"Metric file {metric_file} for best epoch {best_epoch} not found. Trying to find overall best from logs."
                )
                # Fallback: iterate all metric files to find the one with highest val_auc_score
                return self._find_overall_best_threshold_from_val_logs()
        except Exception as e:
            logging.error(
                f"Error determining threshold from best checkpoint ({self.best_model_path}): {e}. Using fallback."
            )
            return self._find_overall_best_threshold_from_val_logs()

    def _find_overall_best_threshold_from_val_logs(self):
        # Helper to find the threshold associated with the globally best val_auc_score across all epochs
        best_auc_overall = -1.0
        threshold_for_overall_best_auc = 0.5  # Default

        if not self.log_dir or not self.log_dir.exists():
            logging.warning(
                "Log directory not found. Cannot determine overall best threshold."
            )
            return threshold_for_overall_best_auc  # Return default

        # Adjust path to find metric files in the parent directory of the specific run's log_dir
        metric_files_parent_dir = self.log_dir.parent
        metric_files = list(metric_files_parent_dir.glob("metrics_epoch_*.json"))

        if not metric_files:
            logging.warning(
                "No epoch metric files found. Cannot determine overall best threshold."
            )
            return threshold_for_overall_best_auc  # Return default

        found_one = False
        for metric_file in metric_files:
            try:
                with open(metric_file, "r") as f:
                    metrics = json.load(f)
                current_auc = metrics["auc_score"]  # Changed to auc_score
                current_threshold = metrics["best_threshold"]

                if current_auc > best_auc_overall:
                    best_auc_overall = current_auc
                    threshold_for_overall_best_auc = current_threshold
                    found_one = True
            except Exception as e:
                logging.warning(f"Skipping metric file {metric_file} due to error: {e}")
                continue

        if found_one:
            logging.info(
                f"Overall best val_auc_score found: {best_auc_overall:.4f} with threshold: {threshold_for_overall_best_auc:.4f}"
            )
        else:
            logging.warning(
                "Could not find any valid val_auc_score in metric files. Defaulting threshold to 0.5."
            )
        return threshold_for_overall_best_auc

    def evaluate(self, X_test_grouped, y_test_np, mda_test, subset="Test"):
        if not self.lightning_model:
            if self.best_model_path and Path(self.best_model_path).exists():
                logging.info(
                    f"Loading model from {self.best_model_path} for evaluation."
                )
                self.lightning_model = LightningRCMAClassifier.load_from_checkpoint(
                    self.best_model_path, map_location=torch.device(self.device_type)
                )
                # Determine the threshold for this loaded best model
                self.best_eval_threshold = self._get_threshold_for_best_checkpoint()
            else:
                raise ValueError(
                    "Model not trained or loaded, and no best_model_path found."
                )

        # Ensure trainer is available for predictions if not already set up for test
        if self.trainer is None:
            self.trainer = pl.Trainer(
                accelerator=self.accelerator, devices=self.devices, logger=False
            )  # Minimal trainer for .predict

        self.lightning_model.to(self.device_type)
        self.lightning_model.eval()

        pre_test_embeds = None
        if self.trainable_sbert_layers == 0 and mda_test:
            pre_test_embeds = self.precompute_sbert_embeddings(
                mda_test, self.val_batch_size, dataset_type=subset.lower()
            )

        test_ds = MDAFinancialDataset(
            torch.tensor(X_test_grouped, dtype=torch.float32),
            torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1),
            mda_test,
            self.tokenizer_for_trainable_sbert,
            self.max_mda_length,
            pre_test_embeds,
        )
        test_loader = DataLoader(
            test_ds,
            self.val_batch_size,
            num_workers=min(os.cpu_count() // 2 if os.cpu_count() else 0, 2),
        )

        # Get probabilities
        predictions_outputs = self.trainer.predict(
            self.lightning_model, dataloaders=test_loader
        )
        preds_probas_all = (
            torch.cat([p for p in predictions_outputs]).squeeze().cpu().float().numpy()
        )

        # Use the determined best_eval_threshold
        eval_threshold_to_use = self.best_eval_threshold
        logging.info(
            f"Using threshold for {subset} evaluation: {eval_threshold_to_use:.4f}"
        )

        y_pred_test_binary = (preds_probas_all >= eval_threshold_to_use).astype(int)
        y_true_test = y_test_np.astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_true_test, y_pred_test_binary)
        precision = precision_score(
            y_true_test, y_pred_test_binary, pos_label=1, zero_division=0
        )
        recall = recall_score(
            y_true_test, y_pred_test_binary, pos_label=1, zero_division=0
        )
        f1 = f1_score(y_true_test, y_pred_test_binary, pos_label=1, zero_division=0)
        macro_f1 = f1_score(
            y_true_test, y_pred_test_binary, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true_test, y_pred_test_binary, average="weighted", zero_division=0
        )
        auc_score = roc_auc_score(y_true_test, preds_probas_all)
        report = classification_report(
            y_true_test,
            y_pred_test_binary,
            target_names=["Not Fraud", "Fraud"],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true_test, y_pred_test_binary)

        logging.info(
            f"--- {subset} Set Evaluation Results (Threshold: {eval_threshold_to_use:.4f}) ---"
        )
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision (Fraud): {precision:.4f}")
        logging.info(f"Recall (Fraud): {recall:.4f}")
        logging.info(f"F1 Score (Fraud): {f1:.4f}")
        logging.info(f"Macro F1 Score: {macro_f1:.4f}")
        logging.info(f"Weighted F1 Score: {weighted_f1:.4f}")
        logging.info(f"AUC Score: {auc_score:.4f}")
        logging.info(
            f"Classification Report:\n{classification_report(y_true_test, y_pred_test_binary, target_names=['Not Fraud', 'Fraud'], zero_division=0)}"
        )
        logging.info(f"Confusion Matrix:\n{cm}")

        # Create standardized metrics dictionary to match LLM classifier format
        metrics = {
            "threshold_used": float(eval_threshold_to_use),
            "accuracy": float(accuracy),
            "precision_fraud": float(precision),
            "recall_fraud": float(recall),
            "f1_fraud": float(f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "auc_score": float(auc_score),
            "ap": float(
                average_precision_score(y_true_test, preds_probas_all)
            ),  # Keep original AP metric
            "report": report,
            "confusion_matrix": cm.tolist(),  # Keep original confusion matrix
            "num_test_samples": len(y_true_test),
            "num_fraud_samples_test": int(y_true_test.sum()),
        }

        # Save detailed predictions with standardized column names
        pred_df = pd.DataFrame(
            {
                "y_true_id": y_true_test,
                "fraud_probability": preds_probas_all,
                "y_pred_id": y_pred_test_binary,
            }
        )
        pred_df.to_csv(self.log_dir / f"{subset.lower()}_predictions.csv", index=False)

        return metrics

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.log_dir / f"{self.model_alias}_final_model.pt"
        else:
            filepath = Path(filepath)

        if self.lightning_model:
            # Save the PyTorch Lightning model state (includes hyperparameters)
            # This is different from just saving state_dict. For PL, saving checkpoint is better.
            # The best checkpoint is already saved by ModelCheckpoint.
            # This method can be used to save the *currently loaded* model state if needed explicitly.
            if self.trainer:
                self.trainer.save_checkpoint(filepath)
                logging.info(
                    f"Saved current Lightning model state to {filepath} via trainer.save_checkpoint."
                )
            else:  # Fallback if trainer not available (e.g. after loading)
                torch.save(
                    self.lightning_model.state_dict(), filepath.with_suffix(".pth")
                )  # Save only state_dict
                logging.info(
                    f"Saved model state_dict to {filepath.with_suffix('.pth')}"
                )
        else:
            logging.warning("No model to save.")

    def load_model(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Try loading as a PyTorch Lightning Checkpoint first
        try:
            self.lightning_model = LightningRCMAClassifier.load_from_checkpoint(
                str(filepath), map_location=torch.device(self.device_type)
            )
            logging.info(f"Loaded model from PyTorch Lightning checkpoint: {filepath}")
            self.best_model_path = str(
                filepath
            )  # Update best_model_path if loaded externally
            self.best_eval_threshold = (
                self._get_threshold_for_best_checkpoint()
            )  # Re-evaluate threshold
        except Exception as e_pl:
            logging.warning(
                f"Failed to load as PL checkpoint ({e_pl}), trying as raw state_dict."
            )
            # Fallback to loading as a raw state_dict (if saved that way)
            # This requires model architecture to be initialized first.
            # For this to work, self.lightning_model should be an instance of LightningRCMAClassifier.
            # This part is tricky if the model wasn't initialized with all hparams.
            # Best practice is to always save/load via PL checkpoints.
            # If loading a raw .pth, ensure the model instance is properly created first.
            # For now, prioritize PL checkpoints.
            raise e_pl  # Re-raise if PL checkpoint load fails, as it's the preferred method

    def save_experiment_results(self, val_metrics_at_fit_end, test_metrics):
        output_path = self.log_dir / "experiment_summary.yaml"
        hparams_dict = self.lightning_model.hparams if self.lightning_model else {}

        best_epoch_num = -1
        best_val_auc_score = -1.0  # Changed from f1
        # Try to get epoch from best_model_path
        if self.best_model_path and Path(self.best_model_path).exists():
            try:
                filename = Path(self.best_model_path).name
                epoch_str = filename.split("-")[1].split("=")[1]
                best_epoch_num = int(epoch_str)
                # Also get the auc score from filename if possible
                auc_str = (
                    filename.split("-")[2].split("=")[1].replace(".ckpt", "")
                )  # Changed from f1_str
                best_val_auc_score = float(auc_str)  # Changed from f1
            except Exception:
                logging.warning(
                    "Could not parse epoch/auc from best_model_path filename for summary."
                )

        # Get metrics from the best epoch's JSON file for more reliability
        metrics_from_best_epoch_json = {}
        if best_epoch_num != -1:
            metric_file = self.log_dir.parent / f"metrics_epoch_{best_epoch_num}.json"
            if metric_file.exists():
                with open(metric_file, "r") as f:
                    metrics_from_best_epoch_json = json.load(f)
            if (
                "auc_score"
                in metrics_from_best_epoch_json  # Changed from val_f1_optimized
                and best_val_auc_score == -1.0
            ):  # if not parsed from filename
                best_val_auc_score = metrics_from_best_epoch_json[
                    "auc_score"
                ]  # Changed from val_f1_optimized

        experiment_dict = {
            "model_alias": self.model_alias,
            "log_dir": str(self.log_dir),
            "dataset_version": self.dataset_version,
            "fold_id": self.fold_id,
            "hyperparameters_model_init": numpy_to_scalar(
                {
                    k: v
                    for k, v in self.__dict__.items()
                    if isinstance(v, (str, int, float, bool, list, dict))
                    and k not in ["lightning_model", "trainer"]
                }
            ),  # Selected init hparams
            "hyperparameters_lightning_module": numpy_to_scalar(dict(hparams_dict)),
            "best_epoch_num_from_checkpoint": best_epoch_num,
            "best_val_auc_score_from_checkpoint": best_val_auc_score,  # Changed from f1
            "best_val_metrics_from_json_log": torch_to_scalar(
                metrics_from_best_epoch_json
            ),
            "test_metrics_final": torch_to_scalar(test_metrics),
            "num_original_financial_features_used": self.num_original_financial_features,
            "best_model_checkpoint_path": str(self.best_model_path),
            "final_evaluation_threshold_used_for_test": self.best_eval_threshold,
        }

        # Save the test_metrics json
        test_metrics_json_path = self.log_dir / f"test_metrics.json"
        with open(test_metrics_json_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logging.info(f"Saved test metrics to {test_metrics_json_path}")

        with open(output_path, "w") as f:
            yaml.dump(experiment_dict, f, indent=2, sort_keys=False)
        logging.info(f"Saved experiment summary to {output_path}")


def train_and_evaluate_rcma_model(config):
    wrapper = RCMAClassifier(**config)  # decision_threshold removed from direct pass
    pl.seed_everything(SEED_TRAINING, workers=True)  # Ensure seed is set
    train_path_str, test_path_str = load_cross_validation_path(config)

    dataset_splits = wrapper.load_data(
        train_path_str=str(train_path_str), test_path_str=str(test_path_str)
    )

    val_metrics_fit_end = wrapper.fit(
        dataset_splits
    )  # This will now use optimized AUC for checkpointing

    test_data = dataset_splits["test"]
    # The evaluate method now uses the threshold from the best validation epoch
    test_metrics_final = wrapper.evaluate(
        test_data["x_num"], test_data["y"].values, test_data["x_mda"], subset="Test"
    )

    wrapper.save_experiment_results(val_metrics_fit_end, test_metrics_final)
    # wrapper.save_model() # Best model is already saved by ModelCheckpoint. This could save final state if needed.
    return wrapper, test_metrics_final


if __name__ == "__main__":

    rcma_config = {
        "sbert_model_name": "nomic-ai/nomic-embed-text-v1",  # "all-MiniLM-L6-v2",
        "sbert_output_dim": 768,  # 384 for MiniLM
        "trainable_sbert_layers": 0,  # Set to >0 to fine-tune SBERT layers
        "num_original_financial_features": len(
            EXTENDED_FINANCIAL_FEATURES
        ),  # Auto-detected if possible
        "num_financial_groups": 7,
        "financial_embedding_dim": 64,
        "text_embedding_dim": 64,
        "mlp_hidden_dims_paper": [512, 256],
        "dropout_rate": 0.3,
        "learning_rate": 1e-5,
        "batch_size": 64,
        "val_batch_size": 128,
        "epochs": 200,  # Increased for actual run
        "patience": 100,
        # "decision_threshold": 0.5, # Removed
        "pos_weight_beta": 0.75,  # For FocalLoss
        "focal_gamma": 2.0,  # For FocalLoss
        "consistency_loss_weight": 0.05,
        "oversample": False,  # Set to True to oversample fraud cases in training
        "dataset_version": "company_isolated_splitting",
        "fold_id": 1,
        "max_mda_length": 8192,  # Max MDA sequence length
        "embedding_prefix_for_sbert": "classification:",  # e.g. "search_document: " or "search_query: " for some models
    }
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    Path(RCMA_EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)

    # Example of running:
    train_and_evaluate_rcma_model(rcma_config)
    print("RCMA Model training and evaluation example finished.")
