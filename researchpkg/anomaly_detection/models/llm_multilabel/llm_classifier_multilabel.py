import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    GPTQConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
NOT_FRAUD_TOKEN = "NO"
FRAUD_TOKEN = "YES"

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    LIST_MISTATEMENT_TYPE_FOR_TRAINING,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.utils import (
    get_last_checkpoint,
    get_train_test_splitter,
)
from researchpkg.utils import configure_logger

LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]


def find_best_threshold_multilabel(
    y_true, y_scores, class_name, epoch, log_dir
) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1s = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    if len(f1s) == 0:
        return 0.5, f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)

    if plt and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            range = min(len(thresholds), len(f1s))
            thresholds = thresholds[:range]
            f1s = f1s[:range]
            ax.plot(thresholds, f1s, "b-o")
            ax.set(
                title=f"F1 Score vs. Threshold for {class_name} (Epoch {epoch})",
                xlabel="Threshold",
                ylabel="F1 Score",
            )
            ax.grid(True)
            fig.savefig(log_dir / f"f1_vs_threshold_epoch_{epoch}_{class_name}.png")
            plt.close(fig)
        except Exception as e:
            logging.error(f"Plotting error for {class_name}: {e}")

    best_idx = np.argmax(f1s)
    return float(np.clip(thresholds[best_idx], 1e-6, 1 - 1e-6)), float(f1s[best_idx])


class DatasetType(str, Enum):
    TRAIN, VALIDATION, TEST = "TRAIN", "VALIDATION", "TEST"


class MultiLabelTextDataset(TorchDataset):
    def __init__(
        self, df, type, tokenizer, text_col, label_names, max_length, undersample=False
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_names = label_names
        self.max_length = max_length
        self.undersample = undersample and type == DatasetType.TRAIN

        self.texts_full = df[text_col].tolist()
        self.labels_full = df[label_names].astype(int).values
        self.fraud_indices = np.where(df["fraud"].astype(int) == 1)[0]
        self.non_fraud_indices = np.where(df["fraud"].astype(int) == 0)[0]

        self.texts, self.labels = self.texts_full, self.labels_full
        if (
            self.undersample
            and len(self.fraud_indices) > 0
            and len(self.non_fraud_indices) > 0
        ):
            self.permute()
            
        
    def oversample(self):
        """
        Oversample the minority class (fraud) by duplicating its instances.
        """
        
        sample_size = max(len(self.fraud_indices), len(self.non_fraud_indices))
        selected_fraud = np.random.choice(
            self.fraud_indices, size=sample_size, replace=True
        )
        indices = np.concatenate([selected_fraud, self.non_fraud_indices])
        np.random.shuffle(indices)
        self.texts = [self.texts_full[i] for i in indices]
        self.labels = self.labels_full[indices]
        logging.info(f"Training dataset oversampled with {len(self.texts)} examples.")
    

    def permute(self):
        sample_size = min(len(self.fraud_indices), len(self.non_fraud_indices))
        selected_non_fraud = np.random.choice(
            self.non_fraud_indices, size=sample_size, replace=False
        )
        indices = np.concatenate([self.fraud_indices, selected_non_fraud])
        np.random.shuffle(indices)
        self.texts = [self.texts_full[i] for i in indices]
        self.labels = self.labels_full[indices]
        logging.info(f"Training dataset permuted with {len(self.texts)} examples.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DataCollatorForMultiLabelSequenceClassification:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["labels"] = batch["labels"].to(torch.float)
        return batch


class PermutationCallback(TrainerCallback):
    """A lightweight callback to handle dataset permutation for undersampling."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataset = (
            kwargs.get("train_dataloader").dataset
            if "train_dataloader" in kwargs
            else None
        )
        if (
            train_dataset
            and hasattr(train_dataset, "permute")
            and getattr(train_dataset, "undersample", False)
        ):
            is_main_process = (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            )
            if is_main_process:
                train_dataset.permute()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()




class FocalLossMultiLabelTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.alpha is not None:
            alpha_tensor = torch.tensor(self.alpha).to(logits.device)
            alpha_tensor = alpha_tensor[labels == 1].mean() if labels.sum() > 0 else 1.0
        else:
            alpha_tensor = 1.0

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = alpha_tensor * (1 - pt) ** self.gamma * bce_loss

        loss = focal_loss.mean()
        return (loss, outputs) if return_outputs else loss


class LLMClassifierMultiLabel:
    def __init__(self, config: Dict[str, any]):
        assert config is not None
        self.config = config
        self.model_url: str = config["model_url"]
        self.model_alias: str = config["model_alias"]
        self.max_length: int = config["max_context"]
        self.is_binary = config.get("binary", False)

        if not self.is_binary:
            self.label_names: List[str] = self.get_class_labels()
        else:
            self.label_names = ["fraud"]

        self.n_labels: int = len(self.label_names)
        self.text_col: str = config.get("text_col", "text")
        self.per_device_train_batch_size: int = config["per_device_train_batch_size"]
        self.per_device_eval_batch_size: int = config["per_device_eval_batch_size"]
        self.gradient_accumulation_steps: int = config["gradient_accumulation_steps"]
        self.num_train_epochs: int = config["num_train_epochs"]
        self.learning_rate: float = config["learning_rate"]
        self.fold_id: int = config["fold_id"]
        self.checkpoint_timestamp: Union[str, None] = config.get("checkpoint_timestamp")

        self.is_main_process = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self.log_dir = self._setup_logging_and_dirs()

        self.epoch_results = []
        self.best_auc_overall = -1.0
        self.best_epoch_based_on_auc = -1
        self.best_thresholds_at_best_epoch = {name: 0.5 for name in self.label_names}

        self.val_ds, self.model, self.tokenizer, self.trainer = None, None, None, None
        self.load_model_and_tokenizer()
        self.apply_lora()

    def get_class_labels(self) -> List[str]:
        return LIST_MISTATEMENT_TYPE_FOR_TRAINING

    def _setup_logging_and_dirs(self) -> Path:
        timestamp = self.checkpoint_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        experiments_dir = self.config["experiments_dir"]
        log_dir = (
            experiments_dir / self.model_alias / f"fold_{self.fold_id}" / timestamp
        )
        if self.checkpoint_timestamp:
            assert log_dir.exists(), (
                f"Checkpoint directory {log_dir} does not exist. "
                "Please ensure the checkpoint timestamp is correct."
            )

        if self.is_main_process:
            log_dir.mkdir(parents=True, exist_ok=True)
            configure_logger(logFile=log_dir / "experiment.log", logLevel=logging.INFO)
            logging.info(f"Logging to {log_dir}")
        else:
            logging.basicConfig(
                level=logging.WARNING,
                format="[%(levelname)s][Rank %(rank)s] %(message)s",
            )
        self.config["timestamp"] = str(timestamp)
        self.config["log_dir"] = str(log_dir)
        return log_dir

    @staticmethod
    def _calculate_metrics(
        logits,
        labels_true,
        label_names,
        log_dir,
        stage,
        thresholds=None,
        optimize=False,
    ):
        """Calculates all specified per-class and aggregate metrics."""
        preds_probs = torch.sigmoid(torch.tensor(logits)).numpy()
        used_thresholds = {}

        if optimize and thresholds is None:
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                logging.info(
                    f"Optimizing thresholds for each class (Stage: {stage})..."
                )
            for i, name in enumerate(label_names):
                thresh, _ = find_best_threshold_multilabel(
                    labels_true[:, i], preds_probs[:, i], name, stage, log_dir
                )
                used_thresholds[name] = thresh
        else:
            used_thresholds = thresholds or {name: 0.5 for name in label_names}

        thresh_arr = np.array([used_thresholds.get(name, 0.5) for name in label_names])
        pred_bin = (preds_probs >= thresh_arr).astype(int)

        metrics = {}
        per_class_acc, per_class_f1, per_class_prec, per_class_recall = [], [], [], []

        # --- Per-Class Metrics ---
        for i, name in enumerate(label_names):
            y_true, y_pred, y_prob = (
                labels_true[:, i],
                pred_bin[:, i],
                preds_probs[:, i],
            )

            # Use pos_label=1 to be explicit, though it's default for binary
            metrics[f"{name}_f1"] = f1_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
            metrics[f"{name}_precision"] = precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
            metrics[f"{name}_recall"] = recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )
            metrics[f"{name}_accuracy"] = accuracy_score(y_true, y_pred)
            metrics[f"{name}_auc"] = (
                roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            )
            metrics[f"{name}_threshold"] = used_thresholds.get(name, 0.5)

            per_class_f1.append(metrics[f"{name}_f1"])
            per_class_prec.append(metrics[f"{name}_precision"])
            per_class_recall.append(metrics[f"{name}_recall"])
            per_class_acc.append(metrics[f"{name}_accuracy"])

        # --- Aggregate Metrics ---
        # Macro Averages (unweighted mean)
        metrics["macro/f1"] = np.mean(per_class_f1)
        metrics["macro/precision"] = np.mean(per_class_prec)
        metrics["macro/recall"] = np.mean(per_class_recall)
        metrics["macro/accuracy"] = np.mean(per_class_acc)
        metrics["macro/auc"] = np.mean(
            [metrics[f"{name}_auc"] for name in label_names if f"{name}_auc" in metrics]
        )

        # Weighted Averages (by support/cardinality)
        metrics["weighted/f1"] = f1_score(
            labels_true, pred_bin, average="weighted", zero_division=0
        )
        metrics["weighted/precision"] = precision_score(
            labels_true, pred_bin, average="weighted", zero_division=0
        )
        metrics["weighted/recall"] = recall_score(
            labels_true, pred_bin, average="weighted", zero_division=0
        )
        metrics["weighted/auc"] = (
            roc_auc_score(labels_true, preds_probs, average="weighted")
            if np.any(labels_true.sum(axis=0) > 0)
            else 0.0
        )
        metrics["weighted/accuracy"] = accuracy_score(
            labels_true.flatten(), pred_bin.flatten()
        )  # This is the definition of weighted accuracy

        # Micro accuracy for completeness
        metrics["micro/accuracy"] = metrics["weighted/accuracy"]

        return metrics, used_thresholds

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        if not self.is_main_process:
            return {}

        # check if training_completed then skip evaluation
        current_epoch = int(self.trainer.state.epoch) if self.trainer.state.epoch else 0
        logging.info(f"--- Starting Evaluation for Epoch {current_epoch} ---")

        metrics, thresholds = self._calculate_metrics(
            eval_pred.predictions,
            eval_pred.label_ids,
            self.label_names,
            self.log_dir,
            current_epoch,
            optimize=True,
        )

        self._log_and_save_epoch_results(
            metrics,
            thresholds,
            eval_pred.label_ids,
            eval_pred.predictions,
            current_epoch,
        )

        return {k.replace("/", "_"): v for k, v in metrics.items() if "/" in k}

    def _log_and_save_epoch_results(self, metrics, thresholds, labels, logits, epoch):
        logging.info(f"--- Epoch {epoch} Results ---")
        logging.info(
            f"MACRO:   AUC={metrics['macro/auc']:.4f}, F1={metrics['macro/f1']:.4f}, Recall={metrics['macro/recall']:.4f}, Precision={metrics['macro/precision']:.4f}, Accuracy={metrics['macro/accuracy']:.4f}"
        )
        logging.info(
            f"WEIGHTED: AUC={metrics['weighted/auc']:.4f}, F1={metrics['weighted/f1']:.4f}, Recall={metrics['weighted/recall']:.4f}, Precision={metrics['weighted/precision']:.4f}, Accuracy={metrics['weighted/accuracy']:.4f}"
        )

        # Save predictions CSV
        preds_probs = torch.sigmoid(torch.tensor(logits)).numpy()
        thresh_arr = np.array([thresholds.get(n, 0.5) for n in self.label_names])
        pred_bin = (preds_probs >= thresh_arr).astype(int)

        df = (
            self.val_ds.df.copy()
            if hasattr(self.val_ds, "df") and len(self.val_ds.df) == len(labels)
            else pd.DataFrame(index=range(len(labels)))
        )
        for i, name in enumerate(self.label_names):
            df[f"y_true_{name}"], df[f"prob_{name}"], df[f"y_pred_opt_{name}"] = (
                labels[:, i],
                preds_probs[:, i],
                pred_bin[:, i],
            )
        # save_all_columns_except_text
        columns_to_save = [col for col in df.columns if col != self.text_col]
        df[columns_to_save].to_csv(
            self.log_dir / f"val_predictions_epoch_{epoch}.csv", index=False
        )
        logging.info(
            f"Saved predictions to {self.log_dir / f'val_predictions_epoch_{epoch}.csv'}"
        )

        if metrics["weighted/auc"] > self.best_auc_overall:
            self.best_auc_overall = metrics["weighted/auc"]
            self.best_epoch_based_on_auc = epoch
            self.best_thresholds_at_best_epoch = thresholds
            logging.info(
                f"*** New best model found! Epoch {epoch} with Weighted AUC: {self.best_auc_overall:.4f} ***"
            )
            with open(self.log_dir / "best_validation_thresholds.json", "w") as f:
                json.dump(thresholds, f, indent=2)

        epoch_results = {"epoch": epoch, "metrics": metrics, "thresholds": thresholds}  
        epoch_results = {
            "epoch": epoch,
            "n_samples": len(labels),
            "n_fraud": int(df["fraud"].sum()),
            "macro": {
                "auc": metrics["macro/auc"],
                "f1": metrics["macro/f1"],
                "recall": metrics["macro/recall"],
                "precision": metrics["macro/precision"],
                "accuracy": metrics["macro/accuracy"],
            },
            "weighted": {
                "auc": metrics["weighted/auc"],
                "f1": metrics["weighted/f1"],
                "recall": metrics["weighted/recall"],
                "precision": metrics["weighted/precision"],
                "accuracy": metrics["weighted/accuracy"],
            },
            "per_class": {
                name: {
                    "auc": metrics.get(f"{name}_auc", 0.0),
                    "f1": metrics.get(f"{name}_f1", 0.0),
                    "recall": metrics.get(f"{name}_recall", 0.0),
                    "precision": metrics.get(f"{name}_precision", 0.0),
                    "accuracy": metrics.get(f"{name}_accuracy", 0.0),
                    "threshold": thresholds.get(name, 0.5),
                    "n_fraud": int(df[name].sum() if name in df else 0),
                    "tp": int(
                        df[(df[name] == 1) & (df[f"y_pred_opt_{name}"] == 1)].shape[0]
                    ),
                    "fp": int(
                        df[(df[name] == 0) & (df[f"y_pred_opt_{name}"] == 1)].shape[0]
                    ),
                    "fn": int(
                        df[(df[name] == 1) & (df[f"y_pred_opt_{name}"] == 0)].shape[0]
                    ),
                    "tn": int(
                        df[(df[name] == 0) & (df[f"y_pred_opt_{name}"] == 0)].shape[0]
                    ),
                }
                for name in self.label_names
            },
        }
        with open(self.log_dir / f"metrics_epoch_{epoch}.json", "w") as f:
            json.dump(epoch_results, f, indent=2)

        tb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
        self.trainer.log(tb_metrics)


    def load_model_and_tokenizer(self):
        if self.is_main_process:
            logging.info(f"Loading {self.model_alias} from {self.model_url}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_url)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine YES token ID
        fraud_token_id = self.tokenizer.convert_tokens_to_ids(FRAUD_TOKEN)

        quant_config = None
        if self.config.get("load_in_4bit", True):
            quant_config = GPTQConfig(bits=4, disable_exllama=True)
            logging.info("Initializing with 4-bit quantization.")
            
            # from transformers import BitsAndBytesConfig
            # quant_config = BitsAndBytesConfig(
            #         load_in_4bit=True, 
            #         bnb_4bit_quant_type="nf4", 
            #         bnb_4bit_compute_dtype="float16", 
            #         bnb_4bit_use_double_quant=True
            #     )

        # Temporarily load a causal model to get the `lm_head` weights for initialization
        
        
        if self.config.get("initialize_classifier_with_yes", False):
            # Load a model that definitely has an lm_head to get the 'YES' token's logit vector
            temp_lm_model_for_weights = AutoModelForCausalLM.from_pretrained(
                self.model_url,
                torch_dtype=torch.bfloat16, # Match the dtype of the final model
                cache_dir=os.environ.get("HF_CACHE", None),
                local_files_q=self.config.get("offline", False),
            )
            temp_lm_model_for_weights.eval() # Put in eval mode, no gradients needed
            lm_head_weight = temp_lm_model_for_weights.lm_head.weight
            lm_head_weights_for_yes = lm_head_weight[fraud_token_id, :]
            
            
            
        # Load the actual AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_url,
            device_map={"": Accelerator().process_index},
            num_labels=self.n_labels,
            problem_type="multi_label_classification",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            cache_dir=os.environ.get("HF_CACHE", None),
            local_files_only=self.config.get("offline", False),
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id


        # Initialize the classifier head weights with the 'YES' token's LM head weights
        
        classifier_head = None
        if hasattr(self.model, 'classifier'): # Common for BERT-like and some Llama variants
            classifier_head = self.model.classifier
        elif hasattr(self.model, 'score'): # Common for some GPT-like models
            classifier_head = self.model.score
        
        
        if self.config.get("initialize_classifier_with_yes", False):
            with torch.no_grad():
                # Create a tensor where each row is the `lm_head_weights_for_yes` vector.
                # This aims to make each output label initially sensitive to patterns
                # that would lead to the model predicting "YES" in a causal LM context.
                new_classifier_weight = torch.stack([lm_head_weights_for_yes for _ in range(self.n_labels)])
                
                # Apply to the weight data of the classifier head.
                # .copy_() ensures the underlying data is replaced.
                classifier_head.weight.copy_(new_classifier_weight.to(classifier_head.weight.dtype).to(classifier_head.weight.device))
                
                logging.info(f"Classifier head weights successfully initialized with 'YES' token's LM head weights.")
                logging.info(f"{self.model_alias} model and tokenizer loaded.")

        
        
    def apply_lora(self):
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha= self.config["lora_alpha"],
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=self.config.get("lora_dropout", 0),
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        self.model = get_peft_model(self.model, config)
        # enable checkpointing
        # Disable cache to avoid incompatibility
        self.model.config.use_cache = False
        
        self.model.gradient_checkpointing_enable(dict(use_reentrant=False))
        if self.is_main_process:
            self.model.print_trainable_parameters()

    def load_data(self, train_path, test_path):
        #Only load data with the main process
        
        if self.is_main_process:
            train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
            if self.config.get("debug", False):
                frac = self.config.get("debug_fraction", 0.01)
                train_df, test_df = train_df.sample(
                    
                    frac=frac, random_state=SEED_TRAINING
                ), test_df.sample(frac=frac, random_state=SEED_TRAINING)
            
            train_df, test_df = self._process_data(train_df), self._process_data(test_df)

            if self.config["max_context"] <= 0:

                max_length_train = self.find_max_length(train_df)
                max_length_test = self.find_max_length(test_df)
                self.max_length = max(max_length_train, max_length_test)
                logging.info(f"Max length set to {self.max_length} based on training data.")

            train_df, val_df = get_train_test_splitter(self.config)(
                train_df, test_size=0.1, seed=SEED_TRAINING
            )
            
            if self.config["dataset_version"] == "company_isolated_splitting":
                #Ensure no CIK overlap
                assert not set(train_df["cik"]).intersection(set(test_df["cik"])), (
                    "CIK overlap found between train and test datasets. Please ensure CIKs are unique across splits."
                )
                assert not set(train_df["cik"]).intersection(set(val_df["cik"])), (
                    "CIK overlap found between train and validation datasets. Please ensure CIKs are unique across splits."
                )
                
            return train_df, val_df, test_df
        else:
            # If not main process, return empty DataFrames
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _process_data(self, df):
        for col in LIST_MISTATEMENT_TYPE_FOR_TRAINING:
            df[col] = df[col].fillna(0).astype(int)
        df["fraud"] = (df[LIST_MISTATEMENT_TYPE_FOR_TRAINING].sum(axis=1) > 0).astype(
            int
        )
        return df

    def train_and_evaluate(self, train_path, test_path):
        seed = self.config.get("seed", SEED_TRAINING)
        torch.manual_seed(seed), np.random.seed(seed), random.seed(seed)

        logging.info(
            f"Starting fine-tuning with HF Trainer for MultiLabel LLM Classifier"
        )
        logging.info("Number of labels: %d", self.n_labels)

        train_df, val_df, test_df = self.load_data(train_path, test_path)

        train_ds = MultiLabelTextDataset(
            train_df,
            DatasetType.TRAIN,
            self.tokenizer,
            self.text_col,
            self.label_names,
            self.max_length,
            self.config.get("undersample", False),
        )
        
        if self.config.get("oversample", False):
            logging.info("Oversampling the training dataset.")
            train_ds.oversample()
        
        self.val_ds = MultiLabelTextDataset(
            val_df,
            DatasetType.VALIDATION,
            self.tokenizer,
            self.text_col,
            self.label_names,
            self.max_length,
        )
        test_ds = MultiLabelTextDataset(
            test_df,
            DatasetType.TEST,
            self.tokenizer,
            self.text_col,
            self.label_names,
            self.max_length,
        )

        
        args = TrainingArguments(
            output_dir=str(self.log_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="eval_weighted_auc",
            greater_is_better=True,
            report_to="tensorboard",
            max_grad_norm=1.0,
            seed=seed,
            bf16=True,
            logging_steps=1,
        )

        use_focal_loss = self.config.get("use_focal_loss", False)
        
        if use_focal_loss:
            logging.info("Using Focal Loss for MultiLabel Classification")
            self.trainer = FocalLossMultiLabelTrainer(
                model=self.model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=self.val_ds,
                data_collator=DataCollatorForMultiLabelSequenceClassification(
                    self.tokenizer
                ),
                compute_metrics=self.compute_metrics,
            )
        else:
            self.trainer = Trainer(
                model=self.model, args=args, train_dataset=train_ds, eval_dataset=self.val_ds,
                data_collator=DataCollatorForMultiLabelSequenceClassification(self.tokenizer),
                compute_metrics=self.compute_metrics
            )

        self.trainer.add_callback(PermutationCallback())

        if self.config.get("early_stopping_patience"):
            self.trainer.add_callback(
                EarlyStoppingCallback(self.config["early_stopping_patience"])
            )
        if self.checkpoint_timestamp:
            last_checkpoint = get_last_checkpoint(self.log_dir)
            logging.info(f"Resuming training from last checkpoint: {last_checkpoint}")
            resume_from_checkpoint = str(last_checkpoint)
        else:
            resume_from_checkpoint = None

        exp_config_path = self.log_dir / "experiment_config.yaml"
        config_to_save = self.config.copy()
        config_to_save["experiments_dir"]= str(config_to_save["experiments_dir"])
        config_to_save["hf_args"] = args.to_dict()

        with open(exp_config_path, "w") as f:
            yaml.dump(config_to_save, f, default_flow_style=False)

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        self.load_best_model()
        if self.is_main_process:
            self.save_model(self.log_dir)
            metrics, df = self.evaluate(test_ds, self.best_thresholds_at_best_epoch)
            return self, metrics, df

        return self, {}, pd.DataFrame()

    def load_best_model(self):
        if self.is_main_process:
            logging.info("Loading best model based on validation AUC...")
        best_epoch, best_auc, best_checkpoint_name = self.find_best_checkpoint()
        if best_epoch == -1:
            logging.warning(
                "No valid best checkpoint found. Using the last saved model."
            )
            return self.model
        if best_checkpoint_name:
            best_checkpoint_path = self.log_dir / best_checkpoint_name
            if not best_checkpoint_path.exists():
                logging.warning(
                    f"Best checkpoint directory {best_checkpoint_path} does not exist. Using the last saved model."
                )
                return self.model
            logging.info(f"Loading model from {best_checkpoint_path}")

            # Directy load the state_dict of the model
            # adapters_path = best_checkpoint_path / "adapter_model.safetensors"
            self.model.load_adapter(
                best_checkpoint_path,
                adapter_name="default",
                device_map={"": Accelerator().process_index},
            )
            return self.model

    def find_best_checkpoint(self):
        """
        Find the best checkpoint based on the validation F1 score ('f1_fraud_optimized') saved by the callback.

        Returns:
            tuple: (best_epoch, best_f1, best_threshold, best_checkpoint_dir_name) or (-1, -1, 0.5, None) if not found.
        """
        # best_f1 = -1.0
        best_auc = -1.0
        best_epoch = -1
        best_checkpoint_name = None

        try:
            # 1. Find the epoch with the best F1 score from metrics files
            metric_files = list(self.log_dir.glob("metrics_epoch_*.json"))
            if not metric_files:
                logging.warning(
                    "No epoch metric files found. Cannot determine best checkpoint."
                )
                return -1, -1, None

            for metric_file in metric_files:
                with open(metric_file, "r") as f:
                    metrics = json.load(f)
                    epoch = metrics.get("epoch", -1)
                    # Prioritize the F1 score calculated during optimization
                    current_auc = metrics["weighted"]["auc"]

                    if current_auc > best_auc:
                        best_auc = current_auc
                        best_epoch = epoch

            if best_epoch == -1:
                logging.warning(
                    "Could not find a valid epoch with F1 score in metric files."
                )
                return -1, -1, 0.5, None

            logging.info(
                f"Best validation AUC ({best_auc:.4f}) found at epoch {best_epoch} (Threshold)."
            )

            # 2. Find the checkpoint directory corresponding to that best epoch
            #    Trainer saves checkpoints like 'checkpoint-XXXX'. We need trainer_state.json inside.
            checkpoint_dirs = [
                d
                for d in self.log_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]

            if not checkpoint_dirs:
                logging.warning("No checkpoint directories found.")
                return best_epoch, best_auc, None

            found_checkpoint_dir = None
            
            for chkpt_dir in checkpoint_dirs:
                state_file = chkpt_dir / "trainer_state.json"
                if state_file.exists():
                    with open(state_file, "r") as f:
                        state = json.load(f)
                        # Epoch in state is float, compare carefully
                        chkpt_epoch_float = state.get("epoch", -1.0)
                        if chkpt_epoch_float == best_epoch:
                            found_checkpoint_dir = chkpt_dir
                            break   


            if found_checkpoint_dir:
                best_checkpoint_name = found_checkpoint_dir.name
                logging.info(
                    f"Found checkpoint directory for best epoch ({best_epoch}): {best_checkpoint_name}"
                )
            else:
                logging.warning(
                    f"Could not find a matching checkpoint directory for the best epoch ({best_epoch}). The final model might be used instead."
                )
                # Optionally, fall back to the very last saved checkpoint?
                # best_checkpoint_name = get_last_checkpoint(self.log_dir) # Might return path or name

            return (
                best_epoch,
                best_auc,
                best_checkpoint_name,
            )

        except Exception as e:
            logging.error(f"Error finding best checkpoint: {e}")
            return -1, -1, None

    def evaluate(self, test_ds, thresholds):
        if not self.is_main_process:
            return {}, pd.DataFrame()

        logging.info(
            f"--- Final Test Set Evaluation using thresholds from Epoch {self.best_epoch_based_on_auc} ---"
        )
        self.model.eval()
        # output = self.trainer.predict(test_dataset=test_ds)
        # the raw logits from the model
        self.trainer.compute_metrics = None
        output = self.trainer.predict(test_dataset=test_ds, metric_key_prefix="test")

        metrics, _ = self._calculate_metrics(
            output.predictions,
            output.label_ids,
            self.label_names,
            self.log_dir,
            "Test_Set",
            thresholds=thresholds,
        )

        logging.info(f"--- Test Set Results ---")
        logging.info(
            f"MACRO:   AUC={metrics['macro/auc']:.4f}, F1={metrics['macro/f1']:.4f}, Recall={metrics['macro/recall']:.4f}, Precision={metrics['macro/precision']:.4f}, Accuracy={metrics['macro/accuracy']:.4f}"
        )
        logging.info(
            f"WEIGHTED: AUC={metrics['weighted/auc']:.4f}, F1={metrics['weighted/f1']:.4f}, Recall={metrics['weighted/recall']:.4f}, Precision={metrics['weighted/precision']:.4f}, Accuracy={metrics['weighted/accuracy']:.4f}"
        )

        preds_probs = torch.sigmoid(torch.tensor(output.predictions)).numpy()
        thresh_arr = np.array([thresholds.get(n, 0.5) for n in self.label_names])
        pred_bin = (preds_probs >= thresh_arr).astype(int)

        df = test_ds.df.copy()
        for i, name in enumerate(self.label_names):
            df[f"prob_{name}"], df[f"y_pred_{name}"] = preds_probs[:, i], pred_bin[:, i]

        # save_all_columns_except_text
        columns_to_save = [col for col in df.columns if col != self.text_col]

        df[columns_to_save].to_csv(self.log_dir / "test_predictions.csv", index=False)
        logging.info(
            f"Saved test predictions to {self.log_dir / 'test_predictions.csv'}"
        )
        test_metrics = {
            "n_samples": len(df),
            "n_fraud": int(df["fraud"].sum()),
            "macro": {
                "auc": metrics["macro/auc"],
                "f1": metrics["macro/f1"],
                "recall": metrics["macro/recall"],
                "precision": metrics["macro/precision"],
                "accuracy": metrics["macro/accuracy"],
            },
            "weighted": {
                "auc": metrics["weighted/auc"],
                "f1": metrics["weighted/f1"],
                "recall": metrics["weighted/recall"],
                "precision": metrics["weighted/precision"],
                "accuracy": metrics["weighted/accuracy"],
            },
            "per_class": {
                name: {
                    "auc": metrics.get(f"{name}_auc", 0.0),
                    "f1": metrics.get(f"{name}_f1", 0.0),
                    "recall": metrics.get(f"{name}_recall", 0.0),
                    "precision": metrics.get(f"{name}_precision", 0.0),
                    "accuracy": metrics.get(f"{name}_accuracy", 0.0),
                    "threshold": thresholds.get(name, 0.5),
                    "n_fraud": int(df[name].sum()),
                    "tp": int(((df[name] == 1) & (df[f"y_pred_{name}"] == 1)).sum()),
                    "tn": int(((df[name] == 0) & (df[f"y_pred_{name}"] == 0)).sum()),
                    "fp": int(((df[name] == 0) & (df[f"y_pred_{name}"] == 1)).sum()),
                    "fn": int(((df[name] == 1) & (df[f"y_pred_{name}"] == 0)).sum()),
                }
                for name in self.label_names
            },
        }

        with open(self.log_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        self.trainer.log(
            {f"test/{k.replace('/', '_')}": v for k, v in metrics.items() if "/" in k}
        )
        return metrics, df

    def save_model(self, path):
        if self.is_main_process:
            self.model.save_pretrained(str(path))
            self.tokenizer.save_pretrained(str(path))

    def find_max_length(self, df: pd.DataFrame) -> int:
        """
        Find the maximum length of the text column in the DataFrame.
        in terms of tokens.
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer is not initialized. Call load_model_and_tokenizer() first."
            )

        max_length = 0
        for text in tqdm(df[self.text_col], desc="Calculating max length"):
            tokens = self.tokenizer(
                text, truncation=False, padding=False, return_tensors="pt"
            )
            max_length = max(max_length, tokens.input_ids.shape[1])

        return max_length
