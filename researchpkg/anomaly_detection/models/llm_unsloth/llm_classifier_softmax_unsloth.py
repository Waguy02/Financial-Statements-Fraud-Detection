import os

from unsloth import FastLanguageModel

from researchpkg.anomaly_detection.models.llm_unsloth.lxt_utils import pdf_heatmap
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_dechow import (
    DECHOW_FEATURES_SHORT_DESCRIPTIONS,  # type: ignore
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_dechow import (
    DECHOW_FEATURES,
)

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic behavior

import gc
import importlib
import json
import logging
import random
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from lxt.efficient import monkey_patch
# from lxt.utils import clean_tokens, pdf_heatmap


from researchpkg.anomaly_detection.models.utils import clean_tokens

from sklearn.metrics import roc_auc_score  # Added for threshold optimization
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from researchpkg.anomaly_detection.config import EXPERIMENTS_DIR, SEED_TRAINING
from researchpkg.anomaly_detection.models.utils import (
    drop_random_keys,
    get_last_checkpoint,
    get_train_test_splitter,
    split_dataset_randomly,
    
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    AGGREGATE_FEATURES,
    BENEISH_PROBM,
    DIFF_FEATURES,
    EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT,
    IMPORTANT_TAGS,
    RATIO_FEATURES,
    RATIO_NET_WORKING_CAPITAL,
)
from researchpkg.utils import (
    configure_logger,  # Ensure reproducibility with random state
)

# Set default target modules for LoRA
# LORA_TARGET_MODULES = ["lm_head", # can easily be trained because it now has a small size
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj"]
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
    "lm_head",
]
# LORA_TARGET_MODULES = ["q_proj", "v_proj"]

EXCLUDED_FINANCIALS_FEATURES = set(
    [
        BENEISH_PROBM  # Too much biasing the model as it a probability of earnings manipulation
    ]
)

CURRENCY_FEATURES = set(AGGREGATE_FEATURES + DIFF_FEATURES + IMPORTANT_TAGS)
CURRENCY_FEATURES.add(RATIO_NET_WORKING_CAPITAL)


def is_with_currency(feature):
    return feature in CURRENCY_FEATURES


PERCENTAGE_FEATURES = set(RATIO_FEATURES) - set([RATIO_NET_WORKING_CAPITAL])
NOT_FRAUD_TOKEN = "NO"
FRAUD_TOKEN = "YES"

# Corresponds to the labels used in the modified DataCollator (0 for Not Fraud, 1 for Fraud)
NOT_FRAUD_LABEL_ID = 0
FRAUD_LABEL_ID = 1


def label_to_token(label):
    if label == "Not Fraud":
        return NOT_FRAUD_TOKEN
    elif label == "Fraud":
        return FRAUD_TOKEN
    raise ValueError(f"Unknown label: {label}")


def token_to_label_id(token):
    if token == NOT_FRAUD_TOKEN:
        return NOT_FRAUD_LABEL_ID
    elif token == FRAUD_TOKEN:
        return FRAUD_LABEL_ID
    raise ValueError(f"Unknown token: {token}")


def label_id_to_token(label_id):
    if label_id == NOT_FRAUD_LABEL_ID:
        return NOT_FRAUD_TOKEN
    elif label_id == FRAUD_LABEL_ID:
        return FRAUD_TOKEN
    raise ValueError(f"Unknown label ID: {label_id}")


# --- Helper function for threshold optimization ---
def find_best_auc_threshold(
    y_true, y_scores, epoch, log_dir, min_threshold=0.05, max_threshold=0.95
):
    """
    Find the best classification threshold to maximize AUC score for the positive class (Fraud).

    Args:
        y_true (list or np.array): True binary labels (0 for Not Fraud, 1 for Fraud).
        y_scores (list or np.array): Predicted probabilities for the positive class (Fraud).

    Returns:
        tuple: (best_threshold, best_auc)
    """
    # Get unique thresholds from the predicted probabilities
    thresholds = np.unique(y_scores)
    thresholds = thresholds[
        (thresholds >= min_threshold) & (thresholds <= max_threshold)
    ]  # Filter reasonable range

    if len(thresholds) == 0:
        return 0.5, roc_auc_score(y_true, y_scores)

    best_auc = 0.0
    best_threshold = 0.5  # Default

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        try:
            # Calculate AUC with current threshold
            current_auc = roc_auc_score(y_true, y_pred)

            if current_auc > best_auc:
                best_auc = current_auc
                best_threshold = threshold
        except ValueError:
            # Can happen if predictions contain only one class
            continue

    # Also compute F1 at the best AUC threshold for reference
    y_pred_at_best = (y_scores >= best_threshold).astype(int)
    f1_at_best = f1_score(y_true, y_pred_at_best)

    # Log results
    logging.info(
        f"Epoch {epoch} - Best AUC: {best_auc:.4f} at threshold: {best_threshold:.4f} (F1: {f1_at_best:.4f})"
    )

    return best_threshold, best_auc


def get_classification_probs(model, tokenizer, prompts: List[str], max_length: int):
    """
    Gets the classification probabilities [prob_not_fraud, prob_fraud] for a batch of prompts.

    Args:
        model: The trained language model.
        tokenizer: The tokenizer.
        prompts (List[str]): A list of input prompts ending before the expected answer.
        max_length (int): Max sequence length for tokenization.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 2) containing probabilities.
                      Column 0: Probability of Not Fraud (Class 0)
                      Column 1: Probability of Fraud (Class 1)
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Only take the first two logits
        last_token_logits = outputs.logits[:, -1, :2]

    # Apply softmax to get probabilities
    probabilities = torch.softmax(last_token_logits, dim=-1).cpu()

    return probabilities


COMPLETION_INSTRUCTION = "\n## My answer is: \n"


class PermutableUndersamplingDataset(TorchDataset):
    """
    A dataset that allows for permuting the order of its examples.

    Raises:
        ValueError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        dataset,
        train_labels,
        tokenizer,
        max_length,
    ):
        is_fraud = train_labels

        # Indices for fraud and non-fraud
        self.fraud_indices = np.where(is_fraud == 1)[0]
        self.non_fraud_indices = np.where(is_fraud == 0)[0]
        self.num_fraud = len(self.fraud_indices)

        assert self.num_fraud > 0, "No fraud cases found in the dataset"
        assert self.num_fraud < len(
            self.non_fraud_indices
        ), "More fraud cases than non-fraud cases"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.whole_dataset = dataset

        self.permute()

    def __len__(self):
        return len(self._dataset)

    def _generate_permutation(self):
        # Randomly select non-fraud samples to match fraud count
        selected_non_fraud_indices = np.random.choice(
            self.non_fraud_indices,
            size=self.num_fraud,
            replace=False,
        )

        # Combine indices
        combined_indices = np.concatenate(
            [self.fraud_indices, selected_non_fraud_indices]
        )
        np.random.shuffle(combined_indices)
        logging.info(
            f"Undersampling dataset: {len(self.fraud_indices)} fraud samples and {len(selected_non_fraud_indices)} non-fraud samples"
        )
        return combined_indices

    def permute(self):
        """
        Permute the dataset by shuffling the order of its examples.
        """
        combined_indices = self._generate_permutation()
        self._dataset = self.whole_dataset.select(combined_indices)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        """
        if idx >= len(self._dataset):
            raise ValueError("Index out of range")

        # Return the permuted example
        item = self._dataset[idx]

        return item


class PermutableUndersamplingDatasetStratified(TorchDataset):
    """
    A dataset that allows for dynamically undersampling and permuting the order of its examples,
    while preserving industry (SIC) and time period distributions.

    This class supports dynamic re-undersampling at each epoch while maintaining
    the distribution of samples across different industry sectors and time periods.
    """

    def __init__(
        self,
        dataset,
        train_labels,
        tokenizer,
        max_length,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.whole_dataset = dataset

        is_fraud = train_labels  # These should be 0/1

        # Indices for fraud and non-fraud
        self.fraud_indices = np.where(is_fraud == FRAUD_LABEL_ID)[0]
        self.non_fraud_indices = np.where(is_fraud == NOT_FRAUD_LABEL_ID)[0]
        self.num_fraud = len(self.fraud_indices)

        assert self.num_fraud > 0, "No fraud cases found in the dataset"
        assert (
            len(self.non_fraud_indices) > 0
        ), "No non-fraud cases found in the dataset"
        # assert self.num_fraud < len(
        #     self.non_fraud_indices # This might not hold if oversampling was applied before undersampling
        # ), "More fraud cases than non-fraud cases"

        # Extract metadata needed for stratified sampling
        self.sic_values = np.array(
            [example["sicagg"] for example in self.whole_dataset]
        )

        # Extract year from quarter information (e.g., "2019Q1" -> "2019")
        self.year_values = np.array(
            [example["quarter"].lower()[:4] for example in self.whole_dataset]
        )

        logging.info(
            f"PermutableUndersamplingDataset initialized with {self.num_fraud} fraud samples and {len(self.non_fraud_indices)} non-fraud samples"
        )

        # Initial permutation
        self.permute()

    def __len__(self):
        return len(self._dataset)

    def _generate_permutation(self):
        """
        Generate a permutation that preserves industry and time period distributions.
        Uses stratified sampling to maintain the distribution of fraud vs non-fraud
        within each industry-year group.
        """
        # Calculate overall sampling ratio for non-fraud (how many non-fraud to keep)
        # We want the final count of non-fraud to be equal to the fraud count
        if len(self.non_fraud_indices) == 0:
            undersampling_ratio = 0  # Avoid division by zero
            logging.warning("No non-fraud samples found for undersampling.")
        else:
            undersampling_ratio = self.num_fraud / len(self.non_fraud_indices)
            logging.info(
                f"Target Undersampling ratio: {undersampling_ratio:.4f} (aiming for {self.num_fraud} non-fraud cases)"
            )

        # Group data by SIC and year
        groups = {}
        for idx in range(len(self.whole_dataset)):
            sic = self.sic_values[idx]
            year = self.year_values[idx]  # Include year for stratification
            group_key = f"{sic}_{year}"  # Stratify by SIC-Year combination

            if group_key not in groups:
                groups[group_key] = {"fraud": [], "non_fraud": []}

            # Check if the index corresponds to fraud or non-fraud
            is_fraud_sample = idx in self.fraud_indices
            is_non_fraud_sample = idx in self.non_fraud_indices

            if is_fraud_sample:
                groups[group_key]["fraud"].append(idx)
            elif is_non_fraud_sample:  # Use elif to be explicit
                groups[group_key]["non_fraud"].append(idx)
            # else: ignore indices not in either set, if any exist

        # Perform stratified sampling within each group
        selected_fraud_indices = []
        selected_non_fraud_indices = []

        for group_key, group_data in groups.items():
            # Take all fraud cases from this group
            selected_fraud_indices.extend(group_data["fraud"])

            # Sample non-fraud cases according to the ratio
            num_non_fraud_in_group = len(group_data["non_fraud"])
            if num_non_fraud_in_group > 0 and undersampling_ratio > 0:
                # Calculate how many non-fraud samples this group *should* contribute
                # Proportional allocation based on group size relative to total non-fraud
                group_non_fraud_proportion = num_non_fraud_in_group / len(
                    self.non_fraud_indices
                )
                num_non_fraud_to_sample = max(
                    1, int(round(self.num_fraud * group_non_fraud_proportion))
                )

                # Ensure we don't sample more than available
                num_non_fraud_to_sample = min(
                    num_non_fraud_to_sample, num_non_fraud_in_group
                )

                # Sample without replacement
                sampled_non_fraud = np.random.choice(
                    group_data["non_fraud"],
                    size=num_non_fraud_to_sample,
                    replace=False,
                )
                selected_non_fraud_indices.extend(sampled_non_fraud)

        # Combine selected indices
        combined_indices = np.concatenate(
            [selected_fraud_indices, selected_non_fraud_indices]
        )
        np.random.shuffle(combined_indices)

        # Log statistics about the undersampled dataset
        final_fraud_count = len(selected_fraud_indices)
        final_non_fraud_count = len(selected_non_fraud_indices)
        logging.info(
            f"Stratified undersampling result: {final_fraud_count} fraud samples and {final_non_fraud_count} non-fraud samples"
        )

        if len(combined_indices) > 0:
            fraud_ratio = final_fraud_count / len(combined_indices)
            logging.info(f"Fraud ratio after undersampling: {fraud_ratio:.2%}")
        else:
            logging.warning("Stratified undersampling resulted in an empty dataset.")
            return np.array([])  # Return empty array if no samples selected

        # (Optional) Calculate and log distribution by industry - Keep original logging if useful
        # ... (logging code omitted for brevity but can be kept) ...

        return combined_indices

    def permute(self):
        """
        Permute the dataset by stratified undersampling and shuffling the order of examples.
        This maintains industry and time period distributions.
        """
        combined_indices = self._generate_permutation()
        if len(combined_indices) > 0:
            self._dataset = self.whole_dataset.select(combined_indices)
            logging.info(f"Dataset permuted with {len(self._dataset)} examples")
        else:
            self._dataset = self.whole_dataset.select(
                []
            )  # Create an empty dataset selection
            logging.warning("Dataset permutation resulted in zero samples.")

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        """
        if not hasattr(self, "_dataset") or len(self._dataset) == 0:
            raise IndexError("Dataset is empty after undersampling.")
        if idx >= len(self._dataset):
            raise IndexError("Index out of range")

        # Return the permuted example
        return self._dataset[idx]


class EvaluationCallback(TrainerCallback):
    """Custom callback for evaluation at the end of each epoch, including threshold optimization."""

    def __init__(
        self,
        trainer,
        tokenizer,
        log_dir,
        max_length,
        # max_new_tokens is no longer relevant for direct classification
        run_eval_on_start=True,
    ):
        self.trainer = trainer
        self.train_dataset = (
            trainer.train_dataset
        )  # This might be the undersampling wrapper
        self.eval_dataset = trainer.eval_dataset

        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.max_length = max_length
        # self.max_new_tokens = max_new_tokens # Removed
        self.run_eval_on_start = run_eval_on_start

        self.lock = threading.Lock()  # Keep if needed for future multi-threading

        if hasattr(trainer, "undersample"):
            self.undersample = trainer.undersample
        else:
            self.undersample = False

        # Fast generation/vllm is less relevant now, but keep logic if needed elsewhere
        if hasattr(trainer, "fast_generation"):
            self.fast_generation = trainer.fast_generation
            self.current_lora_request = None
        else:
            self.fast_generation = False

        self.best_threshold_per_epoch = {}  # Store best threshold found

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.undersample and hasattr(self.train_dataset, "permute"):
            logging.info(
                f"Epoch {state.epoch}: Permuting training dataset for undersampling."
            )
            # Updating the train_dataloader with dynamic undersampling
            self.train_dataset.permute()

        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.epoch == 0 and self.run_eval_on_start:
            logging.info("Running evaluation at the beginning of training...")
            self.on_epoch_end(args, state, control, **kwargs)
        return super().on_train_begin(args, state, control, **kwargs)

    # def extract_answer_prediction(self, prediction_text):
    #     # This is no longer needed as we work with probabilities/class IDs
    #     return prediction_text

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation at the end of each epoch with threshold optimization."""
        logging.info(f"--- Starting Evaluation for Epoch {int(state.epoch)} ---")
        model = self.trainer.model
        model.eval()  # Ensure model is in evaluation mode

        # --- VLLM/Fast Generation Handling (Keep if applicable, but less central now) ---
        vllm_client = None
        if hasattr(self.trainer, "vllm_client"):
            vllm_client = (
                self.trainer.vllm_client
            )  # Might not be usable with direct classification
        elif self.fast_generation:
            # Need to ensure LoRA weights are correctly loaded for manual forward pass
            last_checkpoint = get_last_checkpoint(self.log_dir)
            if last_checkpoint and os.path.exists(
                os.path.join(last_checkpoint, "adapter_model.safetensors")
            ):
                logging.info(
                    f"Loading LoRA weights from {last_checkpoint} for evaluation."
                )
                # This assumes the base model already has the PEFT config applied
            #  adapters_weights = load_peft_weights(last_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
            #  set_peft_model_state_dict(model, adapters_weights)
            else:
                logging.warning(
                    f"Fast generation enabled, but no valid checkpoint found at {self.log_dir}. Using current model state."
                )
            FastLanguageModel.for_inference(
                model
            )  # Prepare for inference if using Unsloth's helper
        else:
            logging.info("Using default model state for evaluation.")

            if hasattr(FastLanguageModel, "for_inference"):
                FastLanguageModel.for_inference(model)

        # --- Data Collection ---
        true_labels_token = []  # Store original 'YES'/'NO'
        true_labels_id = []  # Store 0/1
        predicted_probs = []  # Store probability of FRAUD_LABEL_ID (class 1)
        ciks = []
        sics = []
        quarters = []
        misstatements = []  # Ground truth labels from data source (if different)

        # --- Batch Processing Function ---d
        def process_batch_for_eval(examples, model, tokenizer):
            batch_prompts = []
            batch_true_token = []
            batch_true_id = []
            batch_ciks = []
            batch_sics = []
            batch_quarters = []
            batch_misstatements = []

            for example in examples:
                # Extract prompt and true answer
                input_text = example["text"]
                input_parts = input_text.split(COMPLETION_INSTRUCTION)
                prompt = input_parts[0]  # The part before "## My answer is:"
                true_answer_token = example["answer"].strip()  # Should be YES or NO

                assert true_answer_token in [FRAUD_TOKEN, NOT_FRAUD_TOKEN]

                batch_prompts.append(
                    prompt + COMPLETION_INSTRUCTION
                )  # Add the prompt with instruction
                batch_true_token.append(true_answer_token)
                batch_true_id.append(token_to_label_id(true_answer_token))
                batch_ciks.append(int(example["cik"]))
                batch_sics.append(example["sic"])
                batch_quarters.append(example["quarter"])
                batch_misstatements.append(example["misstatements"])

            # Get classification probabilities using the dedicated function
            # Note: Ensure `get_classification_probs` handles the model (PEFT or base) correctly
            probabilities = get_classification_probs(
                model, tokenizer, batch_prompts, self.max_length
            )
            # probabilities should be shape [batch_size, 2]

            # Extract probability for the positive class (Fraud, ID=1)
            fraud_probabilities = probabilities[:, FRAUD_LABEL_ID].cpu().tolist()

            return (
                batch_true_token,
                batch_true_id,
                fraud_probabilities,
                batch_ciks,
                batch_sics,
                batch_quarters,
                batch_misstatements,
            )

        # --- Evaluation Loop ---
        with torch.no_grad():
            batch_size = self.trainer.args.per_device_eval_batch_size
            eval_dataset = self.trainer.eval_dataset  # Use the actual eval dataset
            dataset_length = len(eval_dataset)

            for i in tqdm(
                range(0, dataset_length, batch_size),
                desc=f"Evaluation at epoch {int(state.epoch)} (batch size: {batch_size})",
            ):
                batch_examples = [
                    eval_dataset[j]
                    for j in range(i, min(i + batch_size, dataset_length))
                ]

                if not batch_examples:
                    continue  # Skip empty batches

                # Process the batch
                (
                    batch_true_token,
                    batch_true_id,
                    batch_fraud_probs,
                    batch_ciks,
                    batch_sics,
                    batch_quarters,
                    batch_misstatements,
                ) = process_batch_for_eval(batch_examples, model, self.tokenizer)

                # Collect results
                true_labels_token.extend(batch_true_token)
                true_labels_id.extend(batch_true_id)
                predicted_probs.extend(batch_fraud_probs)
                ciks.extend(batch_ciks)
                sics.extend(batch_sics)
                quarters.extend(batch_quarters)
                misstatements.extend(batch_misstatements)

                # Log sample predictions occasionally
                if state.epoch == 0 or random.random() < 0.001:
                    text = batch_examples[0]["text"]
                    result = {
                        "text": text,
                        "prediction": batch_fraud_probs[0],
                        "answer": batch_true_token[0],
                        "true_label": batch_true_id[0],
                        "predicted_label": token_to_label_id(batch_true_token[0]),
                        "cik": batch_ciks[0],
                        "sic": batch_sics[0],
                        "quarter": batch_quarters[0],
                        "probabilities": batch_fraud_probs[0],
                    }
                    logging.info(f"Prompt: {result['text']}")
                    logging.info(f"Ground Truth: {result['answer']}")
                    logging.info(f"Fraud Prob: {result['probabilities']}")
                    logging.info(
                        f"CIK: {result['cik']}, SIC: {result['sic']}, Quarter: {result['quarter']}"
                    )  # --- Convert to Numpy Arrays ---

        # --- Threshold Optimization ---
        if not true_labels_id or not predicted_probs:
            logging.error(
                f"Epoch {int(state.epoch)}: No evaluation data processed. Skipping metrics calculation."
            )
            model.train()  # Put model back in train mode
            return control

        best_threshold, optimized_auc = find_best_auc_threshold(
            true_labels_id, predicted_probs, int(state.epoch), self.log_dir
        )
        self.best_threshold_per_epoch[int(state.epoch)] = best_threshold  # Store it

        # --- Recalculate Predictions based on Optimal Threshold ---
        predicted_labels_id = [
            FRAUD_LABEL_ID if prob >= best_threshold else NOT_FRAUD_LABEL_ID
            for prob in predicted_probs
        ]
        predicted_labels_token = [
            label_id_to_token(label_id) for label_id in predicted_labels_id
        ]

        # --- Calculate Metrics using Optimized Predictions ---
        accuracy = accuracy_score(true_labels_id, predicted_labels_id)
        # Use label IDs (0/1) for sklearn metrics, specify pos_label=FRAUD_LABEL_ID (which is 1)
        precision = precision_score(
            true_labels_id,
            predicted_labels_id,
            pos_label=FRAUD_LABEL_ID,
            zero_division=0,
        )
        recall = recall_score(
            true_labels_id,
            predicted_labels_id,
            pos_label=FRAUD_LABEL_ID,
            zero_division=0,
        )
        f1_final = f1_score(
            true_labels_id,
            predicted_labels_id,
            pos_label=FRAUD_LABEL_ID,
            zero_division=0,
        )  # F1 after thresholding

        # Sanity check: f1_final should be very close to optimized_f1 if thresholding works as expected
        if not np.isclose(f1_final, optimized_auc, atol=1e-4):
            logging.warning(
                f"Epoch {int(state.epoch)}: Final F1 ({f1_final:.4f}) differs significantly from optimized F1 ({optimized_auc:.4f}). Check thresholding logic."
            )

        macro_f1 = f1_score(
            true_labels_id, predicted_labels_id, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            true_labels_id, predicted_labels_id, average="weighted", zero_division=0
        )
        auc_score = roc_auc_score(true_labels_id, predicted_probs)
        # Use token labels ('YES'/'NO') for classification report for readability
        report = classification_report(
            true_labels_token,
            predicted_labels_token,
            labels=[FRAUD_TOKEN, NOT_FRAUD_TOKEN],  # Order matters for target names
            target_names=["Fraud", "Not Fraud"],  # Corresponding names
            zero_division=0,
        )

        logging.info(
            f"--- Epoch {int(state.epoch)} Evaluation Results (Optimized Threshold: {best_threshold:.4f}) ---"
        )
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision (Fraud): {precision:.4f}")
        logging.info(f"Recall (Fraud): {recall:.4f}")
        logging.info(
            f"F1 Score (Fraud - Optimized): {f1_final:.4f}"
        )  # Report the F1 found during optimization
        logging.info(f"Macro F1 Score: {macro_f1:.4f}")
        logging.info(f"AUC Score: {auc_score:.4f}")
        logging.info(f"Weighted F1 Score: {weighted_f1:.4f}")
        logging.info(f"Classification Report:\n{report}")

        # --- Save Metrics ---
        metrics_path = os.path.join(
            self.log_dir, f"metrics_epoch_{int(state.epoch)}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "epoch": int(state.epoch),
                    "accuracy": float(accuracy),
                    "precision_fraud": float(precision),
                    "recall_fraud": float(recall),
                    "f1_fraud_optimized": float(optimized_auc),  # Save the optimized F1
                    "f1_fraud_at_threshold": float(
                        f1_final
                    ),  # Also save F1 calculated *after* applying threshold
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                    "auc_score": float(auc_score),
                    "best_threshold": float(best_threshold),  # Save the threshold used
                    "report": report,
                    "num_eval_samples": dataset_length,
                    "num_fraud_samples": sum(true_labels_id),
                },
                f,
                indent=2,
            )
        logging.info(f"Saved epoch metrics to {metrics_path}")

        # --- Log Metrics for TensorBoard ---
        tb_metrics_dict = {
            "eval/accuracy": float(accuracy),
            "eval/precision_fraud": float(precision),
            "eval/recall_fraud": float(recall),
            "eval/f1_fraud_optimized": float(optimized_auc),
            "eval/f1_fraud_at_threshold": float(f1_final),
            "eval/macro_f1": float(macro_f1),
            "eval/weighted_f1": float(weighted_f1),
            "eval/auc_score": float(auc_score),
            "eval/best_threshold": float(best_threshold),
            "epoch": float(state.epoch),  # Ensure epoch is float for tensorboard scalar
            # "step": int(state.global_step), # Global step might be more appropriate than epoch sometimes
        }
        # Use trainer.log which handles tensorboard logging
        self.trainer.log(tb_metrics_dict)

        # --- Save Detailed Predictions ---
        val_results_df = pd.DataFrame(
            {
                "cik": ciks,
                "sic": sics,
                "quarter": quarters,
                "y_true_token": true_labels_token,
                "y_true_id": true_labels_id,
                "y_pred_token": predicted_labels_token,  # Save predictions based on optimized threshold
                "y_pred_id": predicted_labels_id,
                "fraud_probability": predicted_probs,  # Save the raw probability
                "misstatements": misstatements,
            }
        )

        val_csv_path = os.path.join(
            self.log_dir, f"val_predictions_epoch_{int(state.epoch)}.csv"
        )
        val_results_df.to_csv(val_csv_path, index=False)
        logging.info(f"Saved detailed validation predictions to {val_csv_path}")

        FastLanguageModel.for_training(model)  # Switch back to training mode if needed
        logging.info(f"--- Finished Evaluation for Epoch {int(state.epoch)} ---")

        torch.cuda.empty_cache()  # Clear GPU memory if needed
        gc.collect()  # Collect garbage to free up memory
        time.sleep(
            5
        )  # Optional: Sleep to allow for any background processes to catch up

        # Simply save all the trainable parameters of the model
        trainable_params = {
            k: v for k, v in model.named_parameters() if v.requires_grad
        }
        parameters_path = (
            self.log_dir / f"trainable_parameters_epoch_{int(state.epoch)}.pt"
        )
        save_dict = {
            k: v.clone().cpu() for k, v in trainable_params.items()
        }  # Move to CPU for saving
        torch.save(save_dict, parameters_path)
        logging.info(f"Saved trainable parameters to {parameters_path}")

        # self.clean_previous_checkpoints() # Clean up previous checkpoints to save space
        return control

    def clean_previous_checkpoints(self):
        """
        Cleanup chcekpoints of previous epochs except the last one and the best one .
        If the last one is the best one, clean all the others.
        """
        # 1. Find best epoch in metrics_jon files
        metrics_files = list(self.log_dir.glob("metrics_epoch_*.json"))
        best_epoch = None
        all_epochs = []
        best_f1 = -1
        for metrics_file in metrics_files:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                if metrics["f1_fraud_optimized"] > best_f1:
                    best_f1 = metrics["f1_fraud_optimized"]
                    best_epoch = metrics["epoch"]
                    all_epochs.append(metrics["epoch"])
        logging.info(f"Current Best epoch: {best_epoch} with F1: {best_f1}")
        last_epoch = max(all_epochs)
        # 2. Find all checkpoints in the log_dir
        checkpoints = list(self.log_dir.glob("checkpoint-*"))
        # For each checkpoint dir check if the epoch is either last one or best one
        for checkpoint in checkpoints:
            trainer_state = json.load(open(checkpoint / "trainer_state.json", "r"))
            epoch = int(trainer_state["epoch"])
            if epoch != best_epoch and epoch != last_epoch:
                shutil.rmtree(checkpoint)
                # Ælso remove the lm_head file if it exists
                lm_head_path = os.path.join(checkpoint, f"lm_head_epoch_{epoch}.pt")
                if os.path.exists(lm_head_path):
                    os.remove(lm_head_path)
                    logging.info(f"Removed lm_head file: {lm_head_path}")


class LLMClassifierSoftmax:
    """
    LLM classifier for fraud detection using a softmax output layer over 2 classes.
    """

    def __init__(
        self,
        config,
    ):
        """
        Initialize the classifier.

        Args:
            config (dict): Configuration dictionary. Must include:
                - model_url
                - lora_r
                - lora_alpha
                - max_context
                - fold_id
                - Optionally: model_alias, max_new_tokens (should be 1 or unused),
                              per_device_train_batch_size, per_device_eval_batch_size,
                              gradient_accumulation_steps, checkpoint_timestamp,
                              experiments_dir, load_in_4bit, gpu_memory_utilization,
                              reset_llm_head, lora_target_modules, undersample, etc.
        """
        assert config is not None, "Configuration should be provided"
        self.model_url = config["model_url"]
        self.model_alias = config.get("model_alias", self.model_url.split("/")[-1])
        self.lora_r = config["lora_r"]
        self.lora_alpha = config["lora_alpha"]
        self.max_length = config["max_context"]

        # max_new_tokens is irrelevant for direct classification, but check if set
        self.max_new_tokens = config.get("max_new_tokens", 1)
        # if self.max_new_tokens != 1:
        # logging.warning(f"max_new_tokens is set to {self.max_new_tokens}, but this model uses direct classification. It will be ignored during evaluation.")
        # No need for max_new_tokens in this setup

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

        # Training parameters
        self.per_device_train_batch_size = config.get("per_device_train_batch_size", 2)
        self.per_device_eval_batch_size = config.get("per_device_eval_batch_size", 2)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)

        # Get checkpoint timestamp if provided
        self.checkpoint_timestamp = config.get("checkpoint_timestamp", None)

        #

        # Set up experiment directory
        self.experiments_dir = config.get(
            "experiments_dir",
            EXPERIMENTS_DIR / f"llm_fraud_classifier_softmax",  # Modified name
        )

        if not isinstance(self.experiments_dir, Path):
            self.experiments_dir = Path(self.experiments_dir)

        if not self.experiments_dir.exists():
            self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # If we have a checkpoint, use that timestamp for the log directory
        fold_id = self.config["fold_id"]

        self.should_load_train_data= True
        
        self.auto_continue = config.get("auto_continue", False)
        assert not (
            self.auto_continue and self.checkpoint_timestamp
        ), "auto_continue and checkpoint_timestamp arguments are mutually exclusive"
        if self.auto_continue:
            ## Find the last experiments correspond this to this settings
            directory = self.experiments_dir / f"{self.model_alias}" / f"fold_{fold_id}"

            # Find latest experiments

            all_experiments_dirs = [p.stem for p in directory.iterdir() if p.is_dir()]
            list_timestamps = list(sorted(all_experiments_dirs, reverse=True))

            if len(list_timestamps) == 0:
                raise Exception(
                    f"auto_continue mode but no previous run found in {str(directory)}"
                )
            latest_checkpoint_ts = list_timestamps[0]
            previous_run = directory / latest_checkpoint_ts
            if (previous_run / "test_metrics.json").exists():
                print(
                    f"Previous run {previous_run} already finish. Disable auto_continue to make a new run."
                    f"\n Will run test again"
                )
                self.should_load_train_data = False
            
            self.checkpoint_timestamp = latest_checkpoint_ts

        if self.checkpoint_timestamp:
            timestamp = self.checkpoint_timestamp
            self.log_dir = (
                self.experiments_dir
                / f"{self.model_alias}"
                / f"fold_{fold_id}"
                / f"{timestamp}"
            )
            assert (
                self.log_dir.exists()
            ), f"Checkpoint directory {self.log_dir} does not exist"
            
            
            test_metrics_path = self.log_dir / "test_metrics.json"
            if test_metrics_path.exists():
                logging.info(
                    f"Test metrics already exist at {test_metrics_path}"
                )
                self.should_load_train_data = False
            
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = (
                self.experiments_dir
                / f"{self.model_alias}"
                / f"fold_{fold_id}"
                / f"{timestamp}"
            )

        self.log_file = self.log_dir / f"experiment.log"

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        configure_logger(
            logFile=self.log_file,  # Ensure Path is converted to string
            logLevel=logging.INFO,
        )

        if self.checkpoint_timestamp:
            logging.info(f"Continuing training from checkpoint: {self.log_dir}")
        else:
            logging.info(f"Starting new training session: {self.log_dir}")

        self.load_model_and_tokenizer(
            fast_inference=self.config.get(
                "fast_generation", False
            )  # Keep for potential future use
        )
        lora_target_modules = self.config.get(
            "lora_target_modules", LORA_TARGET_MODULES
        )

        num_layers_to_transform = self.config.get("num_layers_to_finetune", 0)
        if num_layers_to_transform == 0:
            logging.info(
                "num_layers_to_finetune is set to 0, No LoRA layers will be transformed."
            )
            num_layers_to_transform = None
        else:
            logging.info(f"LoRA target modules: {lora_target_modules}")
            self.apply_peft_lora(target_modules=lora_target_modules)

    def load_model_and_tokenizer(self, fast_inference=False):
        """
        Load the LLM model and tokenizer from Hugging Face.
        """

        logging.info(f"Loading {self.model_alias} model: {self.model_url}")
        logging.info(f"Maximum context length: {self.max_length}")
        if self.config.get("offline", False):
            logging.info("Loading model in offline mode (local files only).")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_url,
            max_seq_length=self.max_length,
            dtype=None,  # None for auto detection
            full_finetuning=False,
            random_state=self.config.get("seed", SEED_TRAINING),
            load_in_4bit=self.config.get(
                "load_in_4bit", True
            ),  # Use 4bit quantization to reduce memory usage
            fast_inference=fast_inference,  # Use fast inference mode
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.7),
            local_files_only=self.config.get("offline", False),
        )
        self.model.config.use_cache = False  # Reduces memory usagef

        try:
            self.model.config.text_config.use_cache = False  # Reduces memory usage
        except:
            logging.info("No text_config attribute in the model")

        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        logging.info(f"{self.model_alias} model loaded successfully.")
        # Add the fraud and not fraud tokens to the tokenizer

        not_fraud_token_id = self.tokenizer.convert_tokens_to_ids(NOT_FRAUD_TOKEN)
        fraud_token_id = self.tokenizer.convert_tokens_to_ids(FRAUD_TOKEN)

        # MOdify the llm head to have 2 classes but keep the other with weight -100 for retrocompatibility
        reset_lm_head = self.config.get("reset_llm_head", False)
        if not reset_lm_head:
            new_params = torch.nn.Parameter(
                torch.vstack(
                    [
                        self.model.lm_head.weight[not_fraud_token_id, :],
                        self.model.lm_head.weight[fraud_token_id, :],
                    ]
                )
            )

        else:

            # assert "lm_head" in target_modules, "lm_head should be in the target modules because you reiniitialize it for the classifier"
            logging.info("Randomly initializing the new parameters")
            new_params = torch.nn.Parameter(
                torch.randn(
                    2,
                    self.model.lm_head.weight.shape[1],
                    dtype=self.model.lm_head.weight.dtype,
                )
            )

        logging.info(f"Fraud token ID: {fraud_token_id}")
        logging.info(f"Not fraud token ID: {not_fraud_token_id}")
        logging.info(f"New parameters shape: {new_params.shape}")

        # Set the new parameters for the lm_head

        self.model.lm_head.weight = torch.nn.Parameter(
            new_params, requires_grad=False
        )  # Freeze the llm head, wil be fine tuned with LoRA

    def apply_peft_lora(self, target_modules=None):
        """
        Apply PEFT and LoRA to the model. Ensures lm_head is included if modified.

        Args:
            target_modules (list, optional): List of modules to apply LoRA to.
        """
        if target_modules is None:
            target_modules = LORA_TARGET_MODULES

        # Ensure lm_head is targeted because we modified it
        # if "lm_head" not in target_modules:
        #     logging.info("Adding 'lm_head' to LoRA target modules as it was resized.")
        #     target_modules.append("lm_head")

        logging.info("Applying PEFT and LoRA...")
        layers_to_transform = self.config.get("layers_to_transform", None)
        num_layers_to_transform = self.config.get("num_layers_to_finetune", None)

        if not layers_to_transform and num_layers_to_transform:
            try:
                # Attempt to get number of layers from config
                # Common attribute names: num_hidden_layers, n_layer
                num_model_layers = None
                if hasattr(self.model.config, "num_hidden_layers"):
                    num_model_layers = self.model.config.num_hidden_layers
                elif hasattr(self.model.config, "n_layer"):
                    num_model_layers = self.model.config.n_layer
                # Add other potential config attribute names if needed

                if num_model_layers is None:
                    raise AttributeError(
                        "Could not automatically determine number of model layers from config."
                    )

                logging.info(
                    f"Number of model layers found in config: {num_model_layers}"
                )
                if num_layers_to_transform > num_model_layers:
                    raise ValueError(
                        f"num_layers_to_finetune ({num_layers_to_transform}) is greater than the number of model layers ({num_model_layers})"
                    )
                layers_to_transform = list(
                    range(num_model_layers - num_layers_to_transform, num_model_layers)
                )
            except AttributeError as e:
                logging.error(
                    f"Error determining model layers: {e}. Cannot apply 'num_layers_to_finetune'. Apply LoRA to all target modules."
                )
                layers_to_transform = None
            except ValueError as e:
                logging.error(f"{e}. Apply LoRA to all target modules.")
                layers_to_transform = None

        logging.info(
            f"Layers to transform by LoRA: {layers_to_transform if layers_to_transform else 'All specified target modules'}"
        )

        # Common PEFT arguments from config
        use_rslora = self.config.get("use_rslora", False)
        lora_dropout = self.config.get(
            "lora_dropout", 0.05
        )  # Add slight dropTrueout default

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_r,
            use_rslora=use_rslora,
            target_modules=target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Bias="none" is often optimized
            use_gradient_checkpointing="unsloth",  # Recommended by Unsloth
            random_state=3407,  # Use global seed
            max_seq_length=self.max_length,  # Pass max_seq_length here as well
            layers_to_transform=layers_to_transform,
        )

        # if self.config.get("reset_llm_head", False):
        #     logging.info("Making the lm_head trainable")
        #     # Set lm_head to be trainable
        #     for param in self.model.lm_head.parameters():
        #         param.requires_grad = True

        logging.info("PEFT and LoRA applied successfully.")
        self.model.print_trainable_parameters()

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

        
        if self.should_load_train_data:
            logging.info(f"Loading train data from {train_path}")
            train_df = pd.read_csv(train_path)
            self.train_path, self.test_path = train_path, test_path
            

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
                if self.config.get("force_random_splitter", False):
                    logging.info("Force_random_splitting for valiadtion")
                    splitter = split_dataset_randomly
                else:
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
        else:
            logging.info(
                "Skipping loading training data as it was already loaded in auto_continue mode."
            )
            #Only load two rows to satisfy the Trainer requirements
            train_df = pd.read_csv(train_path).head(10)
            val_df = pd.read_csv(train_path).head(10)
            train_df = self._process_loaded_data(train_df)
            val_df = self._process_loaded_data(val_df)

        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        test_df = self._process_loaded_data(test_df)

        return train_df, val_df, test_df

    def load_cv_data(self, train_path, val_path, test_path):
        """Loads data for Cross-Validation folds."""
        logging.info(
            f"Loading CV data: Train={train_path}, Val={val_path}, Test={test_path}"
        )
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # Apply preprocessing
        train_df = self._process_loaded_data(train_df)
        val_df = self._process_loaded_data(val_df)
        test_df = self._process_loaded_data(test_df)

        # Oversampling only on the training fold
        if self.config.get("oversample", False):
            logging.info("Oversampling fraud cases in CV training fold...")
            train_df = self.oversample_fraud_cases(train_df)

        # Logging sizes
        logging.info(
            f"CV Train data size: {len(train_df)} (Fraud: {train_df['is_fraud'].sum()})"
        )
        logging.info(
            f"CV Validation data size: {len(val_df)} (Fraud: {val_df['is_fraud'].sum()})"
        )
        logging.info(
            f"CV Test data size: {len(test_df)} (Fraud: {test_df['is_fraud'].sum()})"
        )

        return train_df, val_df, test_df

    def oversample_fraud_cases(self, df):
        """
        Oversample minority class (assumed 'is_fraud' == True) to match majority class count.

        Args:
            df (pd.DataFrame): The dataframe to oversample (typically training data).
                               Must contain an 'is_fraud' column (boolean or 0/1).

        Returns:
            pd.DataFrame: The oversampled dataframe.
        """
        fraud_label = True if df["is_fraud"].dtype == "bool" else 1
        non_fraud_label = False if df["is_fraud"].dtype == "bool" else 0

        fraud_df = df[df["is_fraud"] == fraud_label]
        non_fraud_df = df[df["is_fraud"] == non_fraud_label]

        num_fraud = len(fraud_df)
        num_non_fraud = len(non_fraud_df)

        if num_fraud == 0:
            logging.warning(
                "Oversampling requested, but no fraud cases found in the input dataframe."
            )
            return df
        if num_non_fraud == 0:
            logging.warning(
                "Oversampling requested, but no non-fraud cases found. Returning only fraud cases."
            )
            return fraud_df  # Or potentially raise error depending on desired behavior

        if num_fraud >= num_non_fraud:
            logging.info(
                f"No oversampling needed: Fraud cases ({num_fraud}) >= Non-fraud cases ({num_non_fraud})."
            )
            return df

        logging.info(
            f"Oversampling: Initial counts - Fraud={num_fraud}, Non-Fraud={num_non_fraud}"
        )

        # Oversample fraud cases to match the number of non-fraud cases
        oversampled_fraud_df = fraud_df.sample(
            n=num_non_fraud, replace=True, random_state=SEED_TRAINING
        )

        # Combine oversampled fraud cases with original non-fraud cases
        oversampled_df = pd.concat(
            [oversampled_fraud_df, non_fraud_df], ignore_index=True
        )

        # Shuffle the combined dataframe
        oversampled_df = oversampled_df.sample(
            frac=1, random_state=SEED_TRAINING
        ).reset_index(drop=True)

        logging.info(
            f"Oversampling complete: Total size={len(oversampled_df)}, Fraud={len(oversampled_fraud_df)}, Non-Fraud={num_non_fraud}"
        )
        logging.info(
            f"Number of unique fraud CIKs after oversampling: {oversampled_fraud_df['cik'].nunique()}"
        )

        return oversampled_df

    def _process_loaded_data(self, df):
        """
        Basic data processing: Ensure 'is_fraud' exists and is boolean/int.
        Subclasses should override this for specific data cleaning/feature engineering.

        Args:
            df (pd.DataFrame): The loaded dataframe

        Returns:
            pd.DataFrame: Processed dataframe
        """
        if "is_fraud" not in df.columns:
            raise ValueError("Input dataframe must contain an 'is_fraud' column.")

        # Convert to integer (0/1) for consistency with metrics and labels
        if df["is_fraud"].dtype == "bool":
            df["is_fraud"] = df["is_fraud"].astype(int)
        elif not pd.api.types.is_integer_dtype(df["is_fraud"]):
            try:
                # Attempt conversion if it looks like numeric/boolean strings
                df["is_fraud"] = (
                    df["is_fraud"]
                    .map(
                        {
                            True: 1,
                            False: 0,
                            "True": 1,
                            "False": 0,
                            "1": 1,
                            "0": 0,
                            1: 1,
                            0: 0,
                        }
                    )
                    .astype(int)
                )
            except Exception as e:
                logging.error(
                    f"Could not convert 'is_fraud' column to 0/1 integers: {e}"
                )
                raise

        # Example: Fill missing numerical values if needed (subclass responsibility)
        # numeric_cols = df.select_dtypes(include=np.number).columns
        # df[numeric_cols] = df[numeric_cols].fillna(0)

        logging.debug(f"Processed data shape: {df.shape}")
        logging.debug(f"Data types after initial processing:\n{df.dtypes}")
        return df

    def generate_prompt(self, row, idx=None, **kwargs):
        """
        Generate the training example dictionary.
        The 'text' field should contain the input prompt followed by the completion instruction.
        The 'answer' field contains the target token (YES/NO).

        Args:
            row (pd.Series): Row from the DataFrame.
            idx (int, optional): Index of the example.
            **kwargs: Additional arguments (like drop_rate).

        Returns:
            dict: Dictionary for training/evaluation (e.g., {'text': ..., 'answer': ..., 'cik': ...}).
                  Return None if the example should be skipped.
        """
        # This method MUST be implemented by subclasses to format the specific input data
        # into the prompt structure.
        raise NotImplementedError("Subclasses must implement generate_prompt method")

    def truncate_and_format_prompt(self, content):
        """
        Ensures the *input* part of the prompt is within the model's context length,
        leaving space for the model's processing and the single classification token.

        Args:
            content (str): The generated input content before the completion instruction.

        Returns:
            str: Truncated content.
        """
        # This method needs careful implementation based on how prompts are built.
        # It should truncate the `content` *before* COMPLETION_INSTRUCTION is added.
        # We need to know the token length of COMPLETION_INSTRUCTION.

        instruction_tokens = self.tokenizer.encode(
            COMPLETION_INSTRUCTION, add_special_tokens=False
        )
        # Reserve a few tokens for model processing overhead + classification token
        reserved_tokens = len(instruction_tokens) + 5

        max_input_tokens = self.max_length - reserved_tokens

        if max_input_tokens <= 0:
            raise ValueError(
                f"max_length ({self.max_length}) is too small to accommodate instructions and minimal input."
            )

        # Tokenize the content
        content_tokens = self.tokenizer.encode(content, add_special_tokens=False)

        # Truncate if necessary
        if len(content_tokens) > max_input_tokens:
            truncated_tokens = content_tokens[:max_input_tokens]
            truncated_content = self.tokenizer.decode(
                truncated_tokens, skip_special_tokens=True
            )
            logging.debug(
                f"Truncated input content from {len(content_tokens)} to {max_input_tokens} tokens."
            )
            return truncated_content
        else:
            return content  # No truncation needed

    def llm_tune(
        self, train_df, val_df, num_epochs=1, learning_rate=1e-5, save_steps=None
    ):
        """
        Fine-tune the model using the SFTTrainer with the custom data collator.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            save_steps (int, optional): How often to save checkpoints (steps). If None, defaults to epoch saving.

        Returns:
            Self: The trained model instance.
        """
        logging.info(
            "Starting fine-tuning with SFTTrainer and Softmax classification..."
        )
        logging.info(
            f"Training Config Snippet: epochs={num_epochs}, lr={learning_rate}, train_batch={self.per_device_train_batch_size}, grad_accum={self.gradient_accumulation_steps}"
        )

        # --- Save Experiment Config ---
        exp_config_path = self.log_dir / "experiment_config.yaml"
        try:
            config_to_save = {
                "model_url": self.model_url,
                "model_alias": self.model_alias,
                "max_length": self.max_length,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "save_steps": save_steps,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "per_device_eval_batch_size": self.per_device_eval_batch_size,
                "base_config": {
                    k: v for k, v in self.config.items() if k != "base_config"
                },  # Avoid nesting base_config
                "log_dir": str(self.log_dir),
                "start_timestamp": self.log_dir.name,  # Get timestamp from log_dir path
            }
            with open(exp_config_path, "w") as f:
                yaml.dump(
                    config_to_save,
                    f,
                    indent=2,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logging.info(f"Saved experiment configuration to {exp_config_path}")
        except Exception as e:
            logging.error(f"Failed to save experiment configuration: {e}")

        # --- Prepare Datasets ---
        drop_rate = self.config.get("drop_rate", 0.0)  # Feature dropout during training
        logging.info(f"Using feature drop rate during training: {drop_rate}")

        # Use multiprocessing? Be careful with large datasets in memory
        # num_proc = min(16, os.cpu_count()) # Limit processes
        num_proc = 1  # Safer default, increase if needed and memory allows

        logging.info("Preparing training data...")
        train_data = [
            self.generate_prompt(
                row, idx=idx, drop_rate=drop_rate
            )  # Pass index if needed by prompt gen
            for idx, row in tqdm(
                train_df.iterrows(), total=len(train_df), desc="Formatting train data"
            )
        ]
        logging.info("Preparing validation data...")
        val_data = [
            self.generate_prompt(
                row, idx=idx, drop_rate=0.0
            )  # No dropout for validation
            for idx, row in tqdm(
                val_df.iterrows(), total=len(val_df), desc="Formatting val data"
            )
        ]

        # Filter out potential None results (e.g., if generate_prompt skips some rows)
        train_data = [d for d in train_data if d is not None]
        val_data = [d for d in val_data if d is not None]
        logging.info(f"Actual training samples after formatting: {len(train_data)}")
        logging.info(f"Actual validation samples after formatting: {len(val_data)}")

        if not train_data:
            raise ValueError(
                "Training data is empty after formatting. Check generate_prompt method."
            )
        # Validation can be empty if no_validation=True, handle downstream

        train_dataset = Dataset.from_list(train_data)
        val_dataset = (
            Dataset.from_list(val_data) if val_data else None
        )  # Handle empty val

        # --- Data Collator for Classification ---
        from transformers import DataCollatorForLanguageModeling

        class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
            def __init__(
                self,
                *args,
                mlm: bool = False,
                ignore_index: int = -100,
                **kwargs,
            ):
                super().__init__(*args, mlm=mlm, **kwargs)
                self.ignore_index = ignore_index

                # know the fraud token id
                self.fraud_token_id = self.tokenizer.encode(
                    FRAUD_TOKEN, add_special_tokens=False
                )[0]
                self.not_fraud_token_id = self.tokenizer.encode(
                    NOT_FRAUD_TOKEN, add_special_tokens=False
                )[0]
                logging.info(f"Fraud token ID: {self.fraud_token_id}")
                logging.info(f"Not fraud token ID: {self.not_fraud_token_id}")

            def torch_call(
                self, examples: List[Union[List[int], Any, Dict[str, Any]]]
            ) -> Dict[str, Any]:
                batch = super().torch_call(examples)
                for i in range(len(examples)):
                    # Find the last non-padding token
                    last_token_idx = (
                        (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
                    )

                    # Set all labels to ignore_index except for the last token
                    batch["labels"][i, :last_token_idx] = self.ignore_index

                    # The old labels for the Yes and No tokens need to be mapped to 1 and 0
                    batch["labels"][i, last_token_idx] = (
                        1
                        if batch["labels"][i, last_token_idx] == self.fraud_token_id
                        else 0
                    )

                return batch

        collator = DataCollatorForLastTokenLM(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # --- Configure SFT Training ---
        # Determine save strategy
        save_steps = None  # Ensure save_steps is None by default
        if save_steps is None:
            save_strategy = "epoch"
            effective_save_steps = None
            logging.info("Saving checkpoints at the end of each epoch.")
        else:
            save_strategy = "steps"
            # Calculate save_steps based on percentage if requested (e.g., "10%")
            if isinstance(save_steps, str) and save_steps.endswith("%"):
                try:
                    percentage = float(save_steps.strip("%")) / 100
                    # Estimate total steps
                    total_train_samples = len(train_dataset)
                    if self.config.get("undersample", False):
                        num_fraud_est = int(
                            train_df["is_fraud"].sum()
                            * (
                                1.0
                                if not self.config.get("oversample", False)
                                else (len(train_df) / train_df["is_fraud"].sum()) / 2
                            )
                        )  # Rough estimate
                        total_train_samples = min(
                            len(train_dataset), max(100, num_fraud_est * 2)
                        )  # Use estimate, ensure > 0
                        logging.info(
                            f"Estimating undersampled training size for save_steps calculation: ~{total_train_samples}"
                        )

                    steps_per_epoch = total_train_samples / (
                        self.per_device_train_batch_size
                        * self.gradient_accumulation_steps
                        * torch.cuda.device_count()
                        if torch.cuda.is_available()
                        else 1
                    )
                    total_steps = steps_per_epoch * num_epochs
                    effective_save_steps = max(
                        1, int(total_steps * percentage)
                    )  # Save at least once
                except Exception as e:
                    logging.warning(
                        f"Could not parse save_steps percentage '{save_steps}': {e}. Defaulting to saving every 500 steps."
                    )
                    effective_save_steps = 500
            else:
                effective_save_steps = int(save_steps)  # Assume integer steps
            logging.info(f"Saving checkpoints every {effective_save_steps} steps.")

        sft_config = SFTConfig(
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            local_rank=self.lora_r,
            dataloader_num_workers=1,
            dataloader_persistent_workers=False,
            max_seq_length=self.max_length,
            output_dir=str(self.log_dir),
            logging_dir=str(self.log_dir),
            logging_steps=2,
            report_to="tensorboard",
            bf16=self.config.get("bf16", True),
            do_eval=self.config.get("do_eval", False),
            eval_strategy="no" if not self.config.get("do_eval", False) else "epoch",
            # constant lr
            # lr_scheduler_type="constant",
            save_strategy=save_strategy,
            # Save each 10 percent of the training dataset
            save_steps=save_steps,
            dataset_num_proc=16,
            max_steps=0
            if self.config.get("zero_shot", False)
            else 3
            if self.config.get("debug", False)
            else -1,
            seed=SEED_TRAINING,
            data_seed=SEED_TRAINING,
        )

        # Create a compute_metrics function to calculate F1 score
        def compute_metrics(eval_pred):
            """Calculate F1 score for validation"""
            logits, labels = eval_pred
            # Get predictions for the 2 classes (for last token only)
            predictions = np.argmax(logits[:, -1, :2], axis=-1)
            # Filter out ignored tokens (should only be the last token that matters)
            valid_indices = labels != -100
            true_labels = labels[valid_indices]
            pred_labels = predictions[valid_indices]

            # Calculate metrics with sklearn
            f1 = f1_score(
                true_labels, pred_labels, pos_label=FRAUD_LABEL_ID, zero_division=0
            )
            precision = precision_score(
                true_labels, pred_labels, pos_label=FRAUD_LABEL_ID, zero_division=0
            )
            recall = recall_score(
                true_labels, pred_labels, pos_label=FRAUD_LABEL_ID, zero_division=0
            )
            accuracy = accuracy_score(true_labels, pred_labels)

            return {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
            }

        # --- Initialize Trainer ---
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            data_collator=collator,
            # tokenizer=self.tokenizer,  # Pass tokenizer explicitly
            compute_metrics=compute_metrics,  # Enable compute_metrics to use F1 for validation
        )
        self.trainer = trainer  # Store trainer instance

        # --- Handle Undersampling ---
        undersample = self.config.get("undersample", False)
        if undersample:
            # Get labels (0/1) from the original train_df *before* oversampling or formatting
            # Need to ensure train_df used here matches the source of train_dataset
            try:
                # Use the original train_df passed to the function
                train_labels_for_undersampling = np.array(train_df["is_fraud"].tolist())
                # Add flag to trainer for callback use
                trainer.undersample = True
                trainer.train_dataset = PermutableUndersamplingDataset(
                    dataset=trainer.train_dataset,  # Wrap the Dataset object
                    train_labels=train_labels_for_undersampling,  # Pass the 0/1 labels
                    max_length=self.max_length,  # Pass necessary args
                    tokenizer=self.tokenizer,
                )
                logging.info("Dynamic undersampling enabled for training.")
            except Exception as e:
                logging.error(
                    f"Failed to set up undersampling: {e}. Proceeding without undersampling."
                )
                trainer.undersample = False
        else:
            trainer.undersample = False

        # --- Fast Generation / VLLM (Less relevant, keep for compatibility) ---
        if self.config.get("fast_generation", False):
            logging.info(
                "Setting fast_generation flag (may have limited use in this setup)."
            )
            trainer.fast_generation = True
            trainer.vllm_client = None  # VLLM likely not used here
        else:
            trainer.fast_generation = False

        # --- Add Custom Evaluation Callback ---
        evaluation_callback = EvaluationCallback(
            trainer=trainer,
            tokenizer=self.tokenizer,
            log_dir=self.log_dir,
            max_length=self.max_length,  # Needed for probability calculation
            run_eval_on_start=self.config.get("run_eval_on_start", True),
        )
        trainer.add_callback(evaluation_callback)
        logging.info("Added custom EvaluationCallback.")

        # --- Add Early Stopping Callback (Optional) ---
        if self.config.get("early_stopping_patience", None):
            from transformers import EarlyStoppingCallback

            patience = self.config["early_stopping_patience"]
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=self.config.get(
                    "early_stopping_threshold", 0.0
                ),
            )
            trainer.add_callback(early_stopping_callback)
            logging.info(f"Added EarlyStoppingCallback with patience={patience}.")

        # --- Start Training ---
        logging.info("Starting model training...")

        if self.config.get("early_test", False):
            pass
        else:
            if self.checkpoint_timestamp:
                last_checkpoint_dir = get_last_checkpoint(self.log_dir)
                logging.info(f"Last checkpint dir: {last_checkpoint_dir}")
                trainer.train(resume_from_checkpoint=last_checkpoint_dir)

                # Get the number of epoch from the checkpoint
                trainer_state = last_checkpoint_dir / "trainer_state.json"
                if trainer_state.exists():
                    with open(trainer_state, "r") as f:
                        trainer_state = json.load(f)
                    last_epoch = int(trainer_state["epoch"])
                    logging.info(f"Last epoch from checkpoint: {last_epoch}")
                    if last_epoch > 0:
                        # Loading last epoch parameters
                        param_file = (
                            self.log_dir / f"trainable_parameters_epoch_{last_epoch}.pt"
                        )
                        self.model.load_state_dict(torch.load(param_file), strict=False)
                        logging.info(f"Loaded  weights from checkpoint: {param_file}")

            else:
                trainer.train()

            logging.info("Instruction tuning complete.")

            # Save model
            self.save_model(filepath=self.log_dir)

        return self

    def evaluate(self, test_df, evaluation_threshold):
        """
        Evaluate the model on a test dataset using a specified threshold.

        Args:
            test_df (pd.DataFrame): Test data.
            evaluation_threshold (float, optional): The classification threshold to use.
                                                    If None, uses the best threshold found
                                                    during validation (or default 0.5 if unavailable).

        Returns:
            tuple: (metrics_dict, test_results_df) containing performance metrics and detailed predictions.
        """

        

        logging.info("--- Starting Final Evaluation on Test Set ---")

        if test_df is None or test_df.empty:
            logging.warning("Test dataframe is empty or None. Skipping evaluation.")
            return {}, pd.DataFrame()

        threshold = evaluation_threshold
        logging.info(f"Using provided evaluation threshold: {threshold:.4f}")

        # --- Prepare Model for Evaluation ---
        # Ensure the correct model state (e.g., best checkpoint loaded) is used.
        # This should ideally be handled by the calling function (train_and_evaluate)
        # before calling this method. Assume self.model is the desired one.
        self.model.eval()
        if hasattr(FastLanguageModel, "for_inference"):
            FastLanguageModel.for_inference(
                self.model
            )  # Prepare if using Unsloth helper

        # --- Prepare Test Dataset ---
        logging.info("Preparing test data for evaluation...")
        test_data = [
            self.generate_prompt(row, idx=idx, drop_rate=0.0)  # No dropout for testing
            for idx, row in tqdm(
                test_df.iterrows(), total=len(test_df), desc="Formatting test data"
            )
        ]
        test_data = [d for d in test_data if d is not None]

        if not test_data:
            logging.error("Test data is empty after formatting. Cannot evaluate.")
            return {}, pd.DataFrame()

        test_dataset = Dataset.from_list(test_data)

        # --- Collect Predictions and Probabilities ---
        true_labels_token = []  # Store original 'YES'/'NO'
        true_labels_id = []  # Store 0/1
        predicted_probs = []  # Store probability of FRAUD_LABEL_ID (class 1)
        ciks = []
        sics = []
        quarters = []
        misstatements = []  # Ground truth labels from data source (if different)

        # --- Re-use Batch Processing Logic from Callback ---
        def process_batch_for_eval(examples, model, tokenizer):
            # (Identical to the one in EvaluationCallback)
            batch_prompts = []
            batch_true_token = []
            batch_true_id = []
            batch_ciks = []
            batch_sics = []
            batch_quarters = []
            batch_misstatements = []

            for example in examples:
                input_text = example["text"]
                input_parts = input_text.split(COMPLETION_INSTRUCTION)
                prompt = input_parts[0]
                true_answer_token = example["answer"].strip()
                batch_prompts.append(prompt + COMPLETION_INSTRUCTION)
                batch_true_token.append(true_answer_token)
                batch_true_id.append(token_to_label_id(true_answer_token))
                batch_ciks.append(int(example["cik"]))
                batch_sics.append(example["sic"])
                batch_quarters.append(example["quarter"])
                batch_misstatements.append(example["misstatements"])

            probabilities = get_classification_probs(
                model, tokenizer, batch_prompts, self.max_length
            )
            fraud_probabilities = probabilities[:, FRAUD_LABEL_ID].cpu().tolist()

            return (
                batch_true_token,
                batch_true_id,
                fraud_probabilities,
                batch_ciks,
                batch_sics,
                batch_quarters,
                batch_misstatements,
            )

        # --- Evaluation Loop ---
        with torch.no_grad():
            batch_size = (
                self.per_device_eval_batch_size
            )  # Use configured eval batch size
            dataset_length = len(test_dataset)

            for i in tqdm(
                range(0, dataset_length, batch_size),
                desc=f"Evaluating on test set (batch size: {batch_size})",
            ):
                batch_examples = [
                    test_dataset[j]
                    for j in range(i, min(i + batch_size, dataset_length))
                ]
                if not batch_examples:
                    continue

                (
                    batch_true_token,
                    batch_true_id,
                    batch_fraud_probs,
                    batch_ciks,
                    batch_sics,
                    batch_quarters,
                    batch_misstatements,
                ) = process_batch_for_eval(batch_examples, self.model, self.tokenizer)

                # Collect results
                true_labels_token.extend(batch_true_token)
                true_labels_id.extend(batch_true_id)
                predicted_probs.extend(batch_fraud_probs)
                ciks.extend(batch_ciks)
                sics.extend(batch_sics)
                quarters.extend(batch_quarters)
                misstatements.extend(batch_misstatements)

        # --- Apply Threshold and Calculate Metrics ---
        predicted_labels_id = [
            FRAUD_LABEL_ID if prob >= threshold else NOT_FRAUD_LABEL_ID
            for prob in predicted_probs
        ]
        predicted_labels_token = [
            label_id_to_token(label_id) for label_id in predicted_labels_id
        ]

        accuracy = accuracy_score(true_labels_id, predicted_labels_id)
        precision = precision_score(
            true_labels_id,
            predicted_labels_id,
            pos_label=FRAUD_LABEL_ID,
            zero_division=0,
        )
        recall = recall_score(
            true_labels_id,
            predicted_labels_id,
            pos_label=FRAUD_LABEL_ID,
            zero_division=0,
        )
        f1 = f1_score(
            true_labels_id,
            predicted_labels_id,
            pos_label=FRAUD_LABEL_ID,
            zero_division=0,
        )
        macro_f1 = f1_score(
            true_labels_id, predicted_labels_id, average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            true_labels_id, predicted_labels_id, average="weighted", zero_division=0
        )
        auc_score = roc_auc_score(true_labels_id, predicted_probs)
        report = classification_report(
            true_labels_token,
            predicted_labels_token,
            labels=[FRAUD_TOKEN, NOT_FRAUD_TOKEN],
            target_names=["Fraud", "Not Fraud"],
            zero_division=0,
        )

        logging.info(
            f"--- Test Set Evaluation Results (Threshold: {threshold:.4f}) ---"
        )
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision (Fraud): {precision:.4f}")
        logging.info(f"Recall (Fraud): {recall:.4f}")
        logging.info(f"F1 Score (Fraud): {f1:.4f}")
        logging.info(f"Macro F1 Score: {macro_f1:.4f}")
        logging.info(f"Weighted F1 Score: {weighted_f1:.4f}")
        logging.info(f"AUC Score: {auc_score:.4f}")
        logging.info(f"Classification Report:\n{report}")

        # --- Save Metrics ---
        metrics_dict = {
            "threshold_used": float(threshold),
            "accuracy": float(accuracy),
            "precision_fraud": float(precision),
            "recall_fraud": float(recall),
            "f1_fraud": float(f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "auc_score": float(auc_score),
            "report": report,
            "num_test_samples": dataset_length,
            "num_fraud_samples_test": sum(true_labels_id),
        }
        test_metrics_path = os.path.join(self.log_dir, "test_metrics.json")
        try:
            with open(test_metrics_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            logging.info(f"Saved test metrics to {test_metrics_path}")
        except Exception as e:
            logging.error(f"Failed to save test metrics: {e}")

        # --- Save Detailed Predictions ---
        test_results_df = pd.DataFrame(
            {
                "cik": ciks,
                "sic": sics,  # Get SIC from original test_df if needed, ensure alignment
                "quarter": quarters,
                "y_true_token": true_labels_token,
                "y_true_id": true_labels_id,
                "y_pred_token": predicted_labels_token,
                "y_pred_id": predicted_labels_id,
                "fraud_probability": predicted_probs,
                "misstatements": misstatements,
            }
        )
        # Add original SIC back if columns match up
        if len(test_results_df) == len(test_df):
            test_results_df["sic"] = test_df["sic"].values
        else:
            logging.warning(
                "Length mismatch between predictions and original test_df. SIC column might be incorrect."
            )

        test_csv_path = os.path.join(self.log_dir, "test_predictions.csv")
        try:
            test_results_df.to_csv(test_csv_path, index=False)
            logging.info(f"Saved detailed test predictions to {test_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save test predictions CSV: {e}")

        # --- Log test metrics to TensorBoard ---
        if hasattr(self, "trainer"):
            tb_test_metrics = {
                f"test/{k}": v
                for k, v in metrics_dict.items()
                if isinstance(v, (int, float))
            }
            self.trainer.log(tb_test_metrics)
            logging.info("Logged test metrics to TensorBoard.")

        return metrics_dict, test_results_df

    def find_best_checkpoint(self):
        """
        Find the best checkpoint based on the validation F1 score ('f1_fraud_optimized') saved by the callback.

        Returns:
            tuple: (best_epoch, best_f1, best_threshold, best_checkpoint_dir_name) or (-1, -1, 0.5, None) if not found.
        """
        # best_f1 = -1.0
        best_auc = -1.0
        best_epoch = -1
        best_threshold_at_best_epoch = 0.5  # Default threshold
        best_checkpoint_name = None

        try:
            # 1. Find the epoch with the best F1 score from metrics files
            metric_files = list(self.log_dir.glob("metrics_epoch_*.json"))
            if not metric_files:
                logging.warning(
                    "No epoch metric files found. Cannot determine best checkpoint."
                )
                return -1, -1, 0.5, None

            for metric_file in metric_files:
                with open(metric_file, "r") as f:
                    metrics = json.load(f)
                    epoch = metrics.get("epoch", -1)
                    # Prioritize the F1 score calculated during optimization
                    current_auc = metrics.get("auc_score", metrics.get("auc_score", -1))

                    if current_auc > best_auc:
                        best_auc = current_auc
                        best_epoch = epoch
                        # Get the threshold associated with this best epoch
                        best_threshold_at_best_epoch = metrics.get(
                            "best_threshold", 0.5
                        )

            if best_epoch == -1:
                logging.warning(
                    "Could not find a valid epoch with F1 score in metric files."
                )
                return -1, -1, 0.5, None

            logging.info(
                f"Best validation AUC ({best_auc:.4f}) found at epoch {best_epoch} (Threshold: {best_threshold_at_best_epoch:.4f})."
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
                return best_epoch, best_auc, best_threshold_at_best_epoch, None

            found_checkpoint_dir = None
            latest_step_for_best_epoch = -1

            for chkpt_dir in checkpoint_dirs:
                state_file = chkpt_dir / "trainer_state.json"
                if state_file.exists():
                    with open(state_file, "r") as f:
                        state = json.load(f)
                        # Epoch in state is float, compare carefully
                        chkpt_epoch_float = state.get("epoch", -1.0)
                        chkpt_step = state.get("global_step", -1)

                        # Check if this checkpoint *finished* at or after the best epoch started
                        # SFTTrainer often saves *at the end* of a step/epoch.
                        # We want the checkpoint saved closest to the *end* of the best epoch.
                        # Compare floor(chkpt_epoch_float) to best_epoch
                        if (
                            chkpt_epoch_float is not None
                            and np.floor(chkpt_epoch_float) == best_epoch
                        ):
                            # If multiple checkpoints match the epoch, take the one with the highest step
                            if chkpt_step > latest_step_for_best_epoch:
                                latest_step_for_best_epoch = chkpt_step
                                found_checkpoint_dir = chkpt_dir

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
                best_threshold_at_best_epoch,
                best_checkpoint_name,
            )

        except Exception as e:
            logging.error(f"Error finding best checkpoint: {e}")
            return -1, -1, 0.5, None

    def get_best_checkpoint_dir(self):
        """Returns the full path to the best checkpoint directory."""
        _, _, _, best_checkpoint_name = self.find_best_checkpoint()
        if best_checkpoint_name:
            return self.log_dir / best_checkpoint_name
        else:
            return None

    def train_and_evaluate(
        self,
        train_path=None,  # Path to train CSV
        val_path=None,  # Path to validation CSV (optional, for CV)
        test_path=None,  # Path to test CSV
        num_epochs=None,
        learning_rate=None,
        save_steps=None,
        use_cv_data=False,  # Flag to load separate CV files
    ):
        """
        Load data, train the model, find the best checkpoint, and evaluate on the test set.

        Args:
            train_path (str/Path): Path to training data.
            val_path (str/Path, optional): Path to validation data (used if use_cv_data=True).
            test_path (str/Path): Path to test data.
            num_epochs (int, optional): Overrides config if provided.
            learning_rate (float, optional): Overrides config if provided.
            save_steps (int/str, optional): Overrides config if provided (e.g., 100 or "10%").
            use_cv_data (bool): If True, load train/val/test from separate paths.
                                If False, load train/test and split train into train/val.

        Returns:
            tuple: (final_metrics_dict, test_predictions_df) from the evaluate method.
        """
        # --- Load Data ---
        if use_cv_data:
            if val_path is None:
                raise ValueError("val_path must be provided when use_cv_data is True.")
            train_df, val_df, test_df = self.load_cv_data(
                train_path, val_path, test_path
            )
        else:
            train_df, val_df, test_df = self.load_data(
                train_path=train_path, test_path=test_path
            )

        # --- Check Model State ---
        # Model/tokenizer should already be loaded and PEFT applied by __init__
        if self.model is None or self.tokenizer is None:
            logging.error("Model or tokenizer not initialized. Cannot proceed.")
            # Re-initialize (optional, might hide issues)
            # self.load_model_and_tokenizer(...)
            # self.apply_peft_lora(...)
            # if self.model is None: # Check again
            return {}, pd.DataFrame()  # Return empty if init fails

        # --- Training ---
        # Override config params if provided
        current_num_epochs = (
            num_epochs
            if num_epochs is not None
            else self.config.get("num_train_epochs", 3)
        )  # Default to 3 epochs
        current_lr = (
            learning_rate
            if learning_rate is not None
            else self.config.get("learning_rate", 2e-5)
        )
        current_save_steps = (
            save_steps
            if save_steps is not None
            else self.config.get("save_steps", None)
        )  # Use config default (often None -> epoch saving)

        if self.config.get("zero_shot", False):
            logging.info("Zero-shot mode enabled. Skipping training.")
            # In zero-shot, we assume the model is already trained and just evaluate
            explain_mode = self.config.get("lxt_explain", False)
            predictions_exists = (
                self.log_dir / "test_predictions.csv").exists()
            
            if explain_mode :
            
                if not predictions_exists:
                    logging.warning(
                        "Lxt explanation requires test predictions. Runing evaluation first."
                    )
                    self.evaluate(
                        test_df,
                        evaluation_threshold=0.5)
                logging.info("Explaining predictions with Lxt")
                self.explain_test_predictions(test_df)
                return None, None
            else:
                return self.evaluate(test_df, evaluation_threshold=0.5)

        self.llm_tune(
            train_df,
            val_df,
            num_epochs=current_num_epochs,
            learning_rate=current_lr,
            save_steps=current_save_steps,
        )

        # --- Load Best Model ---
        (
            best_epoch,
            best_auc,
            best_threshold,
            best_checkpoint_dir_name,
        ) = self.find_best_checkpoint()

        evaluation_threshold = 0.5  # Default threshold

        # Load best checkpoint only if validation was performed and a best checkpoint was found
        if not self.config.get("no_validation", False) and best_checkpoint_dir_name:
            best_checkpoint_path = self.log_dir / best_checkpoint_dir_name

            logging.info(
                f"Loading best model weights from epoch {best_epoch} ({best_checkpoint_path}) with AUC={best_auc:.4f}."
            )

            # The model was already loaded in __init__, so we just need to load the state dict of the best checkpoint
            best_epoch_trainable_params = (
                self.log_dir / f"trainable_parameters_epoch_{best_epoch}.pt"
            )

            # Load the model state dict
            self.model.load_state_dict(
                torch.load(best_epoch_trainable_params), strict=False
            )
            logging.info(f"Loaded model state dict from {best_epoch_trainable_params}")

            self.model = FastLanguageModel.for_inference(self.model)
            evaluation_threshold = best_threshold

        elif self.config.get("no_validation", False):
            logging.warning(
                "`no_validation` was True. Evaluating with the final model state and default threshold 0.5."
            )
            # Keep evaluation_threshold = 0.5
            evaluation_threshold = 0.5
        else:  # No best checkpoint found even with validation
            evaluation_threshold = 0.5

        # --- Final Evaluation ---
        if test_df is not None and not test_df.empty:
            
            # In zero-shot, we assume the model is already trained and just evaluate
            explain_mode = self.config.get("lxt_explain", False)
            predictions_exists = (
                self.log_dir / "test_predictions.csv").exists()
            
            if explain_mode :
            
                if not predictions_exists:
                    logging.warning(
                        "Lxt explanation requires test predictions. Runing evaluation first."
                    )
                    self.evaluate(
                        test_df,
                        evaluation_threshold=0.5)
                logging.info("Explaining predictions with Lxt")
                self.explain_test_predictions(test_df)
                return None, None
            else:
                return self.evaluate(test_df, evaluation_threshold=0.5)
        else:
            logging.warning(
                "No test data provided or test data is empty. Skipping final evaluation."
            )
            return {}, pd.DataFrame()

    def save_model(self, filepath):
        """
        Save the PEFT adapters (LoRA weights) to a directory.

        Args:
            filepath (str or Path): Directory path to save the adapters.

        Returns:
            str: Path where the model adapters were saved.
        """
        save_path = str(filepath)  # Ensure it's a string
        logging.info(f"Saving PEFT model adapters to {save_path}")

        try:
            # Use the PEFT save method for adapters
            self.model.save_pretrained(save_path)

            # Optionally save tokenizer config too
            self.tokenizer.save_pretrained(save_path)

            logging.info(
                f"PEFT adapters and tokenizer saved successfully to {save_path}."
            )
            return save_path
        except Exception as e:
            logging.error(f"Failed to save PEFT model adapters: {e}")
            return None

    def format_financials(self, financials, drop_rate=0):
        """
        Format financial data dictionary into a string for the prompt.
        Handles dropping features, formatting numbers, and adding units.

        Args:
            financials (dict): Dictionary of financial features {feature_name: value}.
            drop_rate (float): Probability (0 to 1) of dropping each feature during formatting.

        Returns:
            str: Formatted string representation of the financials.
        """

        def display_financial_value(value):
            """Formats financial values for display."""
            if pd.isna(value):
                return "N/A"  # Handle missing values explicitly
            try:
                value = float(value)
                if value == 0:
                    return "0"
                elif abs(value) < 0.01 and abs(value) > 0:  # Small non-zero values
                    return f"{value:.2e}"
                elif abs(value) < 10:
                    return f"{value:.2f}"
                else:
                    # Format with commas, no decimal places for large numbers
                    return "{:,.0f}".format(value)
            except (ValueError, TypeError):
                return str(value)  # Return as string if not convertible to float

        # Filter out excluded features and invalid values (NaN, Inf)
        processed_financials = {}
        for k, v in financials.items():
            if (
                k not in EXCLUDED_FINANCIALS_FEATURES
                and pd.notna(v)
                and np.isfinite(v)
                and v != 0
            ):
                processed_financials[k] = v

        # Apply feature dropout if requested
        if drop_rate > 0:
            processed_financials = drop_random_keys(processed_financials, drop_rate)

        # Format the remaining features into strings
        financial_lines = []
        # Sort for consistency (optional)
        sorted_keys = sorted(processed_financials.keys())

        for key in sorted_keys:
            value = processed_financials[key]
            description = EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT.get(
                key, key
            )  # Use key if description missing
            unit = ""
            formatted_value = value  # Start with original value

            # Apply specific formatting based on feature type
            if key in PERCENTAGE_FEATURES:
                unit = "%"
                formatted_value = value * 100  # Convert ratio to percentage
            elif is_with_currency(key):
                unit = "$"  # Assume USD, adjust if needed

            # Format the number using the helper function
            display_value = display_financial_value(formatted_value)

            # Add unit prefix/suffix
            if unit == "$":
                display_str = f"{unit}{display_value}"
            elif unit == "%":
                display_str = f"{display_value}{unit}"
            else:
                display_str = display_value  # No unit

            financial_lines.append(f"- {description}: {display_str}")

        return (
            "\n".join(financial_lines)
            if financial_lines
            else "No financial data available."
        )

    def explain_test_predictions(self, test_df):
        """
        Generates LRP-based explanations for the test set using LXT, focusing
        only on the core content of the prompt.

        For each sample, it creates:
        1. A PDF with highlighted SENTENCE relevance for the core content.
        2. A CSV with top 1000 token attributions from the core content.
        3. A CSV with top 1000 word attributions from the core content.
        4. A CSV with top 1000 sentence attributions from the core content.
        """

        original_test_df = test_df.copy()
        original_test_df["quarter"] = original_test_df.apply(lambda r: f"{r['year']}{r['quarter']}", axis=1)

        test_pred_path = self.log_dir / "test_predictions.csv"
        if not test_pred_path.exists():
            logging.error(f"Test predictions file not found at {test_pred_path}. Run evaluation first.")
            return
        predictions_df = pd.read_csv(test_pred_path)
        df_to_explain = pd.merge(predictions_df, original_test_df, on=["cik", "quarter", "sic"], how="inner")

        # --- Setup Directories ---
        explain_dir = self.log_dir / "test_predictions_explanations"
        pdf_dir = explain_dir / "highlighted_pdf"
        top_tokens_dir = explain_dir / "top_tokens_attributions"
        top_words_dir = explain_dir / "top_words_attributions"
        top_sentences_dir = explain_dir / "top_sentences_attributions"
        for d in [explain_dir, pdf_dir, top_tokens_dir, top_words_dir, top_sentences_dir]:
            d.mkdir(exist_ok=True)
        logging.info(f"Saving explanations to: {explain_dir}")

        # --- Monkey-patch Model for LXT ---
        if not hasattr(self, '_lxt_patched'):
            logging.info("Applying LXT monkey patch to the model...")
            model_type = self.model.config.model_type
            try:
                module_name = f"transformers.models.{model_type}.modeling_{model_type}"
                modeling_module = importlib.import_module(module_name)
                monkey_patch(modeling_module, verbose=True)
                self._lxt_patched = True
            except (ImportError, KeyError) as e:
                logging.error(f"Failed to apply LXT monkey patch for {model_type}: {e}")
                return

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # --- Generate Explanations ---
        for _, row in tqdm(df_to_explain.iterrows(), total=len(df_to_explain), desc="Generating Explanations"):
            try:
                # Ensure generate_prompt returns 'core_content'
                example = self.generate_prompt(row, drop_rate=0.0)
                if not example: continue

                core_content = example["core_content"]
                prompt = example['text'].split(COMPLETION_INSTRUCTION)[0] + COMPLETION_INSTRUCTION
                
                input_ids = self.tokenizer(prompt, return_tensors="pt",
                                           add_special_tokens=True).input_ids.to(self.device)
                input_embeds = self.model.get_input_embeddings()(input_ids)
                input_embeds.requires_grad_(True)

                outputs = self.model(inputs_embeds=input_embeds, use_cache=False)
                
                y_pred_prob_csv = row['fraud_probability']
                y_pred_prob_softmax = torch.softmax(outputs.logits[0, -1, :], dim=-1)
                y_pred_prob_model = y_pred_prob_softmax[FRAUD_LABEL_ID].item()
                logging.info(f"CSV predicted prob: {y_pred_prob_csv}, Model predicted prob: {y_pred_prob_model}")
                
                last_token_logits = outputs.logits[0, -1, :2]
                last_token_logits[FRAUD_LABEL_ID].backward()

                grad = input_embeds.grad[0]
                norm_grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-10)
                full_relevance = (input_embeds * norm_grad).float().sum(-1).detach().cpu()[0]
                
                # --- Locate core content and slice the relevance scores ---
                core_content_ids = self.tokenizer(core_content, return_tensors="pt",
                                                add_special_tokens=True).input_ids
                full_ids_list = input_ids[0].tolist()
                core_ids_list = core_content_ids[0].tolist()
                core_ids_tokens_to_check = core_ids_list[5:100]
                start_idx = -1
                for i in range(len(full_ids_list) - len(core_ids_list) + 1):
                    if full_ids_list[i : i + len(core_ids_tokens_to_check)] == core_ids_tokens_to_check:
                        start_idx = i
                        break
                if start_idx == -1:
                    logging.warning(f"Could not locate core_content for CIK {row.get('cik', 'N/A')}. Skipping.")
                    continue
                start_idx = max(0, start_idx - 4)
                end_idx = start_idx + len(core_ids_list) - 1
                relevance = full_relevance[start_idx:end_idx]
                #normalized_relevance = relevance / (relevance.abs().max() + 1e-10)
                max_relevance = relevance.abs().max()
                relevance = relevance / (max_relevance + 1e-10) if max_relevance > 0 else relevance
                
                full_raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                raw_tokens = full_raw_tokens[start_idx:end_idx]
                
                clean_display_tokens = clean_tokens(raw_tokens)
                base_filename = f"{int(row['cik'])}_{row['quarter']}"

                # START: NEW CODE BLOCK - Prepare data for sentence-level PDF highlighting
                # This block computes the average relevance per sentence and creates a new
                # relevance tensor where each token has the score of the sentence it belongs to.
                sentence_level_token_relevance = np.zeros_like(relevance.numpy())
                current_sentence_start_index = 0
                current_sentence_text_for_check = ""

                for i, token_str in enumerate(raw_tokens):
                    decoded_token = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids([token_str]))
                    current_sentence_text_for_check += decoded_token
                    
                    # Check for sentence boundary (newline or end of tokens)
                    if '\n' in current_sentence_text_for_check or i == len(raw_tokens) - 1:
                        # Slice the tokens and relevances for the current sentence
                        sentence_relevance_slice = relevance[current_sentence_start_index : i + 1]
                        
                        # Calculate the average relevance for this sentence
                        avg_sentence_relevance = sentence_relevance_slice.mean().item()
                        
                        # Assign this average score to all tokens in the sentence
                        sentence_level_token_relevance[current_sentence_start_index : i + 1] = avg_sentence_relevance
                        
                        # Reset for the next sentence
                        current_sentence_start_index = i + 1
                        current_sentence_text_for_check = ""
                # END: NEW CODE BLOCK
                
                # --- Save PDF (now using sentence-level relevance) ---
                pdf_path = pdf_dir / f"{base_filename}_explanation.pdf"
                try:
                    # Pass the NEW relevance scores to the heatmap function
                    pdf_heatmap(clean_display_tokens, sentence_level_token_relevance, path=str(pdf_path), backend='xelatex')
                except Exception:
                    try:
                        pdf_heatmap(clean_display_tokens, sentence_level_token_relevance, path=str(pdf_path), backend='pdflatex')
                    except Exception as e_pdf:
                        logging.error(f"Failed to generate PDF for {base_filename}: {e_pdf}")

                # --- Save Top Token Attributions (using original per-token relevance) ---
                token_df = pd.DataFrame({'token': clean_display_tokens, 'relevance': relevance.numpy()})
                token_df['abs_relevance'] = token_df['relevance'].abs()
                token_df.sort_values(by='abs_relevance', ascending=False).head(1000)[['token', 'relevance']].to_csv(
                    top_tokens_dir / f"{base_filename}_token_attributions.csv", index=False
                )

                # --- Aggregate to Word-Level Attributions ---
                # (This logic remains unchanged)
                word_groups = []
                current_group = []
                for i, token_str in enumerate(raw_tokens):
                    if i > 0 and token_str.startswith('Ġ'):
                        if current_group: word_groups.append(current_group)
                        current_group = []
                    current_group.append((token_str, relevance[i].item()))
                if current_group: word_groups.append(current_group)

                aggregated_words = []
                for group in word_groups:
                    word_tokens, word_relevances = [item[0] for item in group], [item[1] for item in group]
                    full_word = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(word_tokens))
                    avg_relevance = np.mean(word_relevances)
                    aggregated_words.append({'word': full_word.strip(), 'relevance': avg_relevance})
                
                if aggregated_words:
                    word_df = pd.DataFrame(aggregated_words)
                    word_df['abs_relevance'] = word_df['relevance'].abs()
                    word_df.sort_values(by='abs_relevance', ascending=False).head(1000)[['word', 'relevance']].to_csv(
                        top_words_dir / f"{base_filename}_word_attributions.csv", index=False
                    )

                # --- Aggregate to Sentence-Level Attributions for CSV ---
                # (This logic remains unchanged)
                sentence_relevances = []
                current_sentence_text, current_sentence_rels = "", []
                for i, token_str in enumerate(raw_tokens):
                    decoded_token = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids([token_str]))
                    current_sentence_text += decoded_token
                    current_sentence_rels.append(relevance[i].item())
                    if '\n' in current_sentence_text or i == len(raw_tokens) - 1:
                        if current_sentence_rels:
                            clean_text = current_sentence_text.replace('\n', ' ').strip()
                            if clean_text:
                                avg_relevance = np.mean(current_sentence_rels)
                                sentence_relevances.append({'sentence': clean_text, 'relevance': avg_relevance})
                        current_sentence_text, current_sentence_rels = "", []
                
                if sentence_relevances:
                    sentence_df = pd.DataFrame(sentence_relevances)
                    sentence_df['abs_relevance'] = sentence_df['relevance'].abs()
                    sentence_df.sort_values(by='abs_relevance', ascending=False).head(1000)[['sentence', 'relevance']].to_csv(
                        top_sentences_dir / f"{base_filename}_sentence_attributions.csv", index=False
                    )

            except Exception as e:
                logging.error(f"Failed to explain sample CIK {row.get('cik', 'N/A')}: {e}", exc_info=True)

        logging.info("--- Explanation generation complete ---")
            
    def format_financials_dechow(self, financials, drop_rate=0):
        """
        Format financial data dictionary into a string for the prompt.
        Handles dropping features, formatting numbers, and adding units.

        Args:
            financials (dict): Dictionary of financial features {feature_name: value}.
            drop_rate (float): Probability (0 to 1) of dropping each feature during formatting.

        Returns:
            str: Formatted string representation of the financials.
        """

        def display_financial_value(value):
            """Formats financial values for display."""
            if pd.isna(value):
                return "N/A"  # Handle missing values explicitly
            try:
                value = float(value)
                if value == 0:
                    return "0"
                elif abs(value) < 0.01 and abs(value) > 0:  # Small non-zero values
                    return f"{value:.2e}"
                elif abs(value) < 10:
                    return f"{value:.2f}"
                else:
                    # Format with commas, no decimal places for large numbers
                    return "{:,.0f}".format(value)
            except (ValueError, TypeError):
                return str(value)  # Return as string if not convertible to float

        # Filter out excluded features and invalid values (NaN, Inf)
        processed_financials = {}
        for k, v in financials.items():
            if pd.notna(v) and np.isfinite(v) and v != 0:
                processed_financials[k] = v

        # Apply feature dropout if requested
        if drop_rate > 0:
            processed_financials = drop_random_keys(processed_financials, drop_rate)

        # Format the remaining features into strings
        financial_lines = []
        # Sort for consistency (optional)
        sorted_keys = sorted(processed_financials.keys())

        for key in sorted_keys:
            value = processed_financials[key]
            description = DECHOW_FEATURES_SHORT_DESCRIPTIONS.get(
                key, key
            )  # Use key if description missing
            unit = ""
            formatted_value = value  # Start with original value

            # Format the number using the helper function
            display_value = display_financial_value(formatted_value)

            # Add unit prefix/suffix
            if unit == "$":
                display_str = f"{unit}{display_value}"
            elif unit == "%":
                display_str = f"{display_value}{unit}"
            else:
                display_str = display_value  # No unit

            financial_lines.append(f"- {description}: {display_str}")

        return (
            "\n".join(financial_lines)
            if financial_lines
            else "No financial data available."
        )


# --- Base Training Function ---


def llm_softmax_train_and_evaluate_base_model(
    model_class,
    config=None,
    train_path=None,
    val_path=None,  # Added for CV support
    test_path=None,
    use_cv_data=False,  # Flag for CV data loading
):
    """
    Base function to train and evaluate an LLM softmax fraud classifier.

    Args:
        model_class: The specific classifier class (subclass of LLMClassifierSoftmax).
        config (dict): Configuration dictionary for the model.
        train_path (Path/str): Path to training data CSV.
        val_path (Path/str, optional): Path to validation data CSV (if use_cv_data=True).
        test_path (Path/str): Path to test data CSV.
        use_cv_data (bool): Whether to load train/val/test from separate files.

    Returns:
        tuple: (trained_model_instance, final_metrics_dict, test_predictions_df)
    """
    if config is None:
        raise ValueError("Configuration dictionary must be provided.")
    if train_path is None or test_path is None:
        raise ValueError("train_path and test_path must be provided.")
    if use_cv_data and val_path is None:
        raise ValueError("val_path must be provided when use_cv_data is True.")

    # --- Set Seed ---
    # Set seed early, before model initialization if it affects weights
    # Note: Unsloth's PEFT application also uses a seed.
    seed = config.get("seed", SEED_TRAINING)  # Allow overriding seed via config
    logging.info(f"Setting random seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Potentially add deterministic algorithms (might impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # --- Initialize the Model ---
    # Model initialization handles logging setup internally now
    model = model_class(config=config)
    logging.info(f"Initialized model: {model.model_alias}")

    # --- Train and Evaluate ---
    logging.info(f"Starting training and evaluation for {model.model_alias}...")
    final_metrics, test_predictions = model.train_and_evaluate(
        train_path=train_path,
        val_path=val_path,  # Pass val_path
        test_path=test_path,
        num_epochs=config.get(
            "num_train_epochs", None
        ),  # Allow override, fallback to config
        learning_rate=config.get(
            "learning_rate", None
        ),  # Allow override, fallback to config
        save_steps=config.get("save_steps", None),  # Allow override, fallback to config
        use_cv_data=use_cv_data,  # Pass the flag
    )

    logging.info(f"{model.model_alias} training and evaluation completed.")

    # Optionally clean up GPU memory if running multiple models sequentially
    # del model
    # torch.cuda.empty_cache()

    return model, final_metrics, test_predictions
