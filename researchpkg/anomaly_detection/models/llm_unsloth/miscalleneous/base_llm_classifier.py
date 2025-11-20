import json
import logging
import math
import os
import random
import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from joblib import Parallel, delayed
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    PREPROCESSED_PATH,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.utils import (
    calculate_cik_macro_f1,
    drop_random_keys,
    get_last_checkpoint,
    get_tokenizer_completion_instruction,
    get_tokenizer_start_instruction,
    get_train_test_splitter,
    llm_fast_generate,
    llm_generate,
    llm_vllm_generate,
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
LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
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

        is_fraud = train_labels

        # Indices for fraud and non-fraud
        self.fraud_indices = np.where(is_fraud == 1)[0]
        self.non_fraud_indices = np.where(is_fraud == 0)[0]
        self.num_fraud = len(self.fraud_indices)

        assert self.num_fraud > 0, "No fraud cases found in the dataset"
        assert self.num_fraud < len(
            self.non_fraud_indices
        ), "More fraud cases than non-fraud cases"

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
        undersampling_ratio = self.num_fraud / len(self.non_fraud_indices)
        logging.info(
            f"Undersampling ratio: {undersampling_ratio:.4f} (keeping approximately {undersampling_ratio:.1%} of non-fraud cases)"
        )

        # Group data by SIC and year
        groups = {}
        for idx in range(len(self.whole_dataset)):
            sic = self.sic_values[idx]
            group_key = str(sic)

            if group_key not in groups:
                groups[group_key] = {"fraud": [], "non_fraud": []}

            if idx in self.fraud_indices:
                groups[group_key]["fraud"].append(idx)
            else:
                groups[group_key]["non_fraud"].append(idx)

        # Perform stratified sampling within each group
        selected_indices = []
        fraud_count = 0
        non_fraud_count = 0

        for sic, group_data in groups.items():
            # Take all fraud cases from this group
            selected_indices.extend(group_data["fraud"])
            fraud_count += len(group_data["fraud"])

            # Sample non-fraud cases according to the ratio
            if len(group_data["non_fraud"]) > 0:
                num_non_fraud_to_sample = max(
                    1, int(round(len(group_data["non_fraud"]) * undersampling_ratio))
                )

                # If we need more than available, take all of them
                if num_non_fraud_to_sample >= len(group_data["non_fraud"]):
                    selected_non_fraud = group_data["non_fraud"]
                else:
                    # Otherwise, sample without replacement
                    selected_non_fraud = np.random.choice(
                        group_data["non_fraud"],
                        size=num_non_fraud_to_sample,
                        replace=False,
                    )

                selected_indices.extend(selected_non_fraud)
                non_fraud_count += len(selected_non_fraud)

        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)

        # Log statistics about the undersampled dataset
        logging.info(
            f"Stratified undersampling: {fraud_count} fraud samples and {non_fraud_count} non-fraud samples"
        )

        if len(selected_indices) > 0:
            fraud_ratio = fraud_count / len(selected_indices)
            logging.info(f"Fraud ratio after undersampling: {fraud_ratio:.2%}")

        # Calculate and log distribution by industry
        industry_distribution = {}
        for i, idx in enumerate(selected_indices):
            sic = self.sic_values[idx]
            if sic not in industry_distribution:
                industry_distribution[sic] = {"total": 0, "fraud": 0}

            industry_distribution[sic]["total"] += 1
            if idx in self.fraud_indices:
                industry_distribution[sic]["fraud"] += 1

        logging.info("Industry distribution after stratified undersampling:")
        for sic, stats in industry_distribution.items():
            if stats["total"] > 0:
                fraud_pct = stats["fraud"] / stats["total"] * 100
                logging.info(
                    f"  SIC {sic}: {stats['total']} samples, {stats['fraud']} fraud ({fraud_pct:.1f}%)"
                )

        return selected_indices

    def permute(self):
        """
        Permute the dataset by stratified undersampling and shuffling the order of examples.
        This maintains industry and time period distributions.
        """
        combined_indices = self._generate_permutation()
        self._dataset = self.whole_dataset.select(combined_indices)
        logging.info(f"Dataset permuted with {len(self._dataset)} examples")

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        """
        if idx >= len(self._dataset):
            raise ValueError("Index out of range")

        # Return the permuted example
        return self._dataset[idx]


class EvaluationCallback(TrainerCallback):
    """Custom callback for evaluation at the end of each epoch"""

    def __init__(
        self,
        trainer,
        tokenizer,
        log_dir,
        max_length,
        max_new_tokens,
        run_eval_on_start=True,
    ):
        self.trainer = trainer
        self.train_dataset = trainer.train_dataset
        self.eval_dataset = trainer.eval_dataset

        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.run_eval_on_start = run_eval_on_start

        self.lock = threading.Lock()

        if hasattr(trainer, "undersample"):
            self.undersample = trainer.undersample
        else:
            self.undersample = False

        if hasattr(trainer, "fast_generation"):
            self.fast_generation = trainer.fast_generation
            self.current_lora_request = None
        else:
            self.fast_generation = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.undersample:
            # Updating the train_dataloader with dynamic undersampling
            self.train_dataset.permute()

        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.epoch == 0 and self.run_eval_on_start:
            logging.info("Running evaluation at the beginning of training...")
            self.on_epoch_end(args, state, control, **kwargs)
        return super().on_train_begin(args, state, control, **kwargs)

    def extract_answer_prediction(self, prediction_text):
        return prediction_text  # for this model, we don't need to extract the answer

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation at the end of each epoch."""
        with torch.no_grad():
            model = self.trainer.model

            vllm_client = None
            if hasattr(self.trainer, "vllm_client"):
                # If using VLLM, we need to set the model to eval mode
                vllm_client = self.trainer.vllm_client
            elif self.fast_generation:
                # We will load the lora weights only if we are not using vllm
                self.current_lora_request = self.model.load_lora(
                    get_last_checkpoint(self.log_dir), load_tensors=False
                )
                logging.info(
                    f"Fast Inference : Loading lora weights from {self.current_lora_request}"
                )
            else:
                logging.info(
                    "No VLLM client and no fast inference, using default model"
                )
                FastLanguageModel.for_inference(model)

            # Extract true labels and get predictions
            true_labels = []
            predicted_labels = []
            ciks = []
            sics = []
            quarters = []
            glabels = []

            # Process a single example function for parallel processing
            def process_batch(examples, model, tokenizer, vllm_client=None):
                batch_results = []

                # Prepare batch inputs
                batch_prompts = []

                for example in examples:
                    # Extract data from example
                    input_text = example["text"]
                    input_parts = input_text.split(
                        self.tokenizer.completion_instruction
                    )
                    prompt = input_parts[0]
                    true_label = input_parts[1].strip()
                    true_label = "Not Fraud" if "Not" in true_label else "Fraud"

                    # Generate prediction
                    chat_prompt = prompt + self.tokenizer.completion_instruction

                    batch_prompts.append(chat_prompt)

                # Generate predictions for the batch
                if vllm_client:
                    predictions = llm_vllm_generate(
                        model,
                        tokenizer,
                        vllm_client,
                        batch_prompts,
                        self.max_length,
                        self.max_new_tokens,
                    )
                elif self.fast_generation:
                    predictions = llm_fast_generate(
                        model,
                        batch_prompts,
                        self.max_length,
                        self.max_new_tokens,
                        self.current_lora_request,
                    )
                else:
                    predictions = llm_generate(
                        model,
                        tokenizer,
                        batch_prompts,
                        self.max_length,
                        self.max_new_tokens,
                    )

                # Process each prediction
                for i, prediction_text in enumerate(predictions):
                    example = examples[i]
                    answer = example["answer"]

                    # Extract the answer from the XML format
                    prediction_text = prediction_text.split(
                        self.tokenizer.completion_instruction
                    )[-1].strip()
                    extracted_answer = self.extract_answer_prediction(prediction_text)

                    # Clean up prediction to get just the label
                    if extracted_answer not in ["Fraud", "Not Fraud"]:
                        if (
                            "fraud" in extracted_answer.lower()
                            and "not fraud" not in extracted_answer.lower()
                        ):
                            extracted_answer = "Fraud"
                        else:
                            extracted_answer = "Not Fraud"

                    batch_results.append(
                        {
                            "true_label": answer,
                            "predicted_label": extracted_answer,
                            "prediction": prediction_text,
                            "cik": int(example["cik"]),
                            "sic": example["sic"],
                            "quarter": example["quarter"],
                            "glabels": example["glabels"],
                        }
                    )

                return batch_results

            with torch.no_grad():
                vllm_client = None
                try:
                    vllm_client = self.trainer.vllm_client
                except Exception as e:
                    logging.error(f"No vllm client found in GRPO callback: {e}")

                # Process in batches
                batch_size = self.trainer.args.per_device_eval_batch_size
                dataset_length = len(self.trainer.eval_dataset)

                for i in tqdm(
                    range(0, dataset_length, batch_size),
                    desc=f"Evaluation at epoch {int(state.epoch)} (batch size: {batch_size})",
                ):
                    # Get batch of examples
                    batch_examples = [
                        self.trainer.eval_dataset[j]
                        for j in range(i, min(i + batch_size, dataset_length))
                    ]

                    # Process the batch
                    batch_results = process_batch(
                        batch_examples, self.trainer.model, self.tokenizer, vllm_client
                    )

                    # Collect results
                    for result, batch_example in zip(batch_results, batch_examples):
                        true_labels.append(result["true_label"])
                        predicted_labels.append(result["predicted_label"])
                        ciks.append(result["cik"])
                        sics.append(result["sic"])
                        quarters.append(result["quarter"])
                        glabels.append(result["glabels"])

                        # Log sample predictions occasionally
                        if state.epoch == 0 or random.random() < 0.001:
                            logging.info(f"Prompt: {batch_example['text']}")
                            logging.info(f"Prediction: {result['prediction']}")
                            logging.info(f"Answer: {batch_example['answer']}")
                            logging.info(
                                f"True: {result['true_label']}, Pred: {result['predicted_label']}"
                            )
                            logging.info(
                                f"CIK: {result['cik']}, SIC: {result['sic']}, Quarter: {result['quarter']}"
                            )

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(
                true_labels, predicted_labels, pos_label="Fraud"
            )
            recall = recall_score(true_labels, predicted_labels, pos_label="Fraud")
            f1 = f1_score(true_labels, predicted_labels, pos_label="Fraud")

            cik_macro_f1_macro = calculate_cik_macro_f1(
                true_labels, predicted_labels, ciks
            )
            cik_macro_f1_weighted = calculate_cik_macro_f1(
                true_labels, predicted_labels, ciks, average="weighted"
            )

            macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
            weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
            report = classification_report(true_labels, predicted_labels)

            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Macro F1 Score: {macro_f1:.4f}")
            logging.info(f"Weighted F1 Score: {weighted_f1:.4f}")
            logging.info(f"CIK Macro F1 Macro Score: {cik_macro_f1_macro:.4f}")
            logging.info(f"CIK Macro F1 Weighted Score: {cik_macro_f1_weighted:.4f}")
            logging.info(f"Classification Report:\n{report}")

            # Save metrics to a file
            with open(
                os.path.join(self.log_dir, f"metrics_epoch_{int(state.epoch)}.json"),
                "w",
            ) as f:
                json.dump(
                    {
                        "epoch": int(state.epoch),
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "macro_f1": float(macro_f1),
                        "weighted_f1": float(weighted_f1),
                        "cik_macro_f1_macro": float(cik_macro_f1_macro),
                        "cik_macro_f1_weighted": float(cik_macro_f1_weighted),
                        "report": report,
                    },
                    f,
                    indent=2,
                )

            tb_metrics_dict = {
                "epoch": int(state.epoch),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "cik_macro_f1_macro": float(cik_macro_f1_macro),
                "cik_macro_f1_weighted": float(cik_macro_f1_weighted),
                "step": int(state.global_step),
            }
            self.trainer.log_metrics("eval", tb_metrics_dict)

            # Save detailed test predictions to CSV
            val_results_df = pd.DataFrame(
                {
                    "cik": ciks,
                    "sic": sics,
                    "quarter": quarters,
                    "y_true": true_labels,
                    "y_pred": predicted_labels,
                    "glabels": glabels,
                }
            )

            val_csv_path = os.path.join(
                self.log_dir, f"val_predictions_epoch_{int(state.epoch)}.csv"
            )
            val_results_df.to_csv(val_csv_path, index=False)
            logging.info(f"Saved detailed test predictions to {val_csv_path}")

            # Keep model in train mode for further training
            model.train()

            if vllm_client:
                FastLanguageModel.for_training(model)
            return control


class BaseLLMFraudClassifier:
    """
    Base LLM classifier for fraud detection that abstracts common functionality.

    This class provides base methods for loading, training, evaluating, and making predictions
    with LLM-based models for fraud detection. Subclasses should implement data-specific
    methods like load_data and generate_prompt.
    """

    def __init__(
        self,
        config,
    ):
        """
        Initialize the base LLM classifier.

        Args:
            model_url (str): Name of the LLM model to use (from Hugging Face).
            model_alias (str): Alias for the model to use in logging and saving.
            lora_r (int): Lora rank.
            lora_alpha (int): Lora alpha.
            max_length (int): Maximum context length for the model.
            config (dict, optional): Configuration dictionary to override defaults.
        """
        assert config is not None, "Configuration should be provided"
        self.model_url = config["model_url"]
        self.model_alias = config.get("model_alias", self.model_url.split("/")[-1])
        self.lora_r = config["lora_r"]
        self.lora_alpha = config["lora_alpha"]
        self.max_length = config["max_context"]

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

        # Set up experiment directory
        self.experiments_dir = config.get(
            "experiments_dir", EXPERIMENTS_DIR / f"llm_fraud_classifier"
        )

        if not self.experiments_dir.exists():
            self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # If we have a checkpoint, use that timestamp for the log directory
        fold_id = self.config["fold_id"]
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
            logFile=self.log_file,
            logLevel=logging.INFO,
        )

        if self.checkpoint_timestamp:
            logging.info(f"Continuing training from checkpoint: {self.log_dir}")
        else:
            logging.info(f"Starting new training session: {self.log_dir}")

        self.load_model_and_tokenizer(
            fast_inference=self.config.get("fast_generation", False)
        )
        lora_target_modules = self.config.get(
            "lora_target_modules", LORA_TARGET_MODULES
        )
        logging.info(f"LoRA target modules: {lora_target_modules}")
        self.apply_peft_lora(target_modules=lora_target_modules)

    def load_model_and_tokenizer(self, fast_inference=False):
        """
        Load the LLM model and tokenizer from Hugging Face.
        """
        logging.info(f"Loading {self.model_alias} model: {self.model_url}")
        logging.info(f"Maximum context length: {self.max_length}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_url,
            max_seq_length=self.max_length,
            dtype=None,  # None for auto detection
            full_finetuning=False,
            load_in_4bit=self.config.get(
                "load_in_4bit", True
            ),  # Use 4bit quantization to reduce memory usage
            fast_inference=fast_inference,  # Use fast inference mode
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.7),
        )
        self.model.config.use_cache = False  # Reduces memory usagef

        try:
            self.model.config.text_config.use_cache = False  # Reduces memory usage
        except:
            logging.info("No text_config attribute in the model")

        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        self.tokenizer.completion_instruction = get_tokenizer_completion_instruction(
            self.tokenizer
        )
        self.tokenizer.start_instruction = get_tokenizer_start_instruction(
            self.tokenizer
        )

        # if "deepseek" in self.model_url.lower():
        #     self.tokenizer.start_instruction = "<|object_ref_start>"
        #     self.tokenizer.completion_instruction = "<|im_end|>"

        if "QwQ-32B".lower() in self.model_url.lower():
            self.tokenizer.start_instruction = "<|im_start|>system\n"
            self.tokenizer.completion_instruction = "<im_start|>assistant\n"

        logging.info(f"Tokenizer start instruction: {self.tokenizer.start_instruction}")
        logging.info(
            f"Tokenizer completion instruction: {self.tokenizer.completion_instruction}"
        )
        logging.info(f"{self.model_alias} model loaded successfully.")

    def apply_peft_lora(self, target_modules=None):
        """
        Apply PEFT and LoRA to the model.

        Args:
            target_modules (list, optional): List of modules to apply LoRA to.
        """
        if target_modules is None:
            target_modules = LORA_TARGET_MODULES

        logging.info("Applying PEFT and LoRA...")
        layers_to_transform = self.config.get("layers_to_transform", None)

        if not layers_to_transform:
            num_layers_to_transform = self.config.get("num_layers_to_finetune", None)
            if num_layers_to_transform:

                assert (
                    "num_model_layers" in self.config
                ), "number_of_model_layers not found in model config"
                num_model_layers = self.config["num_model_layers"]
                logging.info(f"Number of model layers: {num_model_layers}")
                if num_layers_to_transform > num_model_layers:
                    raise ValueError(
                        f"num_layers_to_finetune {num_layers_to_transform} is greater than the number of model layers {num_model_layers}"
                    )

                layers_to_transform = list(
                    range(num_model_layers - num_layers_to_transform, num_model_layers)
                )

            else:
                layers_to_transform = None

        logging.info(
            f"Layers to transform by lora: {layers_to_transform if layers_to_transform else 'All layers'}"
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_r,
            target_modules=target_modules,
            lora_alpha=self.lora_alpha,
            # finetune_vision_layers=False,  # Turn off for just text
            # finetune_language_layers=True,
            layers_to_transform=layers_to_transform,
            # finetune_attention_modules=self.config.get("finetune_attention_modules", True),
            # finetune_mlp_modules=self.config.get("finetune_mlp_modules", True),
            lora_dropout=self.config.get("lora_dropout", 0),
            bias="none",  # Bias = "none" is currently optimized
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        logging.info("PEFT and LoRA applied successfully.")
        self.model.print_trainable_parameters()

    def load_data(self, train_path=None, test_path=None):
        """
        Load train and test datasets.

        Args:
            train_path (Path, optional): Path to training data.
            test_path (Path, optional): Path to test data.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logging.info(f"Loading tra  in data from {train_path}")
        train_df = pd.read_csv(train_path)
        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Process data with custom method (to be overridden by subclasses)
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        if self.config.get("no_validation", False):
            logging.info(
                "No validation set,Train on the whole training data without split"
            )
            val_df = train_df[
                :1
            ]  # Dummy validation set for compatibility but no validation in practice
        else:
            splitter = get_train_test_splitter(self.config)
            train_df, val_df = splitter(train_df, test_size=0.1)

        if self.config.get("oversample", False):
            logging.info("Oversampling fraud cases in training data...")
            train_df = self.oversample_fraud_cases(train_df)

        logging.info("Train data size: %d", len(train_df))
        logging.info("Validation data size: %d", len(val_df))
        logging.info("Test data size: %d", len(test_df))

        return train_df, val_df, test_df

    def load_cv_data(self, train_path, val_path, test_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        train_df = self._process_loaded_data(train_df)
        val_df = self._process_loaded_data(val_df)
        test_df = self._process_loaded_data(test_df)

        if self.config.get("oversample", False):
            logging.info("Oversampling fraud cases in training data...")
            train_df = self.oversample_fraud_cases(train_df)
        logging.info("Train data size: %d", len(train_df))
        logging.info("Validation data size: %d", len(val_df))
        logging.info("Test data size: %d", len(test_df))

        return train_df, val_df, test_df

    def oversample_fraud_cases(self, df):
        """
        Oversample fraud cases in the dataset.

        Args:
            df (pd.DataFrame): The dataframe to oversample.

        Returns:
            pd.DataFrame: The oversampled dataframe.
        """
        fraud_df = df[df["is_fraud"] == True]
        non_fraud_df = df[df["is_fraud"] == False]

        # Oversample fraud cases
        oversampled_fraud_df = fraud_df.sample(
            n=len(non_fraud_df), replace=True, random_state=SEED_TRAINING
        )

        # Combine oversampled fraud cases with non-fraud cases
        oversampled_df = pd.concat(
            [oversampled_fraud_df, non_fraud_df], ignore_index=True
        )
        logging.info(
            f"Oversampled fraud cases from {len(fraud_df)} to {len(oversampled_fraud_df)}"
        )

        return oversampled_df

    def _process_loaded_data(self, df):
        """
        Process loaded data before creating train/val split.
        Override this method in subclasses to provide specific data processing.

        Args:
            df (pd.DataFrame): The loaded dataframe

        Returns:
            pd.DataFrame: Processed dataframe
        """
        return df

    def generate_prompt(self, row, idx=None, **kwargs):
        """
        Generate and tokenize the prompt for training.

        Args:
            row (pd.Series): Row from the DataFrame.
            idx (int, optional): Index of the example.

        Returns:
            dict: Dictionary with prompts and metadata for training.
        """
        raise NotImplementedError("Subclasses must implement generate_prompt method")

    def truncate_and_format_prompt(self, content):
        """
        Ensure the prompt is within the model's context length.

        Args:
            content: The content to include in the prompt.

        Returns:
            str: Truncated and formatted prompt
        """
        raise NotImplementedError(
            "Subclasses must implement truncate_and_format_prompt method"
        )

    def llm_tune(
        self, train_df, val_df, num_epochs=1, learning_rate=1e-5, save_steps=100
    ):
        """
        Train the model using instruction tuning.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            save_steps (int): How often to save checkpoints.

        Returns:
            Self: The trained model.
        """
        logging.info("Starting instruction tuning with PEFT Trainer...")
        logging.info(f"Config :{self.config}")

        # Save experiment config at the beginning of training
        config = {
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
            "base_config": self.config,
        }
        with open(self.log_dir / "experiment_config.yaml", "w") as f:
            yaml.dump(config, f, indent=2)

        # Generate prompts for train and validation data
        drop_rate = self.config.get("drop_rate", None)

        train_data = [
            self.generate_prompt(row, drop_rate=drop_rate)
            for _, row in tqdm(train_df.iterrows(), desc="Preparing training data")
        ]

        val_data = [
            self.generate_prompt(row, idx)
            for idx, (_, row) in enumerate(
                tqdm(val_df.iterrows(), desc="Preparing validation data")
            )
        ]

        # Filter out any None results (e.g., missing data)
        train_data = [d for d in train_data if d is not None]
        val_data = [d for d in val_data if d is not None]

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        # Standardize data formats for unsloth
        from unsloth.chat_templates import standardize_data_formats

        train_dataset = standardize_data_formats(train_dataset)
        val_dataset = standardize_data_formats(val_dataset)

        save_strategy = "steps"
        save_steps = int((len(train_dataset) // self.per_device_train_batch_size) * 0.1)

        is_70b = "70B" in self.model_url
        if self.config.get("undersample", False) and not is_70b:
            save_strategy = "epoch"
            save_steps = None
        # Configure SFT training
        sft_config = SFTConfig(
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            local_rank=self.lora_r,
            dataloader_num_workers=min(1, os.cpu_count()),
            dataloader_persistent_workers=False,
            max_seq_length=self.max_length,
            output_dir=str(self.log_dir),
            logging_dir=str(self.log_dir),
            logging_steps=2,
            report_to="tensorboard",
            bf16=self.config.get("bf16", True),
            do_eval=self.config.get("do_eval", False),
            eval_strategy="no" if not self.config.get("do_eval", False) else "epoch",
            save_strategy=save_strategy,
            # Save each 10 percent of the training dataset
            save_steps=save_steps, 
            dataset_num_proc=16,
            max_steps= 3 if self.config.get("debug", False) else -1, #if debug, run only 3 steps
            seed = SEED_TRAINING,
            data_seed = SEED_TRAINING,
        )

        is_deepseek = "deepseek" in self.model_url.lower()
        is_qwen_32 = "QwQ-32B" in self.model_url

        if is_deepseek:
            from trl import DataCollatorForCompletionOnlyLM

            collator = DataCollatorForCompletionOnlyLM(
                self.tokenizer.completion_instruction,
                tokenizer=self.tokenizer,
                mlm=False,
            )
        else:
            collator = None

        from transformers import DataCollatorForLanguageModeling

        collator = collator or DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        # Create and configure the trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # metric_for_best_model="val/f1",
            args=sft_config,
            data_collator=collator,
            processing_class=self.tokenizer,
        )
        self.trainer = trainer

        only_completion = self.config.get("only_completion", True)
        if not is_deepseek and not is_qwen_32 and only_completion:
            logging.info("Training on responses only...")
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part=self.tokenizer.start_instruction,
                response_part=self.tokenizer.completion_instruction,
            )
        else:
            logging.info("Training on full prompts...")

        undersample = self.config.get("undersample", False)
        if undersample:
            logging.info("Using dynamic undersampling for training...")
            # Get labels from the dataset
            train_labels = np.array(train_df["is_fraud"].tolist())
            # Add the labels to the trainer
            trainer.undersample = undersample
            trainer.train_dataset = PermutableUndersamplingDataset(
                dataset=trainer.train_dataset,
                train_labels=train_labels,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
            )

        if self.config.get("fast_generation", False):
            logging.info("Using fast generation for training...")
            trainer.fast_generation = True
            trainer.vllm_client = None
            self.best_lora_request = None
        else:
            trainer.fast_generation = False

        # Add custom evaluation callback
        evaluation_callback = EvaluationCallback(
            trainer=trainer,
            tokenizer=self.tokenizer,
            log_dir=self.log_dir,
            max_length=self.max_length,
            max_new_tokens=self.config["max_new_tokens"],
            run_eval_on_start=self.config.get("run_eval_on_start", True),
        )
        trainer.add_callback(evaluation_callback)
        # trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=30))

        if self.config.get("early_test", False):
            pass
        else:
            if self.checkpoint_timestamp:
                last_checkpoint_dir = get_last_checkpoint(self.log_dir)
                logging.info(f"Last checkpint dir: {last_checkpoint_dir}")
                trainer.train(resume_from_checkpoint=last_checkpoint_dir)
            else:
                trainer.train()

            logging.info("Instruction tuning complete.")

            # Save model
            self.save_model(filepath=self.log_dir)

        return self

    def evaluate(self, test_df, vllm_client=None):
        """
        Evaluate the model on a test dataset.

        Args:
            test_df (pd.DataFrame): Test data.

        Returns:
            tuple: (accuracy, report)
        """
        logging.info("Starting evaluation...")

        # Extract true labels and get predictions
        true_labels = []
        predicted_labels = []
        ciks = []
        sics = []
        quarters = []
        glabels = []

        test_data = [
            self.generate_prompt(row)
            for _, row in tqdm(test_df.iterrows(), desc="Preparing test data")
        ]

        # Filter out any None results (e.g., missing data)
        test_data = [d for d in test_data if d is not None]

        test_dataset = Dataset.from_list(test_data)

        from unsloth.chat_templates import standardize_data_formats

        test_dataset = standardize_data_formats(test_dataset)

        # Helper function to process a batch of examples
        def process_batch(examples, model, tokenizer, vllm_client):
            batch_results = []
            batch_prompts = []

            for example in examples:
                input_text = example["text"]
                input_parts = input_text.split(self.tokenizer.completion_instruction)
                prompt = input_parts[0]

                chat_prompt = prompt + self.tokenizer.completion_instruction

                batch_prompts.append(chat_prompt)

            # Generate predictions for the batch
            if vllm_client:
                predictions = llm_vllm_generate(
                    model,
                    tokenizer,
                    vllm_client,
                    batch_prompts,
                    self.max_length,
                    self.config["max_new_tokens"],
                )
            elif self.trainer.fast_generation:
                predictions = llm_fast_generate(
                    model,
                    batch_prompts,
                    self.max_length,
                    self.config["max_new_tokens"],
                    self.best_lora_request,
                )
            else:
                predictions = llm_generate(
                    model,
                    tokenizer,
                    batch_prompts,
                    self.max_length,
                    self.config["max_new_tokens"],
                )

            # Process each prediction
            for i, prediction_text in enumerate(predictions):
                example = examples[i]

                # Extract the answer from the generated text
                extracted_answer = prediction_text.split(
                    tokenizer.completion_instruction
                )[-1].strip()

                # Clean up prediction to get just the label
                if extracted_answer not in ["Fraud", "Not Fraud"]:
                    if (
                        "fraud" in extracted_answer.lower()
                        and "not fraud" not in extracted_answer.lower()
                    ):
                        extracted_answer = "Fraud"
                    else:
                        extracted_answer = "Not Fraud"

                batch_results.append(
                    {
                        "true_label": example["answer"],
                        "predicted_label": extracted_answer,
                        "cik": int(example["cik"]),
                        "sic": example["sic"],
                        "quarter": example["quarter"],
                        "glabels": example["glabels"],
                    }
                )

            return batch_results

        with torch.no_grad():
            # Process in batches for better efficiency
            batch_size = self.per_device_eval_batch_size
            dataset_length = len(test_dataset)

            for i in tqdm(
                range(0, dataset_length, batch_size),
                desc=f"Evaluating on test set (batch size: {batch_size})",
            ):
                # Get batch of examples
                batch_examples = [
                    test_dataset[j]
                    for j in range(i, min(i + batch_size, dataset_length))
                ]

                # Process the batch
                batch_results = process_batch(
                    batch_examples, self.model, self.tokenizer, vllm_client
                )

                # Collect results
                for result in batch_results:
                    true_labels.append(result["true_label"])
                    predicted_labels.append(result["predicted_label"])
                    ciks.append(result["cik"])
                    sics.append(result["sic"])
                    quarters.append(result["quarter"])
                    glabels.append(result["glabels"])

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, pos_label="Fraud")
        recall = recall_score(true_labels, predicted_labels, pos_label="Fraud")
        f1 = f1_score(true_labels, predicted_labels, pos_label="Fraud")
        # Calculate CIK Macro F1
        cik_macro_f1_macro = calculate_cik_macro_f1(true_labels, predicted_labels, ciks)
        cik_macro_f1_weighted = calculate_cik_macro_f1(
            true_labels, predicted_labels, ciks, average="weighted"
        )

        macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
        weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
        report = classification_report(true_labels, predicted_labels)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"CIK Macro F1 Macro Score: {cik_macro_f1_macro:.4f}")
        logging.info(f"CIK Macro F1 Weighted Score: {cik_macro_f1_weighted:.4f}")
        logging.info(f"Macro F1 Score: {macro_f1:.4f}")
        logging.info(f"Weighted F1 Score: {weighted_f1:.4f}")
        logging.info(f"Classification Report:\n{report}")

        # Save metrics to a file
        with open(os.path.join(self.log_dir, "test_metrics.json"), "w") as f:
            json.dump(
                {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                    "cik_macro_f1_macro": float(cik_macro_f1_macro),
                    "cik_macro_f1_weighted": float(cik_macro_f1_weighted),
                    "report": report,
                },
                f,
                indent=2,
            )

        # Save detailed test predictions to CSV
        test_results_df = pd.DataFrame(
            {
                "cik": ciks,
                "sic": test_df["sic"].tolist(),
                "quarter": quarters,
                "y_true": true_labels,
                "y_pred": predicted_labels,
                "glabels": glabels,
            }
        )

        tb_metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "cik_macro_f1_macro": float(cik_macro_f1_macro),
            "cik_macro_f1_weighted": float(cik_macro_f1_weighted),
        }
        self.trainer.log_metrics("test", tb_metrics_dict)

        test_csv_path = os.path.join(self.log_dir, "test_predictions.csv")
        test_results_df.to_csv(test_csv_path, index=False)
        logging.info(f"Saved detailed test predictions to {test_csv_path}")

        return accuracy, report

    def predict(self, content, few_shot_examples=None):
        """
        Make a prediction using the LLM without batch processing.

        Args:
            content: Input content for prediction (financial data or MDA text).
            few_shot_examples (list, optional): Few-shot examples.

        Returns:
            str: Predicted label.
        """
        self.model.eval()  # Set the model to evaluation mode

        # Format the prompt
        truncated_prompt = self.truncate_and_format_prompt(content)
        messages = [{"role": "user", "content": truncated_prompt}]

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and move to the correct device
        inputs = self.tokenizer(
            full_prompt,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=30, do_sample=False, use_cache=False
            )

        # Decode the full output
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract the model's response
        prediction_text = prediction.split(self.tokenizer.completion_instruction)[
            -1
        ].strip()

        # Clean up prediction to get just the label
        if prediction_text not in ["Fraud", "Not Fraud"]:
            if (
                "fraud" in prediction_text.lower()
                and "not" not in prediction_text.lower()
            ):
                prediction_text = "Fraud"
            else:
                prediction_text = "Not Fraud"

        return prediction_text

    def find_best_checkpoint(self):
        """
        Find the best checkpoint based on validation F1 score.

        Returns:
            tuple: (best_epoch, best_f1, best_checkpoint_dir)
        """

        # 1. Find latest checkpoint
        checkpoint_dirs = [
            d
            for d in self.log_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint")
        ]

        checkpoint_dir_per_epochs = {}
        for checkpoint_dir in checkpoint_dirs:
            with open(checkpoint_dir / "trainer_state.json", "r") as f:
                trainer_state = json.load(f)
                epoch = trainer_state["epoch"]
                epoch_num = int(epoch)
                found_epoch = False
                for k, v in checkpoint_dir_per_epochs.items():
                    if int(k) == epoch_num:
                        found_epoch = True
                        # Update if the current greater
                        if epoch > k:
                            checkpoint_dir_per_epochs[epoch] = checkpoint_dir
                            break
                if not found_epoch:
                    checkpoint_dir_per_epochs[epoch] = checkpoint_dir

        # Convert the epoch numbers to integers
        checkpoint_dir_per_epochs = {
            int(k): v for k, v in checkpoint_dir_per_epochs.items()
        }

        # Now find the best epoch per f1

        # now find the best epoch :
        metrics_files = list(self.log_dir.glob("metrics_epoch_*.json"))

        # Find the epoch with that f1 score
        best_epoch = -1
        best_f1 = 0
        for metrics_file in metrics_files:
            with open(metrics_file, "r") as f:
                best_key = "f1"
                metrics = json.load(f)
                if metrics[best_key] > best_f1:
                    # best_f1 = metrics["f1"]
                    best_f1 = metrics[best_key]
                    if "epoch" in metrics:
                        best_epoch = int(metrics["epoch"])
                    else:
                        logging.warning(
                            f"Epoch not found in metrics file: {metrics_file}. Using epoch number from name "
                            "of the file."
                        )
                        best_epoch = int(metrics_file.name.split("_")[2].split(".")[0])

        # No break because we want the last one
        if best_epoch == -1:
            logging.warning("No best epoch found.")
            return -1, best_f1, None

        best_model_checkpoint = checkpoint_dir_per_epochs[best_epoch]
        logging.info(f"Best model checkpoint: {best_model_checkpoint}")
        return best_epoch, best_f1, best_model_checkpoint.name

    def get_best_checkpoint_dir(self):
        return self.log_dir / self.find_best_checkpoint()[2]

    def train_and_evaluate(
        self,
        train_path=None,
        test_path=None,
        num_epochs=1,
        learning_rate=1e-5,
        save_steps=10,
    ):
        """
        Load data, instruction tune, and evaluate the model.

        Args:
            train_path (Path, optional): Path to training data.
            test_path (Path, optional): Path to test data.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            save_steps (int): How often to save checkpoints.

        Returns:
            tuple: (accuracy, report)
        """
        train_df, val_df, test_df = self.load_data(
            train_path=train_path, test_path=test_path
        )

        # Load model and apply PEFT/LoRA if they haven't been loaded already
        if self.model is None:
            self.load_model_and_tokenizer(
                fast_inference=self.config.get("fast_generation", False)
            )
            lora_target_modules = self.config.get(
                "lora_target_modules", LORA_TARGET_MODULES
            )
            logging.info(f"LoRA target modules: {lora_target_modules}")
            self.apply_peft_lora(target_modules=lora_target_modules)

        # Train the model
        self.llm_tune(
            train_df,
            val_df,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_steps=save_steps,
        )

        # Find and load the best model checkpoint based on validation F1 score
        best_epoch, best_f1, best_checkpoint_dir = self.find_best_checkpoint()

        if best_epoch >= 0 and not self.config.get(
            "no_validation", False
        ):  # if no_validation then we don't load the best checkpoint
            checkpoint_subdir = self.log_dir / best_checkpoint_dir
            if checkpoint_subdir.exists():
                best_model_path = checkpoint_subdir
                logging.info(
                    f"Loading best model from {best_model_path} with score: {best_f1:.4f}, epoch: {best_epoch}"
                )

                try:
                    vllm_client = self.trainer.vllm_client
                except Exception as e:
                    logging.info(
                        f"No vllm client found. Fast generate will be prefered."
                    )
                    vllm_client = None

                if vllm_client:
                    from transformers import AutoModelForCausalLM

                    self.model = AutoModelForCausalLM.from_pretrained(best_model_path)
                    logging.info(
                        f"Model loaded from {best_model_path} with score: {best_f1:.4f}"
                    )
                    vllm_client.update_model_params(self.model)
                    logging.info(
                        f"Model updated in VLLM client with  score: {best_f1:.4f}"
                    )
                elif self.trainer.fast_generation:
                    # Load the lora requests from best checkpoint
                    logging.info(
                        f"Loading LoRA requests from {best_model_path} with score: {best_f1:.4f}"
                    )
                    self.best_lora_request = self.model.load_lora(
                        self.get_best_checkpoint_dir(), load_tensors=False
                    )
                    logging.info(
                        f"Model loaded from {best_model_path} with score: {best_f1:.4f}"
                    )

                else:
                    load_in_8bit = self.config.get("load_in_8bit", False)
                    self.model, _ = FastLanguageModel.from_pretrained(
                        model_name=str(best_model_path),
                        full_finetuning=False,
                        load_in_4bit=self.config.get("load_in_4bit", not load_in_8bit),
                        load_in_8bit=load_in_8bit,
                        max_seq_length=self.max_length,
                    )
                    FastLanguageModel.for_inference(self.model)

            else:
                logging.warning(
                    f"Best checkpoint directory not found: {checkpoint_subdir}"
                )
        else:
            logging.warning("Using the last saved model checkpoint for evaluation")

        vllm_client = (
            self.trainer.vllm_client if hasattr(self.trainer, "vllm_client") else None
        )
        # Evaluate on test data
        if test_df is not None and len(test_df) > 0:
            # accuracy, report = self.evaluate(test_df,lora_request=lora_request, vllm_client=vllm_client)
            accuracy, report = self.evaluate(test_df, vllm_client=vllm_client)
            logging.info(f"Test accuracy: {accuracy:.4f}")
            logging.info(f"Test Classification Report:\n{report}")
            return accuracy, report
        else:
            logging.warning("No test data provided. Skipping evaluation on test set.")
            return None, None

    def save_model(self, filepath):
        """
        Save the PEFT adapters to a directory.

        Args:
            filepath: Path to save the model.

        Returns:
            str: Path where the model was saved.
        """
        logging.info(f"Saving PEFT model to {filepath}")

        self.model.save_pretrained(filepath)  # This saves the LoRA adapters

        logging.info("PEFT model saved successfully.")
        return filepath

    def format_financials(self, financials, drop_rate=0):
        """
        Create a prompt for the LLM model using only financial data.
        """

        def display_financial_value(value):
            """
            Fomrat the value in the form X,XXX,XXX but without decimal points.
            """
            if value == 0:
                return "0"
            elif abs(value) < 10:
                return "{:.2f}".format(value)
            else:
                return "{:,.0f}".format(value)

        # Convert the financials dictionary to a string format
        financials = {
            k: v
            for k, v in financials.items()
            if k not in EXCLUDED_FINANCIALS_FEATURES and v != 0 and not np.isnan(v)
        }

        # if drop
        if drop_rate > 0:
            financials = drop_random_keys(financials, drop_rate)

        # Convert ratios to percentages
        for key in financials.keys():
            if key in PERCENTAGE_FEATURES:
                financials[key] = financials[key] * 100

        financials_str = "\n".join(
            [
                f"- {EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT[key]} : {'$' if is_with_currency(key) else ''}{display_financial_value(value)}{'%' if key in PERCENTAGE_FEATURES else ''}"
                for key, value in financials.items()
            ]
        )

        return financials_str


def llm_train_and_evaluate_base_model(
    model_class,
    config=None,
    train_path=None,
    test_path=None,
):
    """
    Base function to train and evaluate an LLM fraud classifier.

    Args:
        model_class: Class of the model to train.
        config (dict): Configuration for the model.
        train_path (Path, optional): Path to training data.
        test_path (Path, optional): Path to test data.

    Returns:
        tuple: (trained_model, accuracy, report)
    """
    if config is None:
        raise Exception("Configuration should be provided")

    # Initialize the model
    model = model_class(config=config)

    logging.info(f"Starting {model.model_alias} training and evaluation")

    # Seed after model intiilaiation
    logging.info(f"Setting random seed to {SEED_TRAINING}")
    np.random.seed(SEED_TRAINING)
    random.seed(SEED_TRAINING)
    torch.manual_seed(SEED_TRAINING)

    # Train and evaluate the model
    accuracy, report = model.train_and_evaluate(
        train_path=train_path,
        test_path=test_path,
        num_epochs=config.get("num_train_epochs", 1),
        learning_rate=config.get("learning_rate", 2e-5),
        save_steps=config.get("save_steps", 10),
    )

    logging.info(f"{model.model_alias} training and evaluation completed")
    return model, accuracy, report
