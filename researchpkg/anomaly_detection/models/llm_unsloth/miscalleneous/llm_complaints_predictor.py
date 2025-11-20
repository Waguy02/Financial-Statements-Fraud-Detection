"""
Use the LLM to Regress the content of the AAER
"""
import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
    PREPROCESSED_PATH,
    PREPROCESSED_PATH_EXTENDED,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    BaseLLMFraudClassifier,
    PermutableUndersamplingDataset,
)
from researchpkg.anomaly_detection.models.utils import (
    get_last_checkpoint,
    get_train_test_splitter,
    llm_fast_generate,
    llm_generate,
    llm_vllm_generate,
    load_cik_company_mapping,
    load_sic_industry_title_index,
    load_train_test_path,
    split_dataset_by_cik_label_agnostic,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    AGGREGATE_FEATURES,
    BENEISH_PROBM,
    DIFF_FEATURES,
    EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT,
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
    IMPORTANT_TAGS,
    RATIO_FEATURES,
    RATIO_NET_WORKING_CAPITAL,
)
from researchpkg.anomaly_detection.preprocessing.utils import clean_mda_content

# Constants
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
AAER_INDEX_WITH_REFORMULATED_COMPLAINTS = (
    PREPROCESSED_PATH_EXTENDED
    / "v4_unbalanced_cik_unbiased_index_with_reformulated_complaints.csv"
)
MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"
MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"

SYSTEM_INSTRUCTION_AND_TURN_COUNT = 350

LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]

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

LLM_SYSTEM_PROMPT = """You are a financial analyst specializing in corporate fraud characterization. You will be given two types of information:
1. The Management Discussion and Analysis (MDA) section from a company's quarterly financial report
2. A structured set of financial indicators derived from the company's quarterly financial statements

The MDA text contains management's narrative explanation of the company's financial condition, results of operations, and future prospects.

Each financial variable includes its name and numerical value (in absolute terms or percentages).

Additionally, the company's **industry sector** will be provided to help you interpret the financial context appropriately.

It has already been proven that this company committed financial statement fraud during this reporting period. Your task is to thoroughly characterize the nature of this fraudulent activity by examining both the MDA text and financial data.

Please follow these strict instructions:

1. Analyze both the MDA text and the financial data to identify specific evidence of financial statement fraud
2. Focus on inconsistencies, misleading statements, unusual patterns, or discrepancies that reveal fraud
3. Provide 3-5 concise bullet points that characterize the specific nature of the fraud
4. Each bullet point should:
   - Describe a specific aspect of the fraudulent activity
   - Use precise financial terminology
    - Be factual and objective and don't repeat the same information in different words
5. Your bullet points should collectively explain WHAT type of fraud was committed (revenue recognition, expense manipulation, asset misvaluation, etc.) and HOW it manifested in the reporting

Format your response as bullet points only, without any additional explanations, introductions, or conclusions.
"""

USER_PROMPT = """
The company operates in the {industry_title} sector.

### Content of the MDA Section:
{mda_text}

### The financial variables:
{financials_str}
"""


class SentenceTransformerEvaluationCallback(TrainerCallback):
    """Custom callback for evaluation using sentence transformers"""

    def __init__(
        self,
        trainer,
        tokenizer,
        log_dir,
        max_length,
        max_new_tokens,
        sentence_transformer_model_name,
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

        if hasattr(trainer, "undersample"):
            self.undersample = trainer.undersample
        else:
            self.undersample = False

        if hasattr(trainer, "fast_generation"):
            self.fast_generation = trainer.fast_generation
            self.current_lora_request = None
        else:
            self.fast_generation = False

        self.sentence_transformer_model_name = sentence_transformer_model_name

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.undersample:

            logging.warning(
                "undersample was set to True, but not supported in complaints predictor"
            )

        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.epoch == 0 and self.run_eval_on_start:
            logging.info("Running evaluation at the beginning of training...")
            self.on_epoch_end(args, state, control, **kwargs)
        return super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation at the end of each epoch."""

        # Only load the embedding model when needed
        # Load sentence transformer on the fly
        logging.info(
            f"Loading sentence transformer model: {self.sentence_transformer_model_name}"
        )
        sentence_transformer = SentenceTransformer(self.sentence_transformer_model_name)
        sentence_transformer.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            model = self.trainer.model

            vllm_client = None
            if hasattr(self.trainer, "vllm_client"):
                # If using VLLM, we need to set the model to eval mode
                vllm_client = self.trainer.vllm_client
            elif self.fast_generation:
                # We will load the lora weights only if we are not using vllm
                self.current_lora_request = model.load_lora(
                    get_last_checkpoint(self.log_dir)
                )
                logging.info(
                    f"Fast Inference: Loading lora weights from {get_last_checkpoint(self.log_dir)}"
                )
            else:
                from unsloth import FastLanguageModel

                logging.info(
                    "No VLLM client and no fast inference, using default model"
                )
                FastLanguageModel.for_inference(model)

            # Extract true sentences and get predictions
            true_sentences = []
            predicted_sentences = []
            true_token_counts = []
            predicted_token_counts = []
            ciks = []
            sics = []
            quarters = []
            aaer_nos = []

            # Process a batch of examples
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
                    true_complaint = input_parts[1].strip()

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
                    true_complaint = example["answer"]

                    # Extract the prediction text
                    prediction_text = prediction_text.split(
                        tokenizer.completion_instruction
                    )[-1].strip()

                    # Count tokens in true complaint and prediction
                    true_tokens = len(tokenizer.encode(true_complaint))
                    pred_tokens = len(tokenizer.encode(prediction_text))

                    batch_results.append(
                        {
                            "text": example["text"],
                            "true_complaint": true_complaint,
                            "predicted_complaint": prediction_text,
                            "true_token_count": true_tokens,
                            "predicted_token_count": pred_tokens,
                            "cik": int(example["cik"]),
                            "sic": example["sic"],
                            "quarter": example["quarter"],
                            "aaer_no": example["aaer_no"],
                        }
                    )

                return batch_results

            # Process in batches
            batch_size = self.trainer.args.per_device_eval_batch_size
            dataset_length = len(self.trainer.eval_dataset)

            for i in tqdm(
                range(0, dataset_length, batch_size),
                desc=f"ST Evaluation at epoch {int(state.epoch)} (batch size: {batch_size})",
            ):
                # Get batch of examples
                batch_examples = [
                    self.trainer.eval_dataset[j]
                    for j in range(i, min(i + batch_size, dataset_length))
                ]

                # Process the batch
                batch_results = process_batch(
                    batch_examples, model, self.tokenizer, vllm_client
                )

                # Collect results
                for result in batch_results:
                    true_sentences.append(result["true_complaint"])
                    predicted_sentences.append(result["predicted_complaint"])
                    true_token_counts.append(result["true_token_count"])
                    predicted_token_counts.append(result["predicted_token_count"])
                    ciks.append(result["cik"])
                    sics.append(result["sic"])
                    quarters.append(result["quarter"])
                    aaer_nos.append(result["aaer_no"])

                    # Log sample predictions occasionally
                    if state.epoch == 0 or random.random() < 0.1:
                        prompt = result["text"].split(
                            self.tokenizer.completion_instruction
                        )[0]
                        logging.info(f"\n Prompt : {prompt}")
                        logging.info(
                            f"---------------------------"
                            f"\nPrediction: {result['predicted_complaint']}"
                        )
                        logging.info(
                            f"-----------------------------\n"
                            f"True: {result['true_complaint']}"
                        )

                        logging.info(
                            f"CIK: {result['cik']}, SIC: {result['sic']}, Quarter: {result['quarter']}"
                        )

            # Calculate sentence transformer similarity scores
            true_embeddings = sentence_transformer.encode(
                true_sentences, convert_to_tensor=True
            )
            pred_embeddings = sentence_transformer.encode(
                predicted_sentences, convert_to_tensor=True
            )

            # Calculate cosine similarity scores
            similarities = []
            for i in range(len(true_sentences)):
                true_emb = true_embeddings[i].unsqueeze(0)
                pred_emb = pred_embeddings[i].unsqueeze(0)
                similarity = torch.nn.functional.cosine_similarity(
                    true_emb, pred_emb
                ).item()
                similarities.append(similarity)

            # Calculate metrics
            avg_similarity = np.mean(similarities)
            median_similarity = np.median(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)

            # Calculate token count statistics
            avg_true_tokens = np.mean(true_token_counts)
            avg_pred_tokens = np.mean(predicted_token_counts)
            med_true_tokens = np.median(true_token_counts)
            med_pred_tokens = np.median(predicted_token_counts)
            max_true_tokens = np.max(true_token_counts)
            max_pred_tokens = np.max(predicted_token_counts)
            min_true_tokens = np.min(true_token_counts)
            min_pred_tokens = np.min(predicted_token_counts)

            logging.info(f"Average Semantic Similarity: {avg_similarity:.4f}")
            logging.info(f"Median Semantic Similarity: {median_similarity:.4f}")
            logging.info(f"Min Similarity: {min_similarity:.4f}")
            logging.info(f"Max Similarity: {max_similarity:.4f}")
            logging.info(
                f"Avg true tokens: {avg_true_tokens:.1f}, Avg predicted tokens: {avg_pred_tokens:.1f}"
            )
            logging.info(
                f"Median true tokens: {med_true_tokens}, Median predicted tokens: {med_pred_tokens}"
            )

            # Save metrics to a file
            import json

            with open(
                os.path.join(self.log_dir, f"metrics_epoch_{int(state.epoch)}.json"),
                "w",
            ) as f:
                json.dump(
                    {
                        "epoch": int(state.epoch),
                        "avg_similarity": float(avg_similarity),
                        "median_similarity": float(median_similarity),
                        "min_similarity": float(min_similarity),
                        "max_similarity": float(max_similarity),
                        "token_stats": {
                            "avg_true_tokens": float(avg_true_tokens),
                            "avg_pred_tokens": float(avg_pred_tokens),
                            "median_true_tokens": float(med_true_tokens),
                            "median_pred_tokens": float(med_pred_tokens),
                            "max_true_tokens": int(max_true_tokens),
                            "max_pred_tokens": int(max_pred_tokens),
                            "min_true_tokens": int(min_true_tokens),
                            "min_pred_tokens": int(min_pred_tokens),
                        },
                    },
                    f,
                    indent=2,
                )

            # Save detailed predictions to CSV
            eval_results_df = pd.DataFrame(
                {
                    "cik": ciks,
                    "sic": sics,
                    "quarter": quarters,
                    "aaer_no": aaer_nos,
                    "true_complaint": true_sentences,
                    "predicted_complaint": predicted_sentences,
                    "true_token_count": true_token_counts,
                    "predicted_token_count": predicted_token_counts,
                    "similarity_score": similarities,
                }
            )

            csv_path = os.path.join(
                self.log_dir, f"eval_predictions_epoch_{int(state.epoch)}.csv"
            )
            eval_results_df.to_csv(csv_path, index=False)
            logging.info(f"Saved detailed evaluation predictions to {csv_path}")

            # Keep model in train mode for further training
            model.train()

            if vllm_client:
                from unsloth import FastLanguageModel

                FastLanguageModel.for_training(model)
            return control


class LLM_complaints_Predictor(BaseLLMFraudClassifier):
    """
    LLM model for predicting AAER complaint characterizations from financial data and MDA text
    """

    def __init__(self, config):
        if "experiments_dir" not in config:
            version = config.get("dataset_version", "v3")
            oversample = False
            undersample = False
            only_completion = config.get("only_completion", True)
            assert not (
                oversample and undersample
            ), "You cannot set both undersample and oversample args to True."

            use_last_4_quarters = config.get("include_last_4_quarters", False)

            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_complaints_predictor_dataset_{version}_fin_and_mda"
                f"{'_raw_complaints' if config.get('use_raw_complaints', False) else ''}"
            )

        # Ensure sentence transformer model name is provided
        assert (
            "sentence_transformer_model" in config
        ), "sentence_transformer_model must be provided in config"

        super().__init__(config)

        # Load SIC index and company mapping
        self.sic_index = load_sic_industry_title_index()
        self.cik_company_name_mapping = load_cik_company_mapping()

    def load_mda_content(self, mda_quarter_id):
        """
        Load the content of an MDA file.
        """
        mda_path = (
            MDA_PATH if self.config.get("raw_mda", False) else MDA_PATH_SUMMARIZED
        )
        mda_file = mda_path / f"{mda_quarter_id}.txt"

        if not mda_file.exists():
            raise FileNotFoundError(f"MDA file {mda_file} does not exist.")

        with open(mda_file, "r", encoding="utf-8") as file:
            mda_content = file.read()
            # Clean the MDA content
            # mda_content = clean_mda_content(mda_content)

            return mda_content

    def _process_loaded_data(self, df):
        """Process and merge financial data with the dataset."""
        # Load financial data if not already loaded
        if not hasattr(self, "_full_financials_df"):
            logging.info(f"Loading full financials data from {FULL_FINANCIAL_PATH}")
            self._full_financials_df = pd.read_csv(FULL_FINANCIAL_PATH)
            self._full_financials_df = self._full_financials_df[
                ["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES
            ]

        # Drop existing feature count columns if present
        df = df.drop(columns=EXTENDED_FINANCIAL_FEATURES_COUNT_COLS, errors="ignore")

        # Merge with financials data
        df = df.merge(
            self._full_financials_df, on=["cik", "year", "quarter"], how="left"
        )

        # Remove rows without aaer_no
        df = df.dropna(subset=["aaer_no"])

        # Read the index of aaer reformulated

        key = (
            "reformulated_complaints"
            if not self.config.get("use_raw_complaints", False)
            else "complaints"
        )

        # remove the key df
        df = df.drop(columns=[key], errors="ignore")

        logging.info(
            f"Loading AAER index with {key} from {AAER_INDEX_WITH_REFORMULATED_COMPLAINTS}"
        )

        df_aaer_index = pd.read_csv(AAER_INDEX_WITH_REFORMULATED_COMPLAINTS)[
            ["aaer_no", key]
        ]
        df_aaer_index = df_aaer_index.dropna(subset=["aaer_no"]).drop_duplicates()
        df = pd.merge(df, df_aaer_index, on="aaer_no")

        # rename key to "complaints_text"
        df = df.rename(columns={key: "complaints_text"}, inplace=False)
        logging.info("Df columns: %s", df.columns)
        return df

    def load_data(self, train_path=None, test_path=None):
        """
        Load train and test datasets.

        Args:
            train_path (Path, optional): Path to training data.
            test_path (Path, optional): Path to test data.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        assert (
            self.config["dataset_version"] == "company_isolated_splitting"
        ), "Complaints predictor is only available for dataset version v4"

        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)
        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Process data with custom method
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        # Split train data into train and validation
        splitter = split_dataset_by_cik_label_agnostic
        train_df, val_df = splitter(train_df, test_size=0.1)

        logging.info("Train data size: %d", len(train_df))
        logging.info("Validation data size: %d", len(val_df))
        logging.info("Test data size: %d", len(test_df))

        return train_df, val_df, test_df

    def prepare_financial_data(self, row):
        """
        Extract and prepare financial data from a DataFrame row.

        Args:
            row (pd.Series): A row from the DataFrame containing financial data.

        Returns:
            dict: A dictionary of financial variables.
        """
        financial_data = {}
        for feature in filter(
            lambda x: x not in EXCLUDED_FINANCIALS_FEATURES, EXTENDED_FINANCIAL_FEATURES
        ):
            financial_data[feature] = row[feature]
        return financial_data

    def truncate_and_format_prompt(self, mda_text, financials_str, industry_title):
        """
        Ensure the prompt is within the model's context length.

        Args:
            mda_text (str): The MDA text to include in the prompt.
            financials_str (str): The financial data as a formatted string.
            industry_title (str): The industry title based on SIC code.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        financials_max_tokens = 1000

        # Tokenize the financials string
        financials_tokens = self.tokenizer(
            financials_str,
            return_tensors="pt",
            truncation=True,
            max_length=financials_max_tokens,
            padding=False,
        )

        # Decode truncated financials back to text
        truncated_financials = self.tokenizer.decode(
            financials_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Calculate the maximum tokens for MDA (less than half of available context)
        mda_max_tokens = int(
            self.max_length
            - len(financials_tokens["input_ids"][0])
            - SYSTEM_INSTRUCTION_AND_TURN_COUNT
            - self.config.get("max_new_tokens", 200)
        )

        # Tokenize the MDA text
        mda_tokens = self.tokenizer(
            mda_text,
            return_tensors="pt",
            truncation=True,
            max_length=mda_max_tokens,
            padding=False,
        )

        # Decode truncated MDA back to text
        truncated_mda = self.tokenizer.decode(
            mda_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Format user prompt with all data
        formatted_user_prompt = USER_PROMPT.format(
            mda_text=truncated_mda,
            financials_str=truncated_financials,
            industry_title=industry_title,
        )

        return LLM_SYSTEM_PROMPT, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """
        Generate and tokenize the prompt for training.

        Args:
            row (pd.Series): Row from the DataFrame.
            idx (int, optional): Index of the example.

        Returns:
            dict: Dictionary with prompts and metadata for training.
        """
        try:
            # Load MDA content
            mda_content = self.load_mda_content(row["mda_quarter_id"])

            # Prepare financial data
            financials = self.prepare_financial_data(row)
            financials_str = self.format_financials(financials)

            # Get the SIC code and industry title
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index[sic]

            # Get the reformulated complaint text
            complaint_text = row["complaints_text"]

            # Create the full prompt with both data sources
            system_prompt, user_prompt = self.truncate_and_format_prompt(
                mda_content, financials_str, industry_title
            )

            # Format using chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": complaint_text},
            ]

            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            result = {
                "text": full_prompt,
                "answer": complaint_text,
                "sic": sic,
                "cik": row["cik"],
                "quarter": row["quarter"],
                "aaer_no": row["aaer_no"],
            }

            return result
        except FileNotFoundError:
            logging.warning(
                f"MDA file for quarter ID {row.get('mda_quarter_id', 'unknown')} not found, skipping"
            )
            return None

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
        if self.config.get("undersample", False):
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
            bf16=True,
            # do_eval=True,
            # # eval_strategy="epoch",
            do_eval=self.config.get("do_eval", False),
            eval_strategy="no" if not self.config.get("do_eval", False) else "epoch",
            save_strategy=save_strategy,
            # Save each 10 percent of the training dataset
            save_steps=save_steps,
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

        # Create and configure the trainer
        from transformers import DataCollatorForLanguageModeling

        collator = collator or DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # metric_for_best_model="val/f1",
            args=sft_config,
            data_collator=collator,
            tokenizer=self.tokenizer,
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

        if self.config.get("fast_generation", False):
            logging.info("Using fast generation for training...")
            trainer.fast_generation = True
            trainer.vllm_client = None
            self.best_lora_request = None
        else:
            trainer.fast_generation = False

            # Add our custom evaluation callback using sentence transformers
        st_callback = SentenceTransformerEvaluationCallback(
            trainer=self.trainer,
            tokenizer=self.tokenizer,
            log_dir=self.log_dir,
            max_length=self.max_length,
            max_new_tokens=self.config["max_new_tokens"],
            sentence_transformer_model_name=self.config["sentence_transformer_model"],
            run_eval_on_start=self.config.get("run_eval_on_start", True),
        )
        self.trainer.add_callback(st_callback)

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

    def find_best_checkpoint_by_similarity(self):
        """
        Find the best checkpoint based on average similarity score.

        Returns:
            tuple: (best_epoch, best_similarity, best_checkpoint_dir)
        """
        # 1. Find all checkpoints
        checkpoint_dirs = [
            d
            for d in self.log_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint")
        ]

        if not checkpoint_dirs:
            logging.warning("No checkpoint directories found.")
            return -1, 0, None

        # Map checkpoints to their epochs
        checkpoint_dir_per_epochs = {}
        for checkpoint_dir in checkpoint_dirs:
            try:
                with open(checkpoint_dir / "trainer_state.json", "r") as f:
                    trainer_state = json.load(f)
                    epoch = trainer_state["epoch"]
                    epoch_num = int(epoch)

                    # Handle multiple checkpoints for the same epoch
                    found_epoch = False
                    for k, v in checkpoint_dir_per_epochs.items():
                        if int(k) == epoch_num:
                            found_epoch = True
                            # Update if the current epoch is greater (more precise)
                            if epoch > k:
                                checkpoint_dir_per_epochs[epoch] = checkpoint_dir
                                break
                    if not found_epoch:
                        checkpoint_dir_per_epochs[epoch] = checkpoint_dir
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Error reading trainer state from {checkpoint_dir}: {e}"
                )
                continue

        # Convert the epoch numbers to integers
        checkpoint_dir_per_epochs = {
            int(k): v for k, v in checkpoint_dir_per_epochs.items()
        }

        if not checkpoint_dir_per_epochs:
            logging.warning("Could not find any valid checkpoint epochs.")
            return -1, 0, None

        # Find similarity metric files
        pattern = "metrics_epoch_*.json"
        metrics_files = list(self.log_dir.glob(pattern))

        if not metrics_files:
            logging.warning(
                f"No similarity metrics files found with pattern: {pattern}"
            )
            return -1, 0, None

        # Find the epoch with the best similarity score
        best_epoch = -1
        best_similarity = 0
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    if "avg_similarity" in metrics:
                        similarity = metrics["avg_similarity"]
                        if similarity > best_similarity:
                            best_similarity = similarity
                            # Extract epoch from filename
                            match = re.search(r"epoch_(\d+)", metrics_file.name)
                            if match:
                                best_epoch = int(match.group(1))
                            elif "epoch" in metrics:
                                best_epoch = int(metrics["epoch"])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Error processing metrics file {metrics_file}: {e}")
                continue

        if best_epoch == -1:
            logging.warning(f"No best epoch found for avg_similarity metric.")
            return -1, best_similarity, None

        if best_epoch not in checkpoint_dir_per_epochs:
            logging.warning(f"Best epoch {best_epoch} not found in checkpoints.")
            return best_epoch, best_similarity, None

        best_model_checkpoint = checkpoint_dir_per_epochs[best_epoch]
        logging.info(
            f"Best model checkpoint for avg_similarity: {best_model_checkpoint} (value: {best_similarity:.4f})"
        )
        return best_epoch, best_similarity, best_model_checkpoint.name

    def get_best_similarity_checkpoint_dir(self):
        """Get the directory path of the checkpoint with the best similarity score."""
        _, _, checkpoint_dir_name = self.find_best_checkpoint_by_similarity()
        if checkpoint_dir_name:
            return self.log_dir / checkpoint_dir_name
        return None

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
            tuple: (avg_similarity, results_dict)
        """
        # Load data

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

        # Find and load the best model checkpoint based on average similarity score
        (
            best_epoch,
            best_similarity,
            best_checkpoint_dir,
        ) = self.find_best_checkpoint_by_similarity()

        if best_epoch >= 0 and not self.config.get("no_validation", False):
            checkpoint_subdir = self.log_dir / best_checkpoint_dir
            if checkpoint_subdir.exists():
                best_model_path = checkpoint_subdir
                logging.info(
                    f"Loading best model from {best_model_path} with avg_similarity: {best_similarity:.4f}"
                )

                try:
                    vllm_client = self.trainer.vllm_client
                except Exception as e:
                    logging.info(
                        f"No vllm client found. Fast generate will be preferred."
                    )
                    vllm_client = None

                if vllm_client:
                    from transformers import AutoModelForCausalLM

                    self.model = AutoModelForCausalLM.from_pretrained(best_model_path)
                    logging.info(
                        f"Model loaded from {best_model_path} with avg_similarity: {best_similarity:.4f}"
                    )
                    vllm_client.update_model_params(self.model)
                    logging.info(
                        f"Model updated in VLLM client with avg_similarity: {best_similarity:.4f}"
                    )
                elif self.trainer.fast_generation:
                    # Load the lora requests from best checkpoint
                    logging.info(
                        f"Loading LoRA requests from {best_model_path} with avg_similarity: {best_similarity:.4f}"
                    )
                    self.best_lora_request = self.model.load_lora(
                        checkpoint_subdir, load_tensors=False
                    )
                    logging.info(
                        f"Model loaded from {best_model_path} with avg_similarity: {best_similarity:.4f}"
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
            avg_similarity, results_dict = self.evaluate(
                test_df, vllm_client=vllm_client
            )
            logging.info(f"Test average similarity: {avg_similarity:.4f}")
            return avg_similarity, results_dict
        else:
            logging.warning("No test data provided. Skipping evaluation on test set.")
            return None, None

    def evaluate(self, test_df, vllm_client=None):
        """
        Evaluate the model on a test dataset using sentence transformers.

        Args:
            test_df (pd.DataFrame): Test data.
            vllm_client: Optional VLLM client for generation.

        Returns:
            tuple: (avg_similarity, results_dict)
        """
        logging.info("Starting evaluation with sentence transformers...")

        test_data = [
            self.generate_prompt(row)
            for _, row in tqdm(test_df.iterrows(), desc="Preparing test data")
        ]

        # Filter out any None results (e.g., missing data)
        test_data = [d for d in test_data if d is not None]

        # Extract true complaints and get predictions
        true_complaints = []
        predicted_complaints = []
        true_token_counts = []
        predicted_token_counts = []
        ciks = []
        sics = []
        quarters = []
        aaer_nos = []

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
            elif (
                hasattr(self, "trainer")
                and hasattr(self.trainer, "fast_generation")
                and self.trainer.fast_generation
            ):
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
                true_complaint = example["answer"]

                # Extract the prediction from the generated text
                extracted_complaint = prediction_text.split(
                    tokenizer.completion_instruction
                )[-1].strip()

                # Count tokens in true complaint and prediction
                true_tokens = len(tokenizer.encode(true_complaint))
                pred_tokens = len(tokenizer.encode(extracted_complaint))

                batch_results.append(
                    {
                        "text": example["text"],
                        "true_complaint": true_complaint,
                        "predicted_complaint": extracted_complaint,
                        "true_token_count": true_tokens,
                        "predicted_token_count": pred_tokens,
                        "cik": int(example["cik"]),
                        "sic": example["sic"],
                        "quarter": example["quarter"],
                        "aaer_no": example["aaer_no"],
                    }
                )

            return batch_results

        sentence_transformer = SentenceTransformer(
            self.config["sentence_transformer_model"]
        )
        sentence_transformer.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            # Process in batches
            batch_size = self.per_device_eval_batch_size
            dataset_length = len(test_data)

            for i in tqdm(
                range(0, dataset_length, batch_size),
                desc=f"Evaluating on test set (batch size: {batch_size})",
            ):
                # Get batch of examples
                batch_examples = [
                    test_data[j] for j in range(i, min(i + batch_size, dataset_length))
                ]

                # Process the batch
                batch_results = process_batch(
                    batch_examples, self.model, self.tokenizer, vllm_client
                )

                # Collect results
                for result in batch_results:
                    true_complaints.append(result["true_complaint"])
                    predicted_complaints.append(result["predicted_complaint"])
                    true_token_counts.append(result["true_token_count"])
                    predicted_token_counts.append(result["predicted_token_count"])
                    ciks.append(result["cik"])
                    sics.append(result["sic"])
                    quarters.append(result["quarter"])
                    aaer_nos.append(result["aaer_no"])

                    # Log some examples
                    if random.random() < 0.001:
                        prompt = result["text"].split(
                            self.tokenizer.completion_instruction
                        )[0]
                        logging.info(f"\nPrompt: {prompt}")
                        logging.info(
                            f"-----------------------------\n"
                            f"\nPredicted: {result['predicted_complaint']}"
                        )
                        logging.info(
                            f"-----------------------------\n"
                            f"True: {result['true_complaint']}"
                        )

        # Calculate sentence transformer embeddings
        true_embeddings = sentence_transformer.encode(true_complaints)
        pred_embeddings = sentence_transformer.encode(predicted_complaints)

        # Calculate cosine similarity scores
        similarities = []
        for i in range(len(true_complaints)):
            similarity = cosine_similarity([true_embeddings[i]], [pred_embeddings[i]])[
                0
            ][0]
            similarities.append(similarity)

        # Calculate metrics
        avg_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)

        # Calculate token count statistics
        avg_true_tokens = np.mean(true_token_counts)
        avg_pred_tokens = np.mean(predicted_token_counts)
        med_true_tokens = np.median(true_token_counts)
        med_pred_tokens = np.median(predicted_token_counts)
        max_true_tokens = np.max(true_token_counts)
        max_pred_tokens = np.max(predicted_token_counts)
        min_true_tokens = np.min(true_token_counts)
        min_pred_tokens = np.min(predicted_token_counts)

        logging.info(f"Average Semantic Similarity: {avg_similarity:.4f}")
        logging.info(f"Median Semantic Similarity: {median_similarity:.4f}")
        logging.info(f"Min Similarity: {min_similarity:.4f}")
        logging.info(f"Max Similarity: {max_similarity:.4f}")
        logging.info(
            f"Avg true tokens: {avg_true_tokens:.1f}, Avg predicted tokens: {avg_pred_tokens:.1f}"
        )
        logging.info(
            f"Median true tokens: {med_true_tokens}, Median predicted tokens: {med_pred_tokens}"
        )

        # Save metrics to a file
        results_dict = {
            "avg_similarity": float(avg_similarity),
            "median_similarity": float(median_similarity),
            "min_similarity": float(min_similarity),
            "max_similarity": float(max_similarity),
            "token_stats": {
                "avg_true_tokens": float(avg_true_tokens),
                "avg_pred_tokens": float(avg_pred_tokens),
                "median_true_tokens": float(med_true_tokens),
                "median_pred_tokens": float(med_pred_tokens),
                "max_true_tokens": int(max_true_tokens),
                "max_pred_tokens": int(max_pred_tokens),
                "min_true_tokens": int(min_true_tokens),
                "min_pred_tokens": int(min_pred_tokens),
            },
        }

        with open(os.path.join(self.log_dir, "test_similarity_metrics.json"), "w") as f:
            json.dump(results_dict, f, indent=2)

        # Save detailed test predictions to CSV
        test_results_df = pd.DataFrame(
            {
                "cik": ciks,
                "sic": sics,
                "quarter": quarters,
                "aaer_no": aaer_nos,
                "true_complaint": true_complaints,
                "predicted_complaint": predicted_complaints,
                "true_token_count": true_token_counts,
                "predicted_token_count": predicted_token_counts,
                "similarity_score": similarities,
            }
        )

        test_csv_path = os.path.join(self.log_dir, "test_predictions.csv")
        test_results_df.to_csv(test_csv_path, index=False)
        logging.info(f"Saved detailed test predictions to {test_csv_path}")

        return avg_similarity, results_dict


def train_and_evaluate_complaints_predictor(config=None):
    """
    Train and evaluate the LLM complaints predictor model.
    """
    from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
        llm_train_and_evaluate_base_model,
    )

    train_path, test_path = load_train_test_path(config)

    model, avg_similarity, results_dict = llm_train_and_evaluate_base_model(
        model_class=LLM_complaints_Predictor,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )

    return model, avg_similarity, results_dict


if __name__ == "__main__":
    import os
    import random

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    # Example configuration
    CONFIG = {
        "model_url": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_context": 10000,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "save_steps": 10,
        "max_new_tokens": 256,
        "sentence_transformer_model": "sentence-transformers/all-MiniLM-L6-v2",
        "fast_generation": True,
        "dataset_version": "company_isolated_splitting",  # Required for complaints predictor
    }

    model, avg_similarity, results = train_and_evaluate_complaints_predictor(CONFIG)

    if avg_similarity is not None:
        logging.info(f"Final test average similarity: {avg_similarity:.4f}")
        logging.info(f"Results details: {results}")
