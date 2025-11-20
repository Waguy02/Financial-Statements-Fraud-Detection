import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import unsloth  # noqa
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from trl import GRPOConfig, GRPOTrainer

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
    PREPROCESSED_PATH,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.grpo_interface import GRPOMixin

# Use the base GRPO class as a starting point
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financials_extended_grpo import (
    GRPO_EvaluationCallback,
    LLM_FinancialsClassifier,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    PermutableUndersamplingDataset,
    llm_train_and_evaluate_base_model,
)

# Imports from financial_and_extra script
from researchpkg.anomaly_detection.models.utils import (
    correctness_reward_func,
    drop_random_keys,
    extract_xml_answer,
    get_last_checkpoint,
    get_train_test_splitter,
    llm_vllm_generate,
    load_cik_company_mapping,
    load_sic_industry_title_index,
    load_train_test_path,
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

LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE = (
    500  # Approximate token count for prompts without data
)

# Constants from financial_and_extra script
EXTRA_FINANCIAL_PATH = PREPROCESSED_PATH / "SEC_MDA_FINANCIAL_JSON" / "quarterly"
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
MAX_FIN_TOKENS = 1000  # Maximum tokens for financial data

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


# Prompts from financial_and_extra script
LLM_SYSTEM_PROMPT = """You are a financial analyst specializing in corporate fraud detection. You will be given a structured set of financial indicators derived from a company's quarterly financial statements and additional financial information extracted from the company's official report.

Each financial variable includes:
- its **name**,
- its **numerical value** (in absolute terms or percentages),

Additionally, the company’s **industry sector** will be provided to help you interpret the financial context appropriately.

Your task is to thoroughly examine the entire set of financial variables and the additional information to determine whether there are signs of fraudulent activity during the reported quarter. No single variable is sufficient to conclude fraud; you must consider **the overall pattern of the data**. If any discrepancies arise, **prioritize the additional financial information**.

Please follow these strict instructions:

1. Carefully analyze the full financial profile in context.
2. Reason logically and concisely based on the data presented.
3. Then respond using **only one** of the following two labels:
   - **"Fraud"** → if you have sufficient indicators suggesting the company likely committed fraud during the quarter.
   - **"Not Fraud"** → if you find no clear evidence or pattern suggesting fraud.

4. Your answer must be formatted **strictly in the following XML structure**:

<reasoning>
  Write a concise explanation of your analysis here, in 8 sentences or fewer. Focus on key anomalies, inconsistencies, or red flags in the data, considering both the variables and the additional information. If there are no such concerns, explain why the financials appear sound.
</reasoning>
<answer>
  Write only one label here: either Fraud or Not Fraud.
</answer>

⚠️ Do not include any text outside the XML tags. Avoid generic or vague reasoning. The reasoning must be directly based on the financial variables and additional information provided.

Remember: you are acting as a professional fraud analyst. Be precise, rigorous, and objective in your evaluation.
"""

LLM_SYSTEM_PROMPT_DEEP_SEEK = LLM_SYSTEM_PROMPT.replace(
    "<reasoning>", "<think>"
).replace("</reasoning>", "</think>")


USER_PROMPT = """
The company operates in the {industry_title} sector.

### Financial Variables:
{financials_str}

### Additional Financial Information:
{additional_financials_str}
"""


# Use the same GRPO Evaluation Callback
# class GRPO_EvaluationCallback(EvaluationCallback): ... (Keep as is from financials_extended_grpo)


class LLM_FinancialAndExtraClassifierGRPO(GRPOMixin, LLM_FinancialsClassifier):
    """
    GRPO-based LLM Fraud classifier using extended financial data and extra financial info.
    Inherits from LLM_FinancialsClassifier for base GRPO setup.
    """

    def __init__(self, config):
        """
        Initialize the GRPO-based LLM classifier.
        """
        if "experiments_dir" not in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_fraud_classifier_dataset_{version}_extended_fin_and_extra_grpo"
                f"{'_oversample' if oversample else ''}{'_undersample' if undersample else ''}"
            )

        # Call parent constructor (LLM_FinancialsClassifier -> BaseLLMFraudClassifier)
        super().__init__(config=config)

        # Load SIC index and company mapping
        self.sic_index = load_sic_industry_title_index()
        self.cik_company_name_mapping = load_cik_company_mapping()

    def _process_loaded_data(self, df):
        """Process and merge extended financial data with the dataset."""
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

        return df

    def load_extra_financial_content(self, mda_quarter_id):
        """
        Load the content of the extra financial JSON File.
        """
        extra_file = EXTRA_FINANCIAL_PATH / f"{mda_quarter_id}.json"

        if not extra_file.exists():
            raise FileNotFoundError(
                f"Extra financial file {extra_file} does not exist."
            )

        with open(extra_file, "r", encoding="utf-8") as file:
            extra_financial_content = json.load(file)
            # Format as string
            extra_financial_content_str = ""
            for key, value in extra_financial_content.items():
                extra_financial_content_str += f"- {key} : {value}\n"
            return extra_financial_content_str

    def prepare_financial_data(self, row):
        """
        Extract and prepare financial data from a DataFrame row.
        Uses EXTENDED_FINANCIAL_FEATURES.
        """
        financial_data = {}
        for feature in filter(
            lambda x: x not in EXCLUDED_FINANCIALS_FEATURES, EXTENDED_FINANCIAL_FEATURES
        ):
            if feature in row and pd.notna(row[feature]):
                financial_data[feature] = row[feature]
        return financial_data

    def truncate_and_format_prompt(
        self, extra_financial_text, financials_str, industry_title
    ):
        """
        Ensure the prompt is within the model's context length, handling both data types.

        Args:
            extra_financial_text (str): The extra financial text.
            financials_str (str): The financial data as a formatted string.
            industry_title (str): The industry title based on SIC code.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        financials_max_tokens = MAX_FIN_TOKENS

        # Tokenize the financials string
        financials_tokens = self.tokenizer(
            financials_str,
            return_tensors="pt",
            truncation=True,
            max_length=financials_max_tokens,
            padding=False,
        )
        truncated_financials = self.tokenizer.decode(
            financials_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Calculate max tokens for extra financial text
        extra_max_tokens = int(
            self.max_length
            - len(financials_tokens["input_ids"][0])
            - SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE  # Reserve space for system/user prompt structure
        )

        # Tokenize the extra financial text
        extra_tokens = self.tokenizer(
            extra_financial_text,
            return_tensors="pt",
            truncation=True,
            max_length=max(0, extra_max_tokens),  # Ensure non-negative
            padding=False,
        )
        truncated_extra = self.tokenizer.decode(
            extra_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Format user prompt with all data
        formatted_user_prompt = USER_PROMPT.format(
            additional_financials_str=truncated_extra,
            financials_str=truncated_financials,
            industry_title=industry_title,
        )

        is_deep_seek = "deepseek" in self.config["model_url"].lower()
        llm_system_prompt = (
            LLM_SYSTEM_PROMPT_DEEP_SEEK if is_deep_seek else LLM_SYSTEM_PROMPT
        )
        return llm_system_prompt, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """Generate and tokenize the prompt for training using financial and extra data."""
        try:
            # Load extra financial content
            extra_financial_content = self.load_extra_financial_content(
                row["mda_quarter_id"]
            )

            # Prepare financial data
            financials = self.prepare_financial_data(row)
            financials_str = self.format_financials(financials)  # Use base class method

            # Get the SIC code and industry title
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index.get(
                sic, "Unknown Industry"
            )  # Handle missing SIC

            # Get the label
            label = "Fraud" if row["is_fraud"] else "Not Fraud"

            # Create the full prompt with both data sources
            system_prompt, user_prompt = self.truncate_and_format_prompt(
                extra_financial_content, financials_str, industry_title
            )

            # Format using chat template for GRPO input
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            result = {
                "text": full_prompt,
                "prompt": full_prompt,
                "answer": label,  # The ground truth label for GRPO reward calculation
                "sic": sic,
                "sicagg": row["sicagg"],
                "cik": row["cik"],
                "quarter": f"{row['year']}{row['quarter']}",
                "glabels": str(row.get("glabels", "no_fraud")),
            }

            return result

        except FileNotFoundError:
            logging.warning(
                f"Extra financial file for quarter ID {row['mda_quarter_id']} not found, skipping"
            )
            return None
        except Exception as e:
            logging.error(
                f"Error generating prompt for row {row.get('cik', 'N/A')}-{row.get('year', 'N/A')}{row.get('quarter', 'N/A')}: {e}"
            )
            return None

    def load_data(self, train_path, test_path):
        """
        Load train and test datasets, merging with financial data.
        Uses the _process_loaded_data method defined in this class.
        """
        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)

        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Process data with custom method (merges financials)
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        splitter = get_train_test_splitter(self.config)
        train_df, val_df = splitter(train_df, test_size=0.1, random_state=SEED_TRAINING)

        if self.config.get("oversample", False):
            logging.info("Oversampling fraud cases in training data...")
            train_df = self.oversample_fraud_cases(train_df)  # Use base class method

        # Filter out rows where prompt generation might fail (e.g., missing extra file)
        train_df = train_df.dropna(subset=["mda_quarter_id"])
        val_df = val_df.dropna(subset=["mda_quarter_id"])
        test_df = test_df.dropna(subset=["mda_quarter_id"])

        logging.info(f"Train data size after processing: {len(train_df)}")
        logging.info(f"Validation data size after processing: {len(val_df)}")
        logging.info(f"Test data size after processing: {len(test_df)}")

        return train_df, val_df, test_df


def train_and_evaluate_financial_and_extra_grpo_model(config=None):
    """
    Train and evaluate the LLM financial and extra classifier with GRPO.
    """
    train_path, test_path = load_train_test_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_FinancialAndExtraClassifierGRPO,  # Use the new GRPO class
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":

    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    CONFIG = {
        "model_url": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_context": 4096,  # Adjust based on model and data size
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 1,  # May need to be small due to larger context
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,  # Accumulate gradients
        "num_generations": 6,
        "num_train_epochs": 10,  # Adjust as needed
        "learning_rate": 5e-5,
        "save_steps": 5,  # Adjust frequency
        "max_new_tokens": 512,
        "dataset_version": "v3",  # Specify dataset version if needed
        # "undersample": True, # Optional: enable undersamplintrain_and_evaluate_softmax_financials_model
    }
