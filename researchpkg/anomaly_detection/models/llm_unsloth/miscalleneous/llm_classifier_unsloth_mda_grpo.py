import json
import logging
import os

import pandas as pd
import torch
import unsloth  # noqa
import yaml
from datasets import Dataset
from joblib import Parallel, delayed
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
    PREPROCESSED_PATH,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financials_extended_and_mda_grpo import (
    SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE,
)
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_mda import (
    LLM_MDAOnlyClassifier,
)
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_mda_grpo import (
    GRPO_EvaluationCallback,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscallaneous.grpo_interface import (
    GRPOMixin,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    llm_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import (
    correctness_reward_func,
    get_last_checkpoint,
    get_train_test_splitter,
    load_cross_validation_path,
    split_dataset_by_cik,
)

MDA_PATH = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"


LLM_SYSTEM_PROMPT = """You are a financial analyst specializing in corporate fraud detection. You will be given two types of information:
1. The Management Discussion and Analysis (MDA) section from a company's quarterly financial report

The MDA text contains management's narrative explanation of the company's financial condition, results of operations, and future prospects.

Additionally, the company's **industry sector** will be provided to help you interpret the financial context appropriately.

Your task is to thoroughly examine both the MDA text to determine whether there are signs of fraudulent activity (Financial statement fraud) during the reported quarter. Look for irregularities, inconsistencies, or discrepancies that may indicate fraud.

Please follow these strict instructions:

1. Carefully analyze both the MDA text looking for any inconsistency.
2. Reason logically and concisely based on the information presented.
3. You should reason step by step to provide the final answer (at most 6 sentences in your reasoning).
4. Then respond using **only one** of the following two labels:
   - "Fraud" → if you have sufficient indicators suggesting the company likely committed fraud during the quarter.
   - "Not Fraud" → if you find no clear evidence or pattern suggesting fraud.

5. Remember that no single variable is sufficient to conclude fraud; you must consider **the overall pattern of the data**.

6. Your answer must be formatted **strictly in the following XML structure**:

<reasoning>
  Write a concise explanation of your analysis here, in 6 sentences or fewer. Focus on key anomalies, inconsistencies, or red flags in the data. If there are no such concerns, explain why the financials appear sound.
</reasoning>
<answer>
  Write only one label here: either Fraud or Not Fraud.
</answer>

8. Be precise, rigorous, and objective in your evaluation.
"""


USER_PROMPT = """
The company operates in the {industry_title} sector.
final
{mda_text}
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


class LLM_MDAOnlyClassiferGRPO(GRPOMixin, LLM_MDAOnlyClassifier):
    def __init__(self, config):
        """
        Initialize the LLM classifier.
        """
        if not "experiments_dir" in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_fraud_classifier_dataset_{version}_only_mda_grpo{'_oversample' if oversample else ''}"
            )

        # Call parent constructor
        super().__init__(
            config=config,
        )

    def truncate_and_format_prompt(self, mda_text, industry_title):
        """
        Ensure the prompt is within the model's context length.

        Args:
            mda_text (str): The MDA text to include in the prompt.
            industry_title (str): The industry title based on SIC code.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # Tokenize the prompt with MDA data
        prompt_tokens = self.tokenizer(
            mda_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
            - SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE,  # Reserve space
            padding=False,
        )

        # Decode truncated prompt back to text
        truncated_mda = self.tokenizer.decode(
            prompt_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Format user prompt with MDA data
        formatted_user_prompt = USER_PROMPT.format(
            mda_text=truncated_mda, industry_title=industry_title
        )

        return LLM_SYSTEM_PROMPT, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """Generate and tokenize the prompt for training using only MDA data."""
        try:
            mda_content = self.load_mda_content(row["mda_quarter_id"])
            label = "Fraud" if row["is_fraud"] else "Not Fraud"

            # Get the SIC code and industry title
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index.get(sic, "Unknown Industry")

            # Create the full prompt
            system_prompt, user_prompt = self.truncate_and_format_prompt(
                mda_content, industry_title
            )

            # Format using chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            logging.debug(f"Full prompt: {full_prompt}")
            result = {
                "text": full_prompt,
                "prompt": full_prompt,
                "answer": label,
                "sic": sic,
                "cik": row["cik"],
                "quarter": row["quarter"],
                "glabels": str(row.get("glabels", "no_fraud")),
            }

            return result
        except FileNotFoundError:
            logging.warning(
                f"MDA file for quarter ID {row['mda_quarter_id']} not found, skipping"
            )
            return None

    def load_data(self, train_path=None, test_path=None):
        """
        Load train and test datasets.

        Args:
            train_path (Path, optional): Path to training data.
            test_path (Path, optional): Path to test data.
        """
        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)

        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Process data with custom method (to be overridden by subclasses)
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        splitter = get_train_test_splitter(self.config)
        train_df, val_df = splitter(train_df, test_size=0.1, random_state=SEED_TRAINING)

        return train_df, val_df, test_df


def train_and_evaluate_mda_grop_model(config=None):
    """
    Train and evaluate the LLM MDA-only classifier.
    """
    train_path, test_path = load_cross_validation_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_MDAOnlyClassiferGRPO,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":
    CONFIG = {
        "model_url": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "max_context": 1000,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 8,
        "num_train_epochs": 5,
        "learning_rate": 5e-5,
        "save_steps": 5,
        "max_new_tokens": 256,
    }

    train_and_evaluate_mda_grop_model(CONFIG)
