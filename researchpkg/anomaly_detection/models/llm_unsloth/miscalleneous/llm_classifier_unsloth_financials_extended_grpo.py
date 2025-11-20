import logging
import os

import pandas as pd

from researchpkg.anomaly_detection.config import EXPERIMENTS_DIR, SEED_TRAINING
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financials_extended import (
    LLM_FinancialsClassifier,
)

LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]

from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.grpo_interface import GRPOMixin
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financials_extended import (
    LLM_FinancialsClassifier,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    llm_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import (
    get_train_test_splitter,
    load_cross_validation_path
)

SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE = 500

LLM_SYSTEM_PROMPT = """You are a financial analyst specializing in corporate fraud detection. You will be given a structured set of financial indicators derived from a company's quarterly financial statements.
z
Each financial variable includes:
- its **name**,
- its **numerical value** (in absolute terms or percentages),


Additionally, the company’s **industry sector** will be provided to help you interpret the financial context appropriately.

Your task is to thoroughly examine the entire set of financial variables and determine whether there are signs of fraudulent activity during the reported quarter. No single variable is sufficient to conclude fraud; you must consider **the overall pattern of the data**.

Please follow these strict instructions:

1. Carefully analyze the full financial profile in context.
2. Reason logically and concisely based on the data presented.
3. Then respond using **only one** of the following two labels:
   - **"Fraud"** → if you have sufficient indicators suggesting the company likely committed fraud during the quarter.
   - **"Not Fraud"** → if you find no clear evidence or pattern suggesting fraud.

4. Your answer must be formatted **strictly in the following XML structure**:

<reasoning>
  Write a concise explanation of your analysis here, in 8 sentences or fewer. Focus on key anomalies, inconsistencies, or red flags in the data. If there are no such concerns, explain why the financials appear sound.
</reasoning>
<answer>
  Write only one label here: either Fraud or Not Fraud.
</answer>

⚠️ Do not include any text outside the XML tags. Avoid generic or vague reasoning. The reasoning must be directly based on the financial variables provided.

Remember: you are acting as a professional fraud analyst. Be precise, rigorous, and objective in your evaluation.
"""

LLM_SYSTEM_PROMPT_DEEP_SEEK = LLM_SYSTEM_PROMPT.replace(
    "<reasoning>", "<think>"
).replace("</reasoning>", "</think>")


USER_PROMPT = """
The company operates in the {industry_title} sector.

### Financial Features:
{financials_str}
"""


class LLM_ExtendedFinancialsClassifierGRPO(GRPOMixin, LLM_FinancialsClassifier):
    """
    GRPO-based LLM Fraud classifier using extended financial data
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
                / f"llm_fraud_classifier_dataset_{version}_extended_fin_grpo"
                f"{'_oversample' if oversample else ''}{'_undersample' if undersample else ''}"
            )

        # Call parent constructor
        super().__init__(config=config)

    def truncate_and_format_prompt(self, financials_str, industry_title):
        """
        Ensure the prompt is within the model's context length.

        Args:
            financials_str (str): The financial data as a formatted string.
            industry_title (str): The industry title based on SIC code.

        Returns:
            tuple: (system_prompt, user_prompt)
        """

        # Tokenize the prompt with financial data
        prompt_tokens = self.tokenizer(
            financials_str,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
            - SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE,  # Reserve space
            padding="max_length",
        )

        # Decode truncated prompt back to text
        truncated_financials = self.tokenizer.decode(
            prompt_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Format user prompt with financial data
        formatted_user_prompt = USER_PROMPT.format(
            financials_str=truncated_financials, industry_title=industry_title
        )

        is_deep_seek = "deepseek" in self.config["model_url"].lower()

        llm_system_prompt = (
            LLM_SYSTEM_PROMPT_DEEP_SEEK if is_deep_seek else LLM_SYSTEM_PROMPT
        )
        return llm_system_prompt, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """Generate and tokenize the prompt for training using only financial data."""
        financials = self.prepare_financial_data(row)
        financials_str = self.format_financials(financials)
        label = "Fraud" if row["is_fraud"] else "Not Fraud"

        # Get the SIC code and industry title
        sic = str(row["sic"]).zfill(4)
        industry_title = self.sic_index[sic]

        # Create the full prompt
        system_prompt, user_prompt = self.truncate_and_format_prompt(
            financials_str, industry_title
        )

        # Format using chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": label},
        ]

        

        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        partial_prompt = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False
        )
        

        result = {
            "text": full_prompt,
            "prompt": partial_prompt,
            "answer": label,
            "sic": sic,
            "sicagg": row["sicagg"],
            "cik": row["cik"],
            "quarter": f"{row['year']}{row['quarter']}",
            "glabels": str(row.get("glabels", "no_fraud")),
        }

        return result

    def load_data(self, train_path, test_path):
        """
        Load train and test datasets.

        Args:
            train_path (Path, optional): Path to training data.
            test_path (Path, optional): Path to test data.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)

        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Process data with custom method (to be overridden by subclasses)
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        splitter = get_train_test_splitter(self.config)
        train_df, val_df = splitter(train_df, test_size=0.1, seed=SEED_TRAINING)

        if self.config.get("oversample", False):
            # Oversample the training data
            logging.info("Oversampling fraud cases in training data...")
            train_df = self.oversample_fraud_cases(train_df)

        return train_df, val_df, test_df

    def load_model_and_tokenizer(self,fast_inference=False):
        """
        Load the LLM model and tokenizer from Hugging Face.
        """
        super().load_model_and_tokenizer(fast_inference=fast_inference)


def train_and_evaluate_extended_financials_grpo_model(config=None):
    """
    Train and evaluate the LLM extended financials classifier with GRPO.
    """
    train_path, test_path = load_cross_validation_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_ExtendedFinancialsClassifierGRPO,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":

    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    CONFIG = {
        "model_url": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_context": 2200,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 6,
        "num_train_epochs": 20,
        "learning_rate": 5e-5,
        "save_steps": 5,
        "max_new_tokens": 512,
    }
    train_and_evaluate_extended_financials_grpo_model(CONFIG)
