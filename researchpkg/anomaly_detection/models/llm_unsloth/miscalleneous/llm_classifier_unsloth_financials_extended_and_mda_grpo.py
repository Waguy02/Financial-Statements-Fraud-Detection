import logging
import os

import pandas as pd
import unsloth  # noqa

from researchpkg.anomaly_detection.config import EXPERIMENTS_DIR, SEED_TRAINING
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.grpo_interface import GRPOMixin
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financial_extended_and_mda import (
    FULL_FINANCIAL_PATH,
    LLM_FinancialAndMDAClassifier,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    llm_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import (
    get_train_test_splitter,
    load_cik_company_mapping,
    load_sic_industry_title_index,
    load_train_test_path,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
)

SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE = 600
LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]

LLM_SYSTEM_PROMPT = """You are a financial analyst specializing in corporate fraud detection. You will be given two types of information:
1. The Management Discussion and Analysis (MDA) section from a company's quarterly financial report
2. A structured set of financial indicators derived from the company's quarterly financial statements

The MDA text contains management's narrative explanation of the company's financial condition, results of operations, and future prospects.

Each financial variable includes its name and numerical value (in absolute terms or percentages).

Additionally, the company's **industry sector** will be provided to help you interpret the financial context appropriately.

Your task is to thoroughly examine both the MDA text and financial data to determine whether there are signs of fraudulent activity (Financial statement fraud) during the reported quarter. Look for irregularities, inconsistencies, or discrepancies that may indicate fraud.

Please follow these strict instructions:

1. Carefully analyze both the MDA text and the financial data in context, looking for any inconsistencies between them.
2. If there is any ambiguity between the MDA and financial data, prioritize the MDA text as the most reliable source of information.
3. Reason logically and concisely based on the information presented.
4. You should reason step by step to provide the final answer (at most 6 sentences in your reasoning).
5. Then respond using **only one** of the following two labels:
   - "Fraud" → if you have sufficient indicators suggesting the company likely committed fraud during the quarter.
   - "Not Fraud" → if you find no clear evidence or pattern suggesting fraud.

6. Remember that no single variable is sufficient to conclude fraud; you must consider **the overall pattern of the data**.

7. Your answer must be formatted **strictly in the following XML structure**:

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

### Content of the MDA Section:
{mda_text}

### The financial variables:
{financials_str}
"""

LLM_SYSTEM_PROMPT_DEEP_SEEK = LLM_SYSTEM_PROMPT.replace(
    "<reasoning>", "<think>"
).replace("</reasoning>", "</think>")


class LLM_FinancialAndMDAClassifierGRPO(GRPOMixin, LLM_FinancialAndMDAClassifier):
    """
    GRPO-based LLM Fraud classifier using both extended financial data and MDA text
    """

    def __init__(self, config):
        """
        Initialize the GRPO-based LLM classifier.
        """
        if "experiments_dir" not in config:
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            assert not (
                oversample and undersample
            ), "You cannot set both undersample and oversample args to True."
            version = config.get("dataset_version", "v3")
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_fraud_classifier_dataset_{version}_extended_fin_and_mda_grpo{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
            )

        # Call parent constructor
        super().__init__(config=config)

        # Load SIC index and company mapping
        self.sic_index = load_sic_industry_title_index()
        self.cik_company_name_mapping = load_cik_company_mapping(
            dataset_version=self.config["dataset_version"]
        )

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
            - SYSTEM_AND_USER_PROMPT_COUNT_NO_INTERPOLATE
            - self.config["max_new_tokens"]
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

        is_deep_seek = "deepseek" in self.config["model_url"].lower()

        llm_system_prompt = (
            LLM_SYSTEM_PROMPT_DEEP_SEEK if is_deep_seek else LLM_SYSTEM_PROMPT
        )

        return llm_system_prompt, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """Generate and tokenize the prompt for training using both MDA and financial data."""
        try:
            # Load MDA content
            mda_content = self.load_mda_content(row["mda_quarter_id"])

            # Prepare financial data
            financials = self.prepare_financial_data(row)
            financials_str = self.format_financials(financials)

            # Get the SIC code and industry title
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index[sic]

            # Get the label
            label = "Fraud" if row["is_fraud"] else "Not Fraud"

            # Create the full prompt with both data sources
            system_prompt, user_prompt = self.truncate_and_format_prompt(
                mda_content, financials_str, industry_title
            )

            # Format using chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label},
            ]
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            result = {
                "text": full_prompt,
                "prompt": full_prompt,
                "answer": label,
                "sic": sic,
                "cik": row["cik"],
                "quarter": f"{row['year']}{row['quarter']}",
                "glabels": str(row.get("glabels", "no_fraud")),
            }

            return result
        except FileNotFoundError:
            logging.warning(
                f"MDA file for quarter ID {row['mda_quarter_id']} not found, skipping"
            )
            return None

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
        train_df, val_df = splitter(train_df, test_size=0.1)

        return train_df, val_df, test_df


def train_and_evaluate_financial_and_mda_grpo_model(config=None):
    """
    Train and evaluate the LLM classifier using both financial data and MDA text with GRPO.
    """
    train_path, test_path = load_train_test_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_FinancialAndMDAClassifierGRPO,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":
    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    CONFIG = {
        "model_url": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_context": 12000,
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

    train_and_evaluate_financial_and_mda_grpo_model(CONFIG)
