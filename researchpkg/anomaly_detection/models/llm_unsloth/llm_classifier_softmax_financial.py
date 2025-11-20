import logging

import pandas as pd

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_DECHOW,
    FINANCIALS_DIR_EXTENDED,
    LIST_MISTATEMENT_TYPE_RENAMED,
)
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_softmax_unsloth import (
    COMPLETION_INSTRUCTION,
    LLMClassifierSoftmax,
    label_to_token,
    llm_softmax_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import (
    drop_random_keys,
    load_cik_company_mapping,
    load_cross_validation_path,
    load_sic_industry_title_index,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_dechow import (
    DECHOW_FEATURES,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (  # EXTENDED_FINANCIAL_FEATURES,
    BENEISH_PROBM,
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
)

# Constants
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
DECHOW_FIN_PATH = FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv"

SYSTEM_INSTRUCTION_AND_TURN_COUNT = 0

EXCLUDED_FINANCIALS_FEATURES = set(
    [
        BENEISH_PROBM  # Too much biasing the model as it is a probability of earnings manipulation
    ]
)


USER_PROMPT = """
You are a financial forensic analyst.  
The company operates in the **{industry_title}** sector. Below are key financial indicators derived from its income statement, balance sheet, and cash flow statement:  
{financials_str}  
Based on these informations and your knowledge of typical red flags in financial reporting, \
assess whether there is a high likelihood that this company is engaging in Financial Manipulation Fraud. \
Do you think this company is engaging Fraud? Answer with "YES" or "NO"?
"""


class LLM_Softmax_FinancialsClassifier(LLMClassifierSoftmax):
    """
    LLM Fraud classifier using financial data
    """

    def __init__(self, config):
        """
        Initialize the LLM classifier.
        """
        # Override default experiments directory
        if not "experiments_dir" in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            only_completion = config.get("only_completion", True)
            use_dechow = config.get("use_dechow", False)
            assert not (
                oversample and undersample
            ), "You cannot set both undersample and oversample args to True."

            use_last_4_quarters = config.get("include_last_4_quarters", False)

            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_softmax_fraud_classifier_dataset_{version}_fin{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}{'_with_last_4_quarters' if use_last_4_quarters else ''}"
                f"{'_not_only_completion' if not only_completion else ''}"
                f"{'_use_dechow' if use_dechow else ''}"
            )

        # Call parent constructor
        super().__init__(
            config=config,
        )

        # Load SIC index and company mapping
        self.sic_index = load_sic_industry_title_index()
        self.cik_company_name_mapping = load_cik_company_mapping(
            dataset_version=self.config["dataset_version"]
        )

    def load_data(self, train_path, test_path):
        """
        Load train and test datasets, merging with financial data.

        Args:
            train_path (Path, optional): Path to training data.
            test_path (Path, optional): Path to test data.

        Returns:
            tuple: (train_df, val_df, test_df)
        """

        return super().load_data(train_path, test_path)

    def _process_loaded_data(self, df):
        """Process and merge  financial data with the dataset."""
        # Load financial data if not already loaded
        if not hasattr(self, "_full_financials_df"):
            # logging.info(f"Loading full financials data from {FULL_FINANCIAL_PATH}")

            if self.config.get("use_dechow", False):
                self._full_financials_df = pd.read_csv(
                    DECHOW_FIN_PATH,
                    usecols=["cik", "year", "quarter"] + DECHOW_FEATURES,
                )

            else:
                self._full_financials_df = pd.read_csv(
                    FULL_FINANCIAL_PATH,
                    usecols=["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES,
                )
                self._full_financials_df = self._full_financials_df[
                    ["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES
                ]
                # Drop existing feature count columns if present
                df = df.drop(
                    columns=EXTENDED_FINANCIAL_FEATURES_COUNT_COLS, errors="ignore"
                )

            self._full_financials_df = self._full_financials_df.drop_duplicates(
                subset=["cik", "year", "quarter"]
            )

        # Merge with financials data
        df = df.merge(
            self._full_financials_df, on=["cik", "year", "quarter"], how="left"
        )

        return df

    def prepare_financial_data(self, row):
        """
        Extract and prepare financial data from a DataFrame row.

        Args:
            row (pd.Series): A row from the DataFrame containing financial data.

        Returns:
            dict: A dictionary of financial variables.
        """
        financial_data = {}
        if self.config.get("use_dechow", False):
            for feature in DECHOW_FEATURES:
                financial_data[feature] = row[feature]
        else:
            for feature in filter(
                lambda x: x not in EXCLUDED_FINANCIALS_FEATURES,
                EXTENDED_FINANCIAL_FEATURES,
            ):
                financial_data[feature] = row[feature]
        return financial_data

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
            - SYSTEM_INSTRUCTION_AND_TURN_COUNT,  # Reserve space for instructions
            padding=False,
        )

        # Decode truncated prompt back to text
        truncated_financials = self.tokenizer.decode(
            prompt_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Format user prompt with financial data
        formatted_user_prompt = USER_PROMPT.format(
            financials_str=truncated_financials, industry_title=industry_title
        )

        return formatted_user_prompt

    def generate_prompt(self, row, idx=None, drop_rate=None, **kwargs):
        """Generate and tokenize the prompt for training using only financial data."""
        financials = self.prepare_financial_data(row)
        if drop_rate is not None and drop_rate > 0:
            logging.info(f"Dropping {drop_rate*100}% of rows")
            financials = drop_random_keys(financials, drop_rate)

        if self.config.get("use_dechow", False):
            financials_str = self.format_financials_dechow(financials, drop_rate)
        else:
            financials_str = self.format_financials(financials)

        label = "Fraud" if row["is_fraud"] else "Not Fraud"
        label = label_to_token(label)

        # Get the SIC code and industry title
        sic = str(row["sic"]).zfill(4)
        industry_title = self.sic_index[sic]

        # Create the full prompt
        user_prompt = self.truncate_and_format_prompt(financials_str, industry_title)
        full_prompt = user_prompt + COMPLETION_INSTRUCTION + label

        result = {
            "text": full_prompt,
            "core_content": financials_str,
            "answer": label,
            "sic": sic,
            "sicagg": row["sicagg"],
            "cik": row["cik"],
            "quarter": f"{row['year']}{row['quarter']}",
            "misstatements": ";".join(
                [m for m in LIST_MISTATEMENT_TYPE_RENAMED if row[m] == 1]
            ),
        }

        return result


def train_and_evaluate_financial_softmax_model(config):
    """
    Train and evaluate the LLM extended financials classifier.
    """
    train_path, test_path = load_cross_validation_path(config)

    return llm_softmax_train_and_evaluate_base_model(
        model_class=LLM_Softmax_FinancialsClassifier,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":
    CONFIG = {
        "model_url": "unsloth/Llama-3.1-8B-unsloth-bnb-4bit",
        "max_context": 2900,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 14,
        "learning_rate": 5e-5,
        "save_steps": 5,
    }

    train_and_evaluate_financial_softmax_model(CONFIG)
