import json
import logging

import pandas as pd

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    BaseLLMFraudClassifier,
    llm_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import (
    get_train_test_splitter,
    load_cik_company_mapping,
    load_cross_validation_path,
    load_sic_industry_title_index,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    AGGREGATE_FEATURES,
    BENEISH_PROBM,
    DIFF_FEATURES,
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
    IMPORTANT_TAGS,
    RATIO_FEATURES,
    RATIO_NET_WORKING_CAPITAL,
)

# Constants
EXTRA_FINANCIAL_PATH = PREPROCESSED_PATH / "SEC_MDA_FINANCIAL_JSON" / "quarterly"
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"

EXCLUDED_FINANCIALS_FEATURES = set(
    [
        BENEISH_PROBM  # Too much biasing the model as it a probability of earnings manipulation
    ]
)

CURRENCY_FEATURES = set(AGGREGATE_FEATURES + DIFF_FEATURES + IMPORTANT_TAGS)
CURRENCY_FEATURES.add(RATIO_NET_WORKING_CAPITAL)
MAX_FIN_TOKENS = 1000  # Maximum tokens for financial data


def is_with_currency(feature):
    return feature in CURRENCY_FEATURES


PERCENTAGE_FEATURES = set(RATIO_FEATURES) - set([RATIO_NET_WORKING_CAPITAL])

LLM_SYSTEM_PROMPT = """
You are a professional financial analyst specializing in corporate fraud detection. You will receive a structured set of financial indicators derived from a company's quarterly financial statements.

Each financial variable will include:
- its **name**
- its **numerical value** (expressed as an absolute value or percentage)

You will also be provided with the company’s **industry sector** to support contextual interpretation.

Your task is to evaluate the entire financial profile to determine whether there are signs of **financial statement fraud** during the reported quarter. 

Fraud cannot be concluded from a single variable; you must assess the **overall pattern and coherence of the data**.

In addition to the financial variables, you will be given **additional financial information** extracted from the company’s official report. If any discrepancies arise, you must **prioritize the additional financial information** over the original variables.

Please adhere strictly to the following instructions:

1. Carefully analyze the complete financial profile in context.
2. Reason logically and objectively based on the data provided.
3. Respond using **only one** of the two possible labels:
   - **"Fraud"** → if sufficient indicators suggest the company likely engaged in fraudulent activity during the quarter.
   - **"Not Fraud"** → if there is no clear evidence or pattern suggesting fraudulent behavior.

4. Do **not** provide any explanations, justifications, or additional commentary. Your response must consist of **only the selected label**.
"""

USER_PROMPT = """
The company operates in the **{industry_title}** sector.

### Financial Variables:
{financials_str}

### Additional Financial Information:
{additional_financials_str}
"""


class LLM_FinancialAndExtraClassifier(BaseLLMFraudClassifier):
    """
    LLM Fraud classifier using both extended financial data and MDA text
    """

    def __init__(self, config):
        """
        Initialize the LLM classifier.
        """
        # Override default experiments directory
        oversample = config.get("oversample", False)
        undersample = config.get("undersample", False)
        assert not (
            oversample and undersample
        ), "You cannot set both undersample and oversample args to True."
        version = config.get("dataset_version", "v3")
        config["experiments_dir"] = (
            EXPERIMENTS_DIR
            / f"llm_fraud_classifier_dataset_{version}_extended_fin_and_extra"
            f"{'_oversample' if oversample else ''}{'_undersample' if undersample else ''}"
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
        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)

        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Process data with custom method (to be overridden by subclasses)
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)

        if self.config.get("no_validation", False):
            logging.info(
                "No validation set, training on the whole training data without split"
            )
            val_df = train_df[
                :1
            ]  # Dummy validation set for compatibility but no validation in practice
        else:
            splitter = get_train_test_splitter(self.config)
            train_df, val_df = splitter(train_df, test_size=0.1)

        if self.config.get("oversample", False):
            # Oversample the training data
            logging.info("Oversampling fraud cases in training data...")
            train_df = self.oversample_fraud_cases(train_df)

        logging.info("Train data size: %d", len(train_df))
        logging.info("Validation data size: %d", len(val_df))
        logging.info("Test data size: %d", len(test_df))

        return train_df, val_df, test_df

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
        Load the content of a MDA File.
        """
        mda_file = EXTRA_FINANCIAL_PATH / f"{mda_quarter_id}.json"

        if not mda_file.exists():
            raise FileNotFoundError(f"MDA file {mda_file} does not exist.")

        with open(mda_file, "r", encoding="utf-8") as file:
            extra_financial_content = json.load(file)
            # write it as string
            extra_financial_content_str = ""
            for key, value in extra_financial_content.items():
                extra_financial_content_str += f"- {key} : {value}\n"

            return extra_financial_content_str

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

    def truncate_and_format_prompt(
        self, extra_financial_text, financials_str, industry_title
    ):
        """
        Ensure the prompt is within the model's context length.

        Args:
            mda_text (str): The MDA text to include in the prompt.
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

        # Decode truncated financials back to text
        truncated_financials = self.tokenizer.decode(
            financials_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Calculate the maximum tokens for MDA (less than half of available context)
        mda_max_tokens = int(
            self.max_length
            - len(financials_tokens["input_ids"][0])
            - 300  # 300 for instructions
        )

        # Tokenize the MDA text
        mda_tokens = self.tokenizer(
            extra_financial_text,
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
            additional_financials_str=truncated_mda,
            financials_str=truncated_financials,
            industry_title=industry_title,
        )

        return LLM_SYSTEM_PROMPT, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """Generate and tokenize the prompt for training using both MDA and financial data."""
        try:
            # Load MDA content
            exta_financial_content = self.load_extra_financial_content(
                row["mda_quarter_id"]
            )

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
                exta_financial_content, financials_str, industry_title
            )

            # Format using chat template
            if "deepseek" in self.config["model_url"].lower():
                label = f"<think>{label}</think>"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label},
            ]

            # Now add the label to the truncated prompt
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            result = {
                "text": full_prompt,
                "answer": label,
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


def train_and_evaluate_financial_and_extra_model(config=None):
    """
    Train and evaluate the LLM classifier using both financial data and MDA text.
    """
    train_path, test_path = load_cross_validation_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_FinancialAndExtraClassifier,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":
    CONFIG = {
        "model_url": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_context": 10000,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 5,
        "learning_rate": 5e-5,
        "save_steps": 5,
    }

    train_and_evaluate_financial_and_extra_model(CONFIG)
