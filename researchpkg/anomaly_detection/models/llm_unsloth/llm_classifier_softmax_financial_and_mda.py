import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_DECHOW,
    FINANCIALS_DIR_EXTENDED,
    LIST_MISTATEMENT_TYPE_RENAMED,
    PREPROCESSED_PATH,
    SEED_TRAINING,
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

# Constants
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"

MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"
MDA_PATH_SUMMARIZED_PARAGRAPHS = (
    PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED_PARAGRAPHS" / "quarterly"
)

FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
DECHOW_FIN_PATH = FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv"

EXCLUDED_FINANCIALS_FEATURES = set(
    [
        BENEISH_PROBM  # Too much biasing the model as it a probability of earnings manipulation
    ]
)

CURRENCY_FEATURES = set(AGGREGATE_FEATURES + DIFF_FEATURES + IMPORTANT_TAGS)
CURRENCY_FEATURES.add(RATIO_NET_WORKING_CAPITAL)

PERCENTAGE_FEATURES = set(RATIO_FEATURES) - set([RATIO_NET_WORKING_CAPITAL])

SYSTEM_INSTRUCTION_AND_TURN_COUNT = 210

USER_PROMPT = """
The company operates in the {industry_title} sector.

Here are financial variables derived from the income statement, balance sheet, and cash flow statement of the company.
{financials_str}

Also below is the  structured summary of the Management Discussion and Analysis (MDA) section of the quarterly report:
{mda_content}

Based on these informations and your knowledge of typical red flags in financial reporting, \
assess whether there is a high likelihood that this company is Financial Fraud.

Do you think this company is engaging Fraud? Answer with "YES" or "NO"?
"""


def is_with_currency(feature):
    return feature in CURRENCY_FEATURES


class LLMFinancialAndMDASoftmaxClassifier(LLMClassifierSoftmax):
    """
    LLM classifier using softmax with both financial data and MDA text for fraud detection.
    """

    def __init__(self, config):
        """
        Initialize the financial and MDA softmax classifier.

        Args:
            config (dict): Configuration dictionary
        """
        # Set up experiment directory if not provided
        if "experiments_dir" not in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            use_focal_loss = config.get("use_focal_loss", False)
            is_summarized_mda = config.get("summarized_mda", True)
            use_weighted_loss = config.get("loss_weight", False)
            only_completion = config.get("only_completion", True)
            use_full_summary = config.get("use_full_summary", False)

            use_dechow = config.get("use_dechow", False)
            use_dechow_and_extended = config.get("use_dechow_and_extended", False)
            force_random_split = config.get("force_random_split", False)
            assert not (
                use_dechow and use_dechow_and_extended
            ), "You cannot set both use_dechow and use_dechow_and_extended to True."

            assert not (
                oversample and undersample
            ), "You cannot set both undersample and oversample args to True."

            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_softmax_fraud_classifier_dataset_{version}_fin_mda"
                f"{'_summarized' if is_summarized_mda else ''}"
                f"{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
                f"{'_focal_loss' if use_focal_loss else ''}"
                f"{'_weighted_loss' if use_weighted_loss else ''}"
                f"{'_not_only_completion' if not only_completion else ''}"
                f"{'_full_summary' if use_full_summary else '_paragraphs'}"
                f"{'_use_dechow' if use_dechow else ''}"
                f"{'_use_dechow_and_extended' if use_dechow_and_extended else ''}"
                f"{'_force_random_split' if force_random_split else ''}"
            )

        # Call parent constructor
        super().__init__(config)

        # Load SIC index and company mapping
        self.sic_index = load_sic_industry_title_index()
        self.cik_company_name_mapping = load_cik_company_mapping(
            dataset_version=self.config.get("dataset_version", "v3")
        )

    def _process_loaded_data(self, df):
        """Process and merge  financial data with the dataset."""
        # Load financial data if not already loaded
        if not hasattr(self, "_full_financials_df"):
            # logging.info(f"Loading full financials data from {FULL_FINANCIAL_PATH}")

            if self.config.get("use_dechow_and_extended", False):
                self._full_financials_df = pd.read_csv(
                    FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv",
                    usecols=["cik", "year", "quarter"] + DECHOW_FEATURES,
                )

                full_df_extended = pd.read_csv(
                    FULL_FINANCIAL_PATH,
                    usecols=["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES,
                )
                self._full_financials_df = pd.merge(
                    self._full_financials_df,
                    full_df_extended,
                    on=["cik", "year", "quarter"],
                )

            elif self.config.get("use_dechow", False):
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
            row (pd.Series): A row from the DataFrame containing financial data

        Returns:
            dict: Financial data dictionary
        """
        financial_data = {}
        if self.config.get("use_dechow_and_extended", False):
            for feature in DECHOW_FEATURES + EXTENDED_FINANCIAL_FEATURES:
                financial_data[feature] = row[feature]
        elif self.config.get("use_dechow", False):
            for feature in DECHOW_FEATURES:
                financial_data[feature] = row[feature]
        else:
            for feature in filter(
                lambda x: x not in EXCLUDED_FINANCIALS_FEATURES,
                EXTENDED_FINANCIAL_FEATURES,
            ):
                financial_data[feature] = row[feature]
        return financial_data

    def format_financials(self, financials, drop_rate=0):
        """
        Format financial data as a string for the model.

        Args:
            financials (dict): Dictionary of financial features

        Returns:
            str: Formatted financial features string
        """

        def display_financial_value(value):
            """Format the value in a readable form."""
            if value == 0:
                return "0"
            elif abs(value) < 10:
                return "{:.2f}".format(value)
            else:
                return "{:,.0f}".format(value)

        # Filter out zero values and NaN
        financials = {
            k: v
            for k, v in financials.items()
            if k not in EXCLUDED_FINANCIALS_FEATURES and v != 0 and not pd.isna(v)
        }

        # Drop random keys if requested
        if drop_rate > 0:
            financials = drop_random_keys(financials, drop_rate)

        # Convert ratios to percentages
        for key in financials.keys():
            if key in PERCENTAGE_FEATURES:
                financials[key] = financials[key] * 100

        # Create formatted string
        financials_str = "\n".join(
            [
                f"- {EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT[key]}: {'$' if is_with_currency(key) else ''}{display_financial_value(value)}{'%' if key in PERCENTAGE_FEATURES else ''}"
                for key, value in financials.items()
            ]
        )

        return financials_str

    def load_mda_content(self, mda_quarter_id):
        """
        Load the content of an MDA file.

        Args:
            mda_quarter_id (str): Quarter ID for the MDA file

        Returns:
            str: MDA content
        """
        mda_root_dir = (
            MDA_PATH_SUMMARIZED
            if self.config.get("use_full_summary", False)
            else MDA_PATH_SUMMARIZED_PARAGRAPHS
        )
        mda_path = mda_root_dir if self.config.get("summarized_mda", True) else MDA_PATH
        mda_file = mda_path / f"{mda_quarter_id}.txt"

        if not mda_file.exists():
            logging.warning(f"MDA file {mda_file} does not exist.")
            return "No MDA content available."

        with open(mda_file, "r", encoding="utf-8") as file:
            mda_content = file.read()
            return mda_content

    def truncate_and_format_prompt(self, financials_str, mda_content, industry_title):
        """
        Ensure the prompt is within the model's context length.

        Args:
            financials_str (str): The financial data as a formatted string.
            mda_content (str): The MDA content.
            industry_title (str): The industry title based on SIC code.

        Returns:
            str: Formatted user promptcore_çonten
        """
        # Create full prompt
        full_prompt = USER_PROMPT.format(
            financials_str=financials_str,
            mda_content=mda_content,
            industry_title=industry_title,
        )

        # Tokenize the prompt
        prompt_tokens = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - SYSTEM_INSTRUCTION_AND_TURN_COUNT,
            padding=False,
        )

        # Decode truncated prompt back to text
        truncated_prompt = self.tokenizer.decode(
            prompt_tokens["input_ids"][0], skip_special_tokens=True
        )

        return truncated_prompt

    def generate_prompt(self, row, idx=None, drop_rate=None, **kwargs):
        """
        Generate a prompt combining financial data and MDA text.

        Args:
            row (pd.Series): Row from DataFrame
            idx (int, optional): Index of the example
            drop_rate (float, optional): Rate to drop financial features

        Returns:
            dict: Dictionary with prompts and metadata for training
        """
        try:
            # Get financial data
            financials = self.prepare_financial_data(row)

            if self.config.get("use_dechow_and_extended", False):
                financials_str = (
                    self.format_financials(
                        {
                            k: v
                            for k, v in financials.items()
                            if k in EXTENDED_FINANCIAL_FEATURES
                        },
                        drop_rate=drop_rate if drop_rate is not None else 0.0,
                    )
                    + "\n"
                    + self.format_financials_dechow(
                        {k: v for k, v in financials.items() if k in DECHOW_FEATURES},
                        drop_rate=drop_rate if drop_rate is not None else 0.0,
                    )
                )

            elif self.config.get("use_dechow", False):
                financials_str = self.format_financials_dechow(financials, drop_rate)
            else:
                financials_str = self.format_financials(financials)

            # Get MDA content
            mda_content = self.load_mda_content(row["mda_quarter_id"])

            # Get industry information
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index.get(sic, "Unknown Industry")

            # Convert the label
            label = "Fraud" if row["is_fraud"] else "Not Fraud"
            label = label_to_token(label)

            # Create the full prompt
            user_prompt = self.truncate_and_format_prompt(
                financials_str, mda_content, industry_title
            )
            full_prompt = user_prompt + COMPLETION_INSTRUCTION + label

            core_content = """{financials_str}
            
Also below is the  structured summary of the Management Discussion and Analysis (MDA) section of the quarterly report:
{mda_content}""".format(
                financials_str=financials_str, mda_content=mda_content
            )
        
            result = {
                "text": full_prompt,
                "core_content": core_content,
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

        except KeyError as e:
            logging.warning(f"Missing key in row: {e}")
            return None
        except FileNotFoundError:
            logging.warning(
                f"MDA file for quarter ID {row.get('mda_quarter_id', 'unknown')} not found"
            )
            return None


def train_and_evaluate_financial_and_mda_softmax_model(config=None):
    """
    Train and evaluate the financial and MDA softmax classifier.

    Args:
        config (dict): Configuration for the model

    Returns:
        tuple: (model, accuracy, report)
    """
    if config is None:
        raise ValueError("Configuration must be provided")

    train_path, test_path = load_cross_validation_path(config)

    return llm_softmax_train_and_evaluate_base_model(
        model_class=LLMFinancialAndMDASoftmaxClassifier,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":
    CONFIG = {
        "model_url": "mistralai/Mistral-7B-v0.1",
        "model_alias": "Mistral-7B",
        "max_context": 4096,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "dataset_version": "v3",
        "fold_id": 0,
        "summarized_mda": True,
    }

    train_and_evaluate_financial_and_mda_softmax_model(CONFIG)
