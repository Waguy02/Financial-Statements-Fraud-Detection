import logging
from pathlib import Path

import numpy as np
import pandas as pd

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
)
from researchpkg.anomaly_detection.models.llm_multilabel.llm_classifier_multilabel import (
    LLMClassifierMultiLabel,
    MultiLabelTextDataset,
)
from researchpkg.anomaly_detection.models.utils import (
    drop_random_keys,
    load_cross_validation_path,
    load_sic_industry_title_index,
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

FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"

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


USER_PROMPT = """
The company operates in the **{industry_title}** sector.
You are provided with financial indicators derived from its income statement, balance sheet, and cash flow statement:
{financials_str}
As a financial forensic analyst, assess whether there is a high likelihood that this company is engaging in financial fraud.
Fraud refers to any intentional misrepresentation or omission of information that could deceive stakeholders, such as investors, creditors, or regulators.
Find patterns, anomalies or inconsistencies in the financial data that may indicate earnings mistatements.
"""


class LLMClassifierMultiLabelFinancial(LLMClassifierMultiLabel):
    def __init__(self, config):

        if not "experiments_dir" in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            assert not (oversample and undersample), "Cannot use both oversampling and undersampling at the same time."
            use_focal_loss = config.get("use_focal_loss", False)
            mode = "binary" if config.get("binary", False) else "multilabel"
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_{mode}_fraud_classifier_dataset_{version}_fin{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
                f"{'_focal_loss' if use_focal_loss else ''}"
            )

        super().__init__(config)
        self.sic_index = load_sic_industry_title_index()
        self._full_financials_df = None

    def _process_data(self, df):
        df = super()._process_data(df)
        inital_size = len(df)
        if self._full_financials_df is None:
            self._full_financials_df = pd.read_csv(FULL_FINANCIAL_PATH)
            self._full_financials_df = self._full_financials_df[
                ["cik", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES
            ]
        df = df.drop(columns=EXTENDED_FINANCIAL_FEATURES_COUNT_COLS, errors="ignore")
        df = df.merge(
            self._full_financials_df, on=["cik", "year", "quarter"], how="left"
        )
        assert len(df) == inital_size, f"Data size mismatch after merging financials: initial {inital_size}, final {len(df)}"
        # Minimal text formatting with financial features
        df["text"] = df.apply(lambda row: self._build_text_prompt(row), axis=1)
        return df

    def _build_text_prompt(self, row):
        sector = self.sic_index.get(str(row.get("sic", "")).zfill(4), "Unknown Sector")

        financials_str = self.format_financials(
            row[EXTENDED_FINANCIAL_FEATURES].to_dict(),
            drop_rate=0.1,  # Adjust dropout rate as needed
        )
        return USER_PROMPT.format(
            industry_title=sector,
            financials_str=financials_str,
        )

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


def train_and_evaluate_financial_multilabel_model(config):
    classifier = LLMClassifierMultiLabelFinancial(config)
    train_path, test_path = load_cross_validation_path(config)
    return classifier.train_and_evaluate(train_path, test_path)


if __name__ == "__main__":
    CONFIG = {
        "model_url": "unsloth/Llama-3.1-8B-unsloth-bnb-4bit",
        "max_context": 1500,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 14,
        "learning_rate": 5e-5,
        "save_steps": 5,
    }

    train_and_evaluate_financial_multilabel_model(CONFIG)