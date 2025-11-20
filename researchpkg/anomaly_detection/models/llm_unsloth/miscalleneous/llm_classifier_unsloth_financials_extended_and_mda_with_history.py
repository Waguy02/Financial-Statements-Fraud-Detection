import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import unsloth  # noqa
from tqdm import tqdm

from researchpkg.anomaly_detection.config import EXPERIMENTS_DIR
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financial_extended_and_mda import (
    LLM_SYSTEM_PROMPT,
    MDA_PATH,
    MDA_PATH_SUMMARIZED,
    LLM_FinancialAndMDAClassifier,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    EXCLUDED_FINANCIALS_FEATURES,
    llm_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import (
    drop_random_keys,
    load_train_test_path,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
)

LLM_SYSTEM_PROMPT_WITH_HISTORY = """You are a financial analyst specializing in corporate fraud detection. You will be given two types of information:
1. The Management Discussion and Analysis (MDA) section from a company's quarterly financial report
2. A structured set of financial indicators derived from the company's quarterly financial statements
3. Historical financial indicators from the company's previous quarters

The MDA text contains management's narrative explanation of the company's financial condition, results of operations, and future prospects.

Each financial variable includes its name and numerical value (in absolute terms or percentages).

Additionally, the company's **industry sector** will be provided to help you interpret the financial context appropriately.

Your task is to thoroughly examine the MDA text, current financial data, and historical data to determine whether there are signs of fraudulent activity (Financial statement fraud) during the reported quarter. Look for irregularities, inconsistencies, or discrepancies that may indicate fraud.

Please follow these strict instructions:

1. Carefully analyze both the MDA text and the financial data in context, looking for any inconsistencies between them.
2. Consider trends and changes when comparing current financial data with historical data from previous quarters.
3. If there is any ambiguity between the MDA and financial data, prioritize the MDA text as the most reliable source of information.
4. Reason logically and concisely based on the information presented.
5. Then respond using **only one** of the following two labels:
   - "Fraud" → if you have sufficient indicators suggesting the company likely committed fraud during the quarter.
   - "Not Fraud" → if you find no clear evidence or pattern suggesting fraud.

6. Remember: you are acting as a professional fraud analyst. Be precise, rigorous, and objective in your evaluation. Respond only with the label, without any additional explanations or justifications. Do not provide any other information or context.
"""

USER_PROMPT_WITH_HISTORY = """
The company operates in the {industry_title} sector.
The current quarter is {current_quarter}.

### Content of the MDA Section:
{mda_text}

### Current quarter's Financial Features:
{financials_str}

### Last {n_quarters} quarters of Financial Features:
{last_financials_str}
"""


class LLM_FinancialAndMDAClassifierWithHistory(LLM_FinancialAndMDAClassifier):
    """
    LLM Fraud classifier using both extended financial data, MDA text, and historical data
    """

    def __init__(self, config):
        """
        Initialize the LLM classifier.
        """
        # Override default experiments directory
        if "experiments_dir" not in config:
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            assert not (
                oversample and undersample
            ), "You cannot set both undersample and oversample args to True."
            version = config.get("dataset_version", "v3")
            is_summarized_mda = config.get("raw_mda", False)
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_fraud_classifier_dataset_{version}_extended_fin_and_mda_with_history"
                f"{'_summarized' if is_summarized_mda else ''}"
                f"{'_oversample' if oversample else ''}{'_undersample' if undersample else ''}"
            )

        # Prepare last financials data as a dictionary indexed by cik and quarter
        self.last_financials_dict = {}

        # Call parent constructor
        super().__init__(config=config)

    def _process_loaded_data(self, df):
        """Process and merge financial data with the dataset, including historical data."""
        # Call parent method to get the basic processed dataframe
        df = super()._process_loaded_data(df)

        # Convert quarter strings to numerical values for sorting
        df["quarter_num"] = df["quarter"].apply(lambda x: int(x.split("q")[1]))
        df["quarter_num"] = df["quarter_num"].astype(int)

        self._full_financials_df["quarter_num"] = self._full_financials_df[
            "quarter"
        ].apply(lambda x: int(x.split("q")[1]))
        self._full_financials_df["quarter_num"] = self._full_financials_df[
            "quarter_num"
        ].astype(int)

        # Build history lookup dictionary
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Getting historical financials"
        ):
            cik = row["cik"]
            year = row["year"]
            quarter = row["quarter"]
            quarter_num = row["quarter_num"]

            # Create a unique key for this entry
            key = f"{cik}_{year}_{quarter}"

            # Get previous quarters for this company
            prev_data = (
                self._full_financials_df.query(
                    f"cik == {cik} and (year == {year} and quarter_num < {quarter_num}) or (year < {year})"
                )
                .sort_values(by=["year", "quarter_num"], ascending=[False, False])
                .drop_duplicates(subset=["year", "quarter_num"])
            )

            # Get the last 4 quarters, sorted from most recent to oldest
            self.last_financials_dict[key] = prev_data.head(4).to_dict(orient="records")

        return df

    def format_last_financials(self, cik, year, quarter, drop_rate=0):
        """
        Format historical financial data for the prompt.

        Args:
            cik: Company identifier
            year: Year of current financial data
            quarter: Quarter of current financial data
            drop_rate: Percentage of features to randomly drop

        Returns:
            Formatted string of historical financial data and number of quarters included
        """
        # Create the lookup key
        key = f"{cik}_{year}_{quarter}"

        if key not in self.last_financials_dict:
            return "", 0

        last_financials = self.last_financials_dict[key]
        n_quarters = len(last_financials)

        if n_quarters == 0:
            return "", 0

        quarters_data = []

        for quarter_data in last_financials:
            year, quarter = quarter_data.get("year"), quarter_data.get("quarter")
            # Filter the actual financial data
            quarter_fin_data = {
                k: v
                for k, v in quarter_data.items()
                if k in EXTENDED_FINANCIAL_FEATURES
            }

            # Filter out excluded features
            quarter_fin_data = {
                k: v
                for k, v in quarter_fin_data.items()
                if k not in EXCLUDED_FINANCIALS_FEATURES and v != 0 and not np.isnan(v)
            }

            # Apply drop rate if needed
            if drop_rate > 0:
                quarter_fin_data = drop_random_keys(quarter_fin_data, drop_rate)

            # Format the quarter data
            quarter_str = self.format_financials(quarter_fin_data)
            year_quarter = f"Year: {year}, Quarter: {quarter}"
            quarters_data.append(f"#### {year_quarter}\n{quarter_str}")

        return "\n\n".join(quarters_data), n_quarters

    def truncate_and_format_prompt(
        self,
        mda_text,
        financials_str,
        industry_title,
        cik=None,
        year=None,
        quarter=None,
    ):
        """
        Ensure the prompt is within the model's context length, including historical data.

        Args:
            mda_text (str): The MDA text to include in the prompt.
            financials_str (str): The financial data as a formatted string.
            industry_title (str): The industry title based on SIC code.
            cik (str): Company identifier.
            year (str): Year of current financial data.
            quarter (str): Quarter of current financial data.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # Get historical financial data
        last_financials_str, n_quarters = self.format_last_financials(
            cik, year, quarter
        )

        # Reserve tokens for system prompt and other components
        system_prompt_tokens = 500
        metadata_tokens = 100  # For industry title and other metadata

        # Calculate token budgets - we'll distribute the available tokens proportionally
        total_available_tokens = (
            self.max_length - system_prompt_tokens - metadata_tokens
        )

        # Allocate tokens to MDA, current financials, and historical financials
        # We'll give MDA text a larger portion since it typically contains more relevant information
        mda_tokens_budget = int(total_available_tokens * 0.5)  # 50% for MDA
        current_fin_tokens_budget = int(
            total_available_tokens * 0.25
        )  # 25% for current financials
        history_tokens_budget = int(
            total_available_tokens * 0.25
        )  # 25% for historical financials

        # Tokenize and truncate each component
        mda_tokens = self.tokenizer(
            mda_text,
            return_tensors="pt",
            truncation=True,
            max_length=mda_tokens_budget,
            padding=False,
        )
        truncated_mda = self.tokenizer.decode(
            mda_tokens["input_ids"][0], skip_special_tokens=True
        )

        current_fin_tokens = self.tokenizer(
            financials_str,
            return_tensors="pt",
            truncation=True,
            max_length=current_fin_tokens_budget,
            padding=False,
        )
        truncated_financials = self.tokenizer.decode(
            current_fin_tokens["input_ids"][0], skip_special_tokens=True
        )

        history_fin_tokens = self.tokenizer(
            last_financials_str,
            return_tensors="pt",
            truncation=True,
            max_length=history_tokens_budget,
            padding=False,
        )
        truncated_history = self.tokenizer.decode(
            history_fin_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Current quarter information
        current_quarter = f"Year: {year}, Quarter: {quarter}"

        # Format user prompt with all data
        formatted_user_prompt = USER_PROMPT_WITH_HISTORY.format(
            mda_text=truncated_mda,
            financials_str=truncated_financials,
            industry_title=industry_title,
            current_quarter=current_quarter,
            last_financials_str=truncated_history,
            n_quarters=n_quarters,
        )

        return LLM_SYSTEM_PROMPT_WITH_HISTORY, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """Generate and tokenize the prompt for training using both MDA and financial data with history."""
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

            # Create the full prompt with both data sources and history
            system_prompt, user_prompt = self.truncate_and_format_prompt(
                mda_content,
                financials_str,
                industry_title,
                cik=row["cik"],
                year=row["year"],
                quarter=row["quarter"],
            )

            # Format using chat template
            if "deepseek" in self.config["model_url"].lower():
                label = f"<think>{label}</think>"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label},
            ]

            # Combine into the full prompt
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
                f"MDA file for quarter ID {row['mda_quarter_id']} not found, skipping"
            )
            return None


def train_and_evaluate_financial_and_mda_model_with_history(config=None):
    """
    Train and evaluate the LLM classifier using both financial data, MDA text, and historical data.
    """
    # Ensure the configuration has history enabled
    config["include_last_4_quarters"] = True

    train_path, test_path = load_train_test_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_FinancialAndMDAClassifierWithHistory,
        config=config,
        train_path=train_path,
        test_path=test_path,
    )


if __name__ == "__main__":
    CONFIG = {
        "model_url": "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "max_context": 8000,
        "max_new_tokens": 20,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 5,
        "learning_rate": 5e-5,
        "save_steps": 5,
    }

    train_and_evaluate_financial_and_mda_model_with_history(CONFIG)
