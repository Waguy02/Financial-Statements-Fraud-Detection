import logging

import numpy as np
import pandas as pd
import unsloth
from tqdm import tqdm

from researchpkg.anomaly_detection.config import EXPERIMENTS_DIR
from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_unsloth_financials_extended import (
    EXCLUDED_FINANCIALS_FEATURES,
    EXTENDED_FINANCIAL_FEATURES,
    SYSTEM_INSTRUCTION_AND_TURN_COUNT,
    LLM_FinancialsClassifier,
    train_and_evaluate_financials_model,
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    llm_train_and_evaluate_base_model,
)
from researchpkg.anomaly_detection.models.utils import load_train_test_path

LLM_SYSTEM_PROMPT_WITH_LAST_4 = """
You are a financial analyst specializing in corporate fraud detection. You will be given a structured set of financial indicators derived from a company's quarterly financial statements.

Each financial variable includes:
- its **name**,
- its **numerical value** (in absolute terms or percentages),


Additionally, the company’s **industry sector** will be provided to help you interpret the financial context appropriately.
Eventually, in addition to the current quarter, you will also be provided with the last 4 quarters of financial data.

Your task is to thoroughly examine the entire set of financial variables and determine whether there are signs of fraudulent activity(Financial statement fraud)  during the reported quarter. No single variable is sufficient to conclude fraud; you must consider **the overall pattern of the data**.


Fraud may include, but is not limited to:  
- Financial misstatement  
- Revenue overstatement or understatement  
- Asset misappropriation  
- Manipulation of expenses or liabilities  
Do you think this company is committing fraud? Answer with "YES" or "NO".
"""

USER_PROMPT_WITH_LAST_4 = """
The company operates in the {industry_title} sector.
The current quarter is {current_quarter}.

### Current quarters Financial Features:
{financials_str}

### Last {n_quarters} quarters of Financial Features:
{last_financials_str}
"""


class LLM_FinancialsClassifierWithHistory(LLM_FinancialsClassifier):
    """
    LLM Fraud classifier using financial data with historical data from the last 4 quarters
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

            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_fraud_classifier_dataset_{version}_extended_fin_with_last4history_"
                f"{'_oversample' if oversample else ''}{'_undersample' if undersample else ''}"
            )
        # Prepare last financials data as a dictionary indexed by cik and quarter
        self.last_financials_dict = {}
        # Call parent constructor
        super().__init__(
            config=config,
        )

    def _process_loaded_data(self, df):
        """Process and merge financial data with the dataset, including historical data."""
        # Call parent method to get the basic processed dataframe
        df = super()._process_loaded_data(df)

        df["quarter_num"] = df["quarter"].apply(lambda x: int(x.split("q")[1]))
        df["quarter_num"] = df["quarter_num"].astype(int)

        self._full_financials_df["quarter_num"] = self._full_financials_df[
            "quarter"
        ].apply(lambda x: int(x.split("q")[1]))
        self._full_financials_df["quarter_num"] = self._full_financials_df[
            "quarter_num"
        ].astype(int)

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Getting last financials"
        ):
            cik = row["cik"]
            year = row["year"]
            quarter = row["quarter"]
            quarter_num = row["quarter_num"]

            # Create a unique key for this entry
            key = f"{cik}_{year}_{quarter}"

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
            quarter: Quarter of current financial datarow
            drop_rate: Percentage of features to randomly drop

        Returns:
            Formatted string of historical financial data
        """
        # Create the lookup key
        key = f"{cik}_{year}_{quarter}"
        last_financials = self.last_financials_dict[key]
        n_quarters = len(last_financials)

        if n_quarters == 0:
            return "", 0

        quarters_data = []

        for i, quarter_data in enumerate(last_financials):
            year, quarter = quarter_data.get("year"), quarter_data.get("quarter")
            # Filter the actual financial data
            quarter_data = {
                k: v
                for k, v in quarter_data.items()
                if k in EXTENDED_FINANCIAL_FEATURES
            }

            # Filter out excluded features
            quarter_data = {
                k: v
                for k, v in quarter_data.items()
                if k not in EXCLUDED_FINANCIALS_FEATURES and v != 0 and not np.isnan(v)
            }

            # Apply drop rate if needed
            if drop_rate > 0:
                quarter_data = self._drop_random_keys(quarter_data, drop_rate)

            # Format the quarter data
            quarter_str = self.format_financials(quarter_data)
            year_quarter = f"Year: {year}, Quarter: {quarter}"
            quarters_data.append(f"#### {year_quarter}\n{quarter_str}")

        return "\n\n".join(quarters_data), n_quarters

    def truncate_and_format_prompt(
        self,
        financials_str,
        industry_title,
        row=None,
        current_quarter=None,
        cik=None,
        year=None,
        quarter=None,
    ):
        """
        Ensure the prompt is within the model's context length, including historical data.

        Args:
            financials_str (str): The financial data as a formatted string.
            industry_title (str): The industry title based on SIC code.
            row (pd.Series, optional): Row data - for backward compatibility.
            current_quarter (str): Current quarter information.
            cik (str, optional): Company identifier - used with year and quarter.
            year (str, optional): Year of current financial data.
            quarter (str, optional): Quarter of current financial data.

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        last_financials_str = ""
        n_quarters = 0

        # Extract cik, year, quarter from row if direct values not provided
        if row is not None and (cik is None or year is None or quarter is None):
            cik = row.get("cik")
            year = row.get("year")
            quarter = row.get("quarter")

        last_financials_str, n_quarters = self.format_last_financials(
            cik, year, quarter
        )

        # Tokenize the prompt with financial data
        combined_financials = f"{financials_str}\n\n{last_financials_str}"
        prompt_tokens = self.tokenizer(
            combined_financials,
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

        # Split the truncated text back into current and historical sections
        split_pos = financials_str.strip()
        if split_pos in truncated_financials:
            current_fin_str = split_pos
            last_fin_str = truncated_financials[
                truncated_financials.find(split_pos) + len(split_pos) :
            ].strip()
        else:
            # If we can't find the exact split, just use the truncated text for current financials
            current_fin_str = truncated_financials
            last_fin_str = "Not provided"

        # Format user prompt with financial data
        formatted_user_prompt = USER_PROMPT_WITH_LAST_4.format(
            financials_str=current_fin_str,
            industry_title=industry_title,
            current_quarter=current_quarter or "N/A",
            last_financials_str=last_fin_str,
            n_quarters=n_quarters,
        )

        return LLM_SYSTEM_PROMPT_WITH_LAST_4, formatted_user_prompt

    def generate_prompt(self, row, idx=None, drop_rate=None, **kwargs):
        """Generate and tokenize the prompt for training using financial data with history."""
        financials = self.prepare_financial_data(row)
        if drop_rate is not None and drop_rate > 0:
            logging.info(f"Dropping {drop_rate*100}% of rows")
            financials = self._drop_random_keys(financials, drop_rate)
        financials_str = self.format_financials(financials)
        label = "YES" if row["is_fraud"] else "NO"

        # Get the SIC code and industry title
        sic = str(row["sic"]).zfill(4)
        industry_title = self.sic_index[sic]

        # Get current quarter information
        current_quarter = f"Year: {row['year']}, Quarter: {row['quarter']}"

        # Pass row data directly instead of using index
        system_prompt, user_prompt = self.truncate_and_format_prompt(
            financials_str=financials_str,
            industry_title=industry_title,
            cik=row["cik"],
            year=row["year"],
            quarter=row["quarter"],
            current_quarter=current_quarter,
        )

        if "deepseek" in self.config["model_url"].lower():
            label = f"<think>{label}</think>"

        # Format using chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": label},
        ]

        # Now add the label to the truncated prompt
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        result = {
            "text": full_prompt,
            "prompt": full_prompt,
            "answer": label,
            "sic": sic,
            "sicagg": row["sicagg"],
            "cik": row["cik"],
            "quarter": f"{row['year']}{row['quarter']}",
            "glabels": str(row.get("glabels", "no_fraud")),
        }

        return result


def train_and_evaluate_financials_model_with_history(config):
    """
    Train and evaluate the LLM extended financials classifier with historical data.
    """
    # Ensure the configuration has history enabled
    config["include_last_4_quarters"] = True

    train_path, test_path = load_train_test_path(config)
    # Use the base function with our model class
    return llm_train_and_evaluate_base_model(
        model_class=LLM_FinancialsClassifierWithHistory,
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

    train_and_evaluate_financials_model_with_history(CONFIG)
