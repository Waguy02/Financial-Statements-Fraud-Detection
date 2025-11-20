import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
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
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    AGGREGATE_FEATURES,
    BENEISH_PROBM,
    DIFF_FEATURES,
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
    EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT,
    IMPORTANT_TAGS,
    RATIO_FEATURES,
    RATIO_NET_WORKING_CAPITAL,
)

# Constants
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
EXTRA_FINANCIAL_PATH = PREPROCESSED_PATH / "SEC_MDA_FINANCIAL_JSON" / "quarterly"

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

Here are financial variables derived from the income statement, balance sheet, and cash flow statement of the company:
{financials_str}

Additionally, here are specific financial metrics extracted from the company's quarterly report:
{extra_financial_str}

Based on these financial indicators and your knowledge of typical red flags in financial reporting, \
assess whether there is a high likelihood that this company is engaging in fraudulent financial behavior. 

Fraud may include, but is not limited to:  
- Financial misstatement  
- Revenue overstatement or understatement  
- Asset misappropriation  
- Manipulation of expenses or liabilities  
Do you think this company is committing fraud? Answer with "YES" or "NO".
"""

def is_with_currency(feature):
    return feature in CURRENCY_FEATURES


class LLMFinancialAndExtraSoftmaxClassifier(LLMClassifierSoftmax):
    """
    LLM classifier using softmax with both financial data and extra financial information for fraud detection.
    """

    def __init__(self, config):
        """
        Initialize the financial and extra financial softmax classifier.
        
        Args:
            config (dict): Configuration dictionary
        """
        # Set up experiment directory if not provided
        if "experiments_dir" not in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            use_focal_loss = config.get("use_focal_loss", False)
            use_weighted_loss = config.get("loss_weight", False)
            only_completion = config.get("only_completion", True)
            
            assert not (
                oversample and undersample
            ), "You cannot set both undersample and oversample args to True."

            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"llm_softmax_fraud_classifier_dataset_{version}_fin_extra"
                f"{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
                f"{'_focal_loss' if use_focal_loss else ''}"
                f"{'_weighted_loss' if use_weighted_loss else ''}"
                f"{'_not_only_completion' if not only_completion else ''}"
            )
            
        # Call parent constructor
        super().__init__(config)
        
        # Load SIC index and company mapping
        self.sic_index = load_sic_industry_title_index()
        self.cik_company_name_mapping = load_cik_company_mapping(
            dataset_version=self.config.get("dataset_version", "v3")
        )
    
    def _process_loaded_data(self, df):
        """
        Process loaded data by merging with financial features.
        
        Args:
            df (pd.DataFrame): Original dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe with financial features
        """
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
    
    def prepare_financial_data(self, row):
        """
        Extract and prepare financial data from a DataFrame row.
        
        Args:
            row (pd.Series): A row from the DataFrame containing financial data
            
        Returns:
            dict: Financial data dictionary
        """
        financial_data = {}
        for feature in filter(
            lambda x: x not in EXCLUDED_FINANCIALS_FEATURES, EXTENDED_FINANCIAL_FEATURES
        ):
            financial_data[feature] = row[feature]
        return financial_data
    
    def format_financials(self, financials, drop_rate=0):
        """
        Format financial data as a string for the model.
        
        Args:
            financials (dict): Dictionary of financial features
            drop_rate (float): Probability of dropping features
            
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
                f"- {EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT.get(key, key)}: {'$' if is_with_currency(key) else ''}{display_financial_value(value)}{'%' if key in PERCENTAGE_FEATURES else ''}"
                for key, value in financials.items()
            ]
        )
        
        return financials_str
    
    def load_extra_financial_content(self, mda_quarter_id):
        """
        Load the extra financial content from JSON file.
        
        Args:
            mda_quarter_id (str): Quarter ID for the JSON file
            
        Returns:
            str: Formatted extra financial content
        """
        extra_file = EXTRA_FINANCIAL_PATH / f"{mda_quarter_id}.json"
        
        if not extra_file.exists():
            logging.warning(f"Extra financial file {extra_file} does not exist.")
            return "No additional financial information available."
        
        try:
            with open(extra_file, "r", encoding="utf-8") as file:
                extra_financial_content = json.load(file)
                
                # Format as a string
                extra_financial_str = ""
                for key, value in extra_financial_content.items():
                    extra_financial_str += f"- {key}: {value}\n"
                
                return extra_financial_str
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON from {extra_file}")
            return "Error reading additional financial information."
    
    def truncate_and_format_prompt(self, financials_str, extra_financial_str, industry_title):
        """
        Ensure the prompt is within the model's context length.

        Args:
            financials_str (str): The financial data as a formatted string.
            extra_financial_str (str): The extra financial content.
            industry_title (str): The industry title based on SIC code.

        Returns:
            str: Formatted user prompt
        """
        # Create full prompt
        full_prompt = USER_PROMPT.format(
            financials_str=financials_str, 
            extra_financial_str=extra_financial_str,
            industry_title=industry_title
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
        Generate a prompt combining financial data and extra financial information.
        
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
            financials_str = self.format_financials(financials, drop_rate=drop_rate if drop_rate else 0)
            
            # Get extra financial content
            extra_financial_str = self.load_extra_financial_content(row["mda_quarter_id"])
            
            # Get industry information
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index.get(sic, "Unknown Industry")
            
            # Convert the label
            label = "Fraud" if row["is_fraud"] else "Not Fraud"
            label = label_to_token(label)
            
            # Create the full prompt
            user_prompt = self.truncate_and_format_prompt(
                financials_str, extra_financial_str, industry_title
            )
            full_prompt = user_prompt + COMPLETION_INSTRUCTION+ label
            
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
            
        except KeyError as e:
            logging.warning(f"Missing key in row: {e}")
            return None
        except FileNotFoundError:
            logging.warning(f"Extra financial file for quarter ID {row.get('mda_quarter_id', 'unknown')} not found")
            return None


def train_and_evaluate_financial_and_extra_softmax_model(config=None):
    """
    Train and evaluate the financial and extra softmax classifier.
    
    Args:
        config (dict): Configuration for the model
        
    Returns:
        tuple: (model, metrics, test_results)
    """
    if config is None:
        raise ValueError("Configuration must be provided")
    
    train_path, test_path = load_cross_validation_path(config)
    
    return llm_softmax_train_and_evaluate_base_model(
        model_class=LLMFinancialAndExtraSoftmaxClassifier,
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
    }
    
    train_and_evaluate_financial_and_extra_softmax_model(CONFIG)
