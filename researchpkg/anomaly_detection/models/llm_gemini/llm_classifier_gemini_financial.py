"""
Gemini classifier for fraud detection using financial data.
"""

import logging
from pathlib import Path

import pandas as pd

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
)
from researchpkg.anomaly_detection.models.llm_gemini.base_llm_classifier_gemini import (
    COMPLETION_INSTRUCTION,
    GeminiClassifier,
    train_and_evaluate_gemini_model,
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

# Features to exclude
EXCLUDED_FINANCIALS_FEATURES = set(
    [
        BENEISH_PROBM  # Too much biasing the model as it is a probability of earnings manipulation
    ]
)

# Features with currency units
CURRENCY_FEATURES = set(AGGREGATE_FEATURES + DIFF_FEATURES + IMPORTANT_TAGS)
CURRENCY_FEATURES.add(RATIO_NET_WORKING_CAPITAL)

# Features that should be displayed as percentages
PERCENTAGE_FEATURES = set(RATIO_FEATURES) - set([RATIO_NET_WORKING_CAPITAL])

# Maximum tokens to reserve for system instruction and prompt structure
SYSTEM_INSTRUCTION_TOKENS = 210

# Prompt template for financial data
USER_PROMPT = """
You are a financial forensic analyst.  
The company operates in the **{industry_title}** sector. Below are key financial indicators derived from its income statement, balance sheet, and cash flow statement:  

{financials_str}  

Based on these financial metrics and your knowledge of typical red flags in financial reporting, assess whether there is a high likelihood that this company is engaging in fraudulent financial behavior. 

Respond only with "YES" if you detect signs consistent with financial fraud, or "NO" if you believe the financials appear legitimate.
"""

def is_with_currency(feature):
    """Check if a feature should be displayed with currency symbol."""
    return feature in CURRENCY_FEATURES


class GeminiFinancialClassifier(GeminiClassifier):
    """
    Gemini classifier for fraud detection using financial data.
    """
    
    def __init__(self, config):
        """
        Initialize the financial classifier.
        
        Args:
            config (dict): Configuration dictionary
        """
        # Override default experiments directory
        if "experiments_dir" not in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"gemini_fraud_classifier_dataset_{version}_fin"
                f"{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
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
        
        return super()._process_loaded_data(df)
    
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
            drop_rate (float): Probability of dropping each feature
            
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
    
    def generate_prompt(self, row, drop_rate=0.0):
        """
        Generate a prompt for the financial classifier.
        
        Args:
            row (pd.Series): Row from the DataFrame
            drop_rate (float): Rate to drop financial features 
            
        Returns:
            str: Formatted prompt
        """
        # Get financial data
        financials = self.prepare_financial_data(row)
        if drop_rate > 0:
            financials = drop_random_keys(financials, drop_rate)
        
        financials_str = self.format_financials(financials)
        
        # Get the SIC code and industry title
        sic = str(row["sic"]).zfill(4)
        industry_title = self.sic_index.get(sic, "Unknown Industry")
        
        # Format the prompt
        prompt = USER_PROMPT.format(
            financials_str=financials_str,
            industry_title=industry_title
        )
        
        # Add completion instruction
        full_prompt = prompt + COMPLETION_INSTRUCTION
        
        return full_prompt


def train_and_evaluate_financial_gemini_model(config):
    """
    Train and evaluate the Gemini financial classifier.
    
    Args:
        config (dict): Configuration for the model
        
    Returns:
        tuple: (model, metrics, predictions)
    """
    
    train_path, test_path = load_cross_validation_path(config)
    
    return train_and_evaluate_gemini_model(
        model_class=GeminiFinancialClassifier,
        config=config,
        train_path=train_path,
        test_path=test_path
    )


if __name__ == "__main__":
    # Example usage
    CONFIG = {
        "model_name": "gemini-2.0-flash",
        "dataset_version": "company_isolated_splitting",
        "fold_id": 1,
        "epochs": 1,
        "oversample": False
    }
    
    train_and_evaluate_financial_gemini_model(CONFIG)
