"""
Gemini classifier for fraud detection using both financial data and MDA text.
"""

import logging
from pathlib import Path

import pandas as pd

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    FINANCIALS_DIR_EXTENDED,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.models.llm_gemini.base_llm_classifier_gemini import (
    COMPLETION_INSTRUCTION,
    GeminiClassifier,
    train_and_evaluate_gemini_model,
)
from researchpkg.anomaly_detection.models.llm_gemini.llm_classifier_gemini_financial import (
    CURRENCY_FEATURES,
    EXCLUDED_FINANCIALS_FEATURES,
    PERCENTAGE_FEATURES,
    is_with_currency,
)
from researchpkg.anomaly_detection.models.utils import (
    drop_random_keys,
    load_cik_company_mapping,
    load_cross_validation_path,
    load_sic_industry_title_index,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
    EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT,
)

# Constants
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"
MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"
MDA_PATH_SUMMARIZED_PARAGRAPHS = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED_PARAGRAPHS" / "quarterly"

# Maximum tokens for system instruction and prompt structure
SYSTEM_INSTRUCTION_TOKENS = 210

# Prompt template for combined financial and MDA data
USER_PROMPT = """
The company operates in the {industry_title} sector.

Here are financial variables derived from the income statement, balance sheet, and cash flow statement of the company:
{financials_str}

Also below is the summary of the Management Discussion and Analysis (MDA) section of the quarterly report:
{mda_content}

Based on these informations and your knowledge of typical red flags in financial reporting, \
assess whether there is a high likelihood that this company is engaging in fraudulent financial behavior. 

Fraud may include, but is not limited to:  
- Financial misstatement  
- Revenue overstatement or understatement  
- Asset misappropriation  
- Manipulation of expenses or liabilities  
Do you think this company is committing fraud? Answer with "YES" or "NO".
"""


class GeminiFinancialAndMDAClassifier(GeminiClassifier):
    """
    Gemini classifier using both financial data and MDA text for fraud detection.
    """
    
    def __init__(self, config):
        """
        Initialize the financial and MDA classifier.
        
        Args:
            config (dict): Configuration dictionary
        """
        # Set up experiment directory if not provided
        if "experiments_dir" not in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            is_summarized_mda = config.get("summarized_mda", True)
            use_full_summary = config.get("use_full_summary", False)
            
            config["experiments_dir"] = (
                EXPERIMENTS_DIR
                / f"gemini_fraud_classifier_dataset_{version}_fin_mda"
                f"{'_summarized' if is_summarized_mda else ''}"
                f"{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
                f"{'_full_summary' if use_full_summary else '_paragraphs'}"
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
    
    def load_mda_content(self, mda_quarter_id):
        """
        Load the content of an MDA file.
        
        Args:
            mda_quarter_id (str): Quarter ID for the MDA file
            
        Returns:
            str: MDA content
        """
        mda_root_dir = MDA_PATH_SUMMARIZED if self.config.get("use_full_summary", False) else MDA_PATH_SUMMARIZED_PARAGRAPHS
        mda_path = (
            mda_root_dir if self.config.get("summarized_mda", True) else MDA_PATH
        )
        mda_file = mda_path / f"{mda_quarter_id}.txt"
        
        if not mda_file.exists():
            logging.warning(f"MDA file {mda_file} does not exist.")
            return "No MDA content available."
        
        with open(mda_file, "r", encoding="utf-8") as file:
            mda_content = file.read()
            return mda_content
    
    def generate_prompt(self, row, drop_rate=0.0):
        """
        Generate a prompt combining financial data and MDA text.
        
        Args:
            row (pd.Series): Row from DataFrame
            drop_rate (float): Rate to drop financial features 
            
        Returns:
            str: Formatted prompt
        """
        try:
            # Get financial data
            financials = self.prepare_financial_data(row)
            financials_str = self.format_financials(financials, drop_rate=drop_rate)
            
            # Get MDA content
            mda_content = self.load_mda_content(row["mda_quarter_id"])
            
            # Get industry information
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index.get(sic, "Unknown Industry")
            
            # Format the prompt
            prompt = USER_PROMPT.format(
                financials_str=financials_str,
                mda_content=mda_content,
                industry_title=industry_title
            )
            
            # Add completion instruction
            full_prompt = prompt + COMPLETION_INSTRUCTION
            
            return full_prompt
            
        except KeyError as e:
            logging.warning(f"Missing key in row: {e}")
            # Return a default prompt that will produce a reasonable result
            return USER_PROMPT.format(
                financials_str="[No financial data available]",
                mda_content="[No MDA content available]",
                industry_title="Unknown Industry"
            ) + COMPLETION_INSTRUCTION
        except Exception as e:
            logging.error(f"Error generating prompt: {str(e)}")
            return USER_PROMPT.format(
                financials_str="[Error loading financial data]",
                mda_content="[Error loading MDA content]",
                industry_title="Unknown Industry"
            ) + COMPLETION_INSTRUCTION


def train_and_evaluate_financial_and_mda_gemini_model(config):
    """
    Train and evaluate the Gemini financial and MDA classifier.
    
    Args:
        config (dict): Configuration for the model
        
    Returns:
        tuple: (model, metrics, predictions)
    """
    train_path, test_path = load_cross_validation_path(config)
    
    return train_and_evaluate_gemini_model(
        model_class=GeminiFinancialAndMDAClassifier,
        config=config,
        train_path=train_path,
        test_path=test_path
    )


if __name__ == "__main__":
    # Example usage
    CONFIG = {
        "model_name": "gemini-2.0-flash",
        "dataset_version": "company_isolated_splitting",
        "fold_id":1,
        "epochs": 1,
        "summarized_mda": True,
        "use_full_summary": True,
    }
    
    train_and_evaluate_financial_and_mda_gemini_model(CONFIG)
