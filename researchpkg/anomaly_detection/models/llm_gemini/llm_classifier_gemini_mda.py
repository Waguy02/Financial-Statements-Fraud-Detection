"""
Gemini classifier for fraud detection using Management Discussion and Analysis (MDA) data.
"""

import logging
from pathlib import Path

import pandas as pd

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.models.llm_gemini.base_llm_classifier_gemini import (
    COMPLETION_INSTRUCTION,
    GeminiClassifier,
    train_and_evaluate_gemini_model,
)
from researchpkg.anomaly_detection.models.utils import (
    load_cik_company_mapping,
    load_cross_validation_path,
    load_sic_industry_title_index,
)

# Constants for MDA paths
MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"
MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"
MDA_PATH_SUMMARIZED_PARAGRAPHS = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED_PARAGRAPHS" / "quarterly"

# Maximum tokens for system instruction and prompt structure
SYSTEM_INSTRUCTION_TOKENS = 250

# Prompt template for MDA data
USER_PROMPT = """
The company operates in the {industry_title} sector.

Below is the summary of the Management Discussion and Analysis (MDA) section of the quarterly report:
{mda_content}

Based on th MDA content and your knowledge of typical red flags in financial reporting, assess whether there is a high likelihood that this company is engaging in fraudulent financial behavior.
Fraud may include, but is not limited to:  
- Financial misstatement  
- Revenue overstatement or understatement  
- Asset misappropriation  
- Manipulation of expenses or liabilities  
- Irregular cash flow activities  

Do you think this company is committing fraud? Answer with "YES" or "NO".
"""


class GeminiMDAClassifier(GeminiClassifier):
    """
    Gemini classifier for fraud detection using MDA data.
    """
    
    def __init__(self, config):
        """
        Initialize the MDA classifier.
        
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
                / f"gemini_fraud_classifier_dataset_{version}_mda"
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
        Generate a prompt for the MDA classifier.
        
        Args:
            row (pd.Series): Row from the DataFrame
            drop_rate (float): Not used for MDA classifier
            
        Returns:
            str: Formatted prompt
        """
        try:
            # Get MDA content
            mda_content = self.load_mda_content(row["mda_quarter_id"])
            
            # Get industry information
            sic = str(row["sic"]).zfill(4)
            industry_title = self.sic_index.get(sic, "Unknown Industry")
            
            # Format the prompt
            prompt = USER_PROMPT.format(
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
                mda_content="[No MDA content available]",
                industry_title="Unknown Industry"
            ) + COMPLETION_INSTRUCTION
        except Exception as e:
            logging.error(f"Error generating prompt: {str(e)}")
            return USER_PROMPT.format(
                mda_content="[Error loading MDA content]",
                industry_title="Unknown Industry"
            ) + COMPLETION_INSTRUCTION


def train_and_evaluate_mda_gemini_model(config):
    """
    Train and evaluate the Gemini MDA classifier.
    
    Args:
        config (dict): Configuration for the model
        
    Returns:
        tuple: (model, metrics, predictions)
    """

    
    train_path, test_path = load_cross_validation_path(config)
    
    return train_and_evaluate_gemini_model(
        model_class=GeminiMDAClassifier,
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
        "summarized_mda": True,
        "use_full_summary": True,
    }
    
    train_and_evaluate_mda_gemini_model(CONFIG)
