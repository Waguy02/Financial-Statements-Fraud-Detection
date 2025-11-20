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
MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"
MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED_PARAGRAPHS" / "quarterly"
EXTRA_FINANCIAL_PATH = PREPROCESSED_PATH / "SEC_MDA_FINANCIAL_JSON" / "quarterly"
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"

EXCLUDED_FINANCIALS_FEATURES = set(
    [
        BENEISH_PROBM  # Too much biasing the model as it a probability of earnings manipulation
    ]
)

CURRENCY_FEATURES = set(AGGREGATE_FEATURES + DIFF_FEATURES + IMPORTANT_TAGS)
CURRENCY_FEATURES.add(RATIO_NET_WORKING_CAPITAL)

SYSTEMS_PROMPT_TOKENS_COUNT = 600  # Increased for the longer system prompt


def is_with_currency(feature):
    return feature in CURRENCY_FEATURES


PERCENTAGE_FEATURES = set(RATIO_FEATURES) - set([RATIO_NET_WORKING_CAPITAL])

LLM_SYSTEM_PROMPT = """You are a financial analyst specializing in corporate fraud detection. You will be given a structured set of financial indicators derived from a company's quarterly financial statements.

Each financial variable includes:
- its **name**,
- its **numerical value** (in absolute terms or percentages),

Additionally, the company's **industry sector** will be provided to help you interpret the financial context appropriately.

You will receive three types of information:
1. Core financial variables from the company's quarterly financial statements
2. Additional financial information extracted from the company's official report
3. Descriptive paragraphs from the Management Discussion & Analysis (MDA) section describing the company's situation, activities, revenue, debt profile, risks, and opportunities

Your task is to thoroughly examine all available information and determine whether there are signs of fraudulent activity (Financial statement fraud) during the reported quarter. No single variable is sufficient to conclude fraud; you must consider **the overall pattern of the data**.

If any discrepancies arise between different data sources, prioritize the additional financial information over the original variables.

Please follow these strict instructions:

1. Carefully analyze the full financial profile in context.
2. Reason logically and concisely based on the data presented.
3. Then respond using **only one** of the following two labels:
   - "Fraud"** → if you have sufficient indicators suggesting the company likely committed fraud during the quarter.
   - "Not Fraud" → if you find no clear evidence or pattern suggesting fraud.

4. Remember: you are acting as a professional fraud analyst. Be precise, rigorous, and objective in your evaluation. Respond only with the label, without any additional explanations or justifications. Do not provide any other information or context.
"""

USER_PROMPT = """
The company operates in the {industry_title} sector and the current quarter is {quarter}.
 
### Core Financial Variables:
{financials_str}

### Additional Financial Information:
{additional_financials_str}

### Descriptive Paragraphs from the MDA:
{mda_text}
"""


class LLM_FinancialAndExtraAndMDAParagraphClassifier(BaseLLMFraudClassifier):
    """
    LLM Fraud classifier using extended financial data, extra financial information, and MDA text
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
                / f"llm_fraud_classifier_dataset_{version}_extended_fin_and_extra_and_mda_paragraphs"
                f"{'_summarized' if is_summarized_mda else ''}"
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

    def load_mda_content(self, mda_quarter_id):
        """
        Load the content of a MDA File.
        """
        mda_path = (
            MDA_PATH if self.config.get("raw_mda", False) else MDA_PATH_SUMMARIZED
        )
        mda_file = mda_path / f"{mda_quarter_id}.txt"

        if not mda_file.exists():
            raise FileNotFoundError(f"MDA file {mda_file} does not exist.")

        with open(mda_file, "r", encoding="utf-8") as file:
            mda_content = file.read()
            return mda_content

    def load_extra_financial_content(self, mda_quarter_id):
        """
        Load the extra financial content from JSON file.
        """
        extra_file = EXTRA_FINANCIAL_PATH / f"{mda_quarter_id}.json"

        if not extra_file.exists():
            raise FileNotFoundError(
                f"Extra financial file {extra_file} does not exist."
            )

        with open(extra_file, "r", encoding="utf-8") as file:
            extra_financial_content = json.load(file)
            # Format as string
            extra_financial_content_str = ""
            for key, value in extra_financial_content.items():
                extra_financial_content_str += f"- {key} : {value}\n"

            return extra_financial_content_str

    def prepare_financial_data(self, row):
        """
        Extract and prepare financial data from a DataFrame row.
        """
        financial_data = {}
        for feature in filter(
            lambda x: x not in EXCLUDED_FINANCIALS_FEATURES, EXTENDED_FINANCIAL_FEATURES
        ):
            financial_data[feature] = row[feature]
        return financial_data

    def truncate_and_format_prompt(
        self, mda_text, extra_financial_text, financials_str, industry_title, quarter
    ):
        """
        Ensure the prompt is within the model's context length.
        """

        financials_max_tokens = 1200

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

        # Calculate remaining available tokens after system prompt and financials
        remaining_tokens = int(
            self.max_length
            - len(financials_tokens["input_ids"][0])
            - SYSTEMS_PROMPT_TOKENS_COUNT
        )

        # Split remaining tokens between extra financials and MDA text
        extra_fin_max_tokens = 2700
        # Tokenize and truncate extra financial text
        extra_fin_tokens = self.tokenizer(
            extra_financial_text,
            return_tensors="pt",
            truncation=True,
            max_length=extra_fin_max_tokens,
            padding=False,
        )

        remaining_tokens -= len(extra_fin_tokens["input_ids"][0])

        truncated_extra_fin = self.tokenizer.decode(
            extra_fin_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Tokenize and truncate MDA text
        mda_tokens = self.tokenizer(
            mda_text,
            return_tensors="pt",
            truncation=True,
            max_length=remaining_tokens - 350,  # 350 tokens for the system prompt
            padding=False,
        )

        truncated_mda = self.tokenizer.decode(
            mda_tokens["input_ids"][0], skip_special_tokens=True
        )

        # Format user prompt with all data
        formatted_user_prompt = USER_PROMPT.format(
            financials_str=truncated_financials,
            additional_financials_str=truncated_extra_fin,
            mda_text=truncated_mda,
            industry_title=industry_title,
            quarter=quarter,
        )

        return LLM_SYSTEM_PROMPT, formatted_user_prompt

    def generate_prompt(self, row, idx=None, **kwargs):
        """
        Generate and tokenize the prompt for training using financial data, extra financial information, and MDA text.
        """
        try:
            # Load MDA content and extra financial content
            mda_content = self.load_mda_content(row["mda_quarter_id"])
            extra_financial_content = self.load_extra_financial_content(
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

            # Create the full prompt with all data sources
            system_prompt, user_prompt = self.truncate_and_format_prompt(
                mda_content,
                extra_financial_content,
                financials_str,
                industry_title,
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
        except FileNotFoundError as e:
            logging.warning(f"File not found: {e}, skipping")
            return None


def train_and_evaluate_financial_and_extra_and_mda_paragraph_model(config=None):
    """
    Train and evaluate the LLM classifier using financial data, extra financial information, and MDA text.
    """
    train_path, test_path = load_cross_validation_path(config)

    return llm_train_and_evaluate_base_model(
        model_class=LLM_FinancialAndExtraAndMDAParagraphClassifier,
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

    train_and_evaluate_financial_and_extra_and_mda_paragraph_model(CONFIG)
