import logging
from pathlib import Path

import pandas as pd

from researchpkg.anomaly_detection.config import EXPERIMENTS_DIR, PREPROCESSED_PATH
from researchpkg.anomaly_detection.models.llm_multilabel.llm_classifier_multilabel import (
    LLMClassifierMultiLabel,
)
from researchpkg.anomaly_detection.models.utils import (
    load_cross_validation_path,
    load_sic_industry_title_index,
)

# MDA paths
MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"

MDA_PATH = PREPROCESSED_PATH / "SEC_MDA" / "quarterly"

USER_PROMPT = """
The company operates in the **{industry_title}** sector.
Here is a structured summary of the Management Discussion and Analysis (MDA) section of the quarterly report:
{mda_content}.
As a financial forensic analyst, assess whether there is a high likelihood that this company is engaging in financial fraud.
Fraud refers to any intentional misrepresentation or omission of information that could deceive stakeholders, such as investors, creditors, or regulators.
Find patterns, anomalies or inconsistencies in the MDA data that may indicate earnings mistatements.
"""


class LLMClassifierMultiLabelMDA(LLMClassifierMultiLabel):
    def __init__(self, config):
        if not "experiments_dir" in config:
            version = config.get("dataset_version", "v3")
            oversample = config.get("oversample", False)
            undersample = config.get("undersample", False)
            assert not (oversample and undersample), "Cannot use both oversampling and undersampling at the same time."
            use_focal_loss = config.get("use_focal_loss", True)
            mode = "binary" if config.get("binary", False) else "multilabel"
            config["experiments_dir"] = (
                EXPERIMENTS_DIR / f"llm_{mode}_fraud_classifier_dataset_{version}_mda"
                f"{'_oversample' if oversample else ''}"
                f"{'_undersample' if undersample else ''}"
                f"{'_focal_loss' if use_focal_loss else ''}"
            )
        super().__init__(config)
        self.sic_index = load_sic_industry_title_index()

    def _process_data(self, df):
        df = super()._process_data(df)
        # Minimal text formatting with MDA content
        df["text"] = df.apply(lambda row: self._build_text_prompt(row), axis=1)
        return df

    def _load_mda_content(self, mda_quarter_id):
        mda_root_dir = (
            MDA_PATH_SUMMARIZED)
        mda_path = mda_root_dir if self.config.get("summarized_mda", True) else MDA_PATH
        mda_file = mda_path / f"{mda_quarter_id}.txt"
        if not mda_file.exists():
            logging.warning(f"MDA file {mda_file} does not exist.")
            return "No MDA content available."
        with open(mda_file, "r", encoding="utf-8") as file:
            return file.read()

    def _build_text_prompt(self, row):
        sector = self.sic_index.get(str(row.get("sic", "")).zfill(4), "Unknown Sector")
        mda_content = self._load_mda_content(row["mda_quarter_id"])
        return USER_PROMPT.format(
            industry_title=sector,
            mda_content=mda_content,
        )


def train_and_evaluate_mda_multilabel_model(config):
    classifier = LLMClassifierMultiLabelMDA(config)
    train_path, test_path = load_cross_validation_path(config)
    return classifier.train_and_evaluate(train_path, test_path)


if __name__ == "__main__":
    CONFIG = {
        "model_url": "unsloth/Llama-3.1-8B-unsloth-bnb-4bit",
        "model_alias": "Llama-3.1-8B-unsloth-bnb-4bit",
        "max_context": 1500,
        "lora_r": 8,
        "lora_alpha": 32,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 14,
        "learning_rate": 5e-5,
        "save_steps": 5,
        "summarized_mda": True,
    }
    train_and_evaluate_mda_multilabel_model(CONFIG)
