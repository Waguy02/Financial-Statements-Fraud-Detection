"""
Global preprocessing script of SEC data
----------------------------------------
- Build an index with the minimal following set of keys:
cik, year, mda_year_id, mda_quarter_id, aaer_no, aaer_labels, num_aaer_labels
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_EXTENDED,
    MDA_DATASET_PATH,
    PREPROCESSED_PATH,
    PREPROCESSED_PATH_CRAJA_IMPUTED,
)
from researchpkg.anomaly_detection.preprocessing.extended.base_preprocessing import (
    ExtendedPreprocessing,
)


class V6TimeSplitPreprocessing(ExtendedPreprocessing):
    NON_FRAUD_PCT = 95
    DATASET_VERSION = "time_splitting"

    def split_dataset_with_fold(self, index_final, k=1):
        """
        Split the dataset into k folds for cross-validation.
        """
        if k != 1:
            logging.warning(
                f"Only k=1 is supported for split_with k_fold. {k} provided. "
                f"1 will be used anyway."
            )
        return self.split_and_save_dataset_kfolds_by_time(
            index_final=index_final, k=1, test_size=0.3
        )

    def split_datasets(self, index_final, test_size=None):
        return self.split_and_save_dataset_by_time(index_final, test_size)


if __name__ == "__main__":
    processor = V6TimeSplitPreprocessing()
    processor.run_k_folds(
        PREPROCESSED_PATH / "AAER_PREPROCESSED" / "df_aaer_filtered_from_2009.xlsx",
        MDA_DATASET_PATH / "mda_ref_quarterly.xlsx",
        FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly_no_financials.csv",
    )
