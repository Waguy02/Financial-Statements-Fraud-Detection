"""
Global preprocessing script of SEC data
----------------------------------------
- Build an index with the minimal following set of keys:
cik, year, mda_year_id, mda_quarter_id, aaer_no, aaer_labels, num_aaer_labels
"""


from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_EXTENDED,
    MDA_DATASET_PATH,
    NUMBER_OF_FOLDS,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.preprocessing.extended.base_preprocessing import (
    ExtendedPreprocessing,
)


class V5UnbalancedRandomSplit(ExtendedPreprocessing):
    NON_FRAUD_PCT = 95
    DATASET_VERSION = "time_splitting"

    def split_datasets(self, index_final, test_size=0.1):
        # Then override the splitting method to use CIK-unbiased splitting
        return self.split_and_save_datasets_random(
            index_final=index_final, test_size=test_size
        )

    def split_dataset_with_fold(self, index_final, k=NUMBER_OF_FOLDS):
        """
        Split the dataset into k folds for cross-validation.
        """
        return self.split_and_save_dataset_kfolds_random(index_final=index_final, k=k)


if __name__ == "__main__":
    processor = V5UnbalancedRandomSplit()
    processor.run_k_folds(
        PREPROCESSED_PATH
        / "AAER_PREPROCESSED"
        / "df_aaer_with_misstatements_since_2009.csv",
        MDA_DATASET_PATH / "mda_ref_quarterly.csv",
        FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly_no_financials.csv",
        NUMBER_OF_FOLDS,
    )
