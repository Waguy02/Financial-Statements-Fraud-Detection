"""
Global preprocessing script of SEC data
----------------------------------------
- Build an index with the minimal following set of keys:
cik, year, mda_year_id, mda_quarter_id, aaer_no, aaer_labels, num_aaer_labels
"""


from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_NOAUG,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.preprocessing.noaug.noaug_base_preprocessing import (
    NoAugPreprocessing,
)


class V3BalancedTimeSplitPreprocessing(NoAugPreprocessing):
    NON_FRAUD_PCT = 60
    DATASET_NAME = "v6_balanced_cik_timesplit"

    def preprocess_final_dataset_index(self, index_aaer, index_mda, index_fin):
        """
        Override to use CIK-unbiased splitting instead of the default random splitting
        """
        # Use the parent class method to do all the preprocessing steps
        index_final, _, _ = super().preprocess_final_dataset_index(
            index_aaer, index_mda, index_fin
        )

        # Then override the splitting method to use CIK-unbiased splitting
        train_index, test_index = self.split_and_save_dataset_ciks_timesplit(
            index_final
        )

        return index_final, train_index, test_index


if __name__ == "__main__":
    processor = V3BalancedTimeSplitPreprocessing()
    processor.run(
        PREPROCESSED_PATH / "AAER_PREPROCESSED" / "df_aaer_filtered_from_2009.xlsx",
        PREPROCESSED_PATH / "SEC_MDA" / "mda_ref_quarterly.xlsx",
        FINANCIALS_DIR_NOAUG / "sec_financials_quarterly_no_financials.csv",
    )
