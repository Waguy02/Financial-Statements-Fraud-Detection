import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyaml
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    GLABEL_MAPPING,
    LIST_MISTATEMENT_TYPE_RENAMED,
    MAX_YEAR,
    PREPROCESSED_PATH_EXTENDED,
    SEED_PREPROCESSING,
    SICAGG_INDEX_FILE,
)

# Import the split_dataset functions from utils
from researchpkg.anomaly_detection.models.utils import (
    split_dataset_by_cik,
    split_dataset_by_time,
    split_dataset_randomly,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
)
from researchpkg.utils import configure_logger, seed_everything


class ExtendedPreprocessing:
    NON_FRAUD_PCT = None
    DATASET_VERSION = None
    DEFAULT_TEST_SIZE = 0.1

    def __init__(self):
        self.label_to_glabel_dict = {}
        for glabel in GLABEL_MAPPING:
            for l in GLABEL_MAPPING[glabel]:
                self.label_to_glabel_dict[l] = glabel

    def merge_data(self, index_aaer, index_mda, index_fin):
        """
        Merge AAER, MDA, and financial data into a single dataset.

        Args:
            index_aaer (pd.DataFrame): DataFrame containing AAER (fraud) data
            index_mda (pd.DataFrame): DataFrame containing MDA data
            index_fin (pd.DataFrame): DataFrame containing financial data

        Returns:
            tuple: (merged_dataframe, cik_to_company_dict)
        """
        # Standardize company names
        index_aaer["company"] = index_aaer.company.str.lower()
        index_mda["company"] = index_mda.company.str.lower()
        index_fin["company"] = index_fin.company.str.lower()

        # Create a dictionary mapping CIKs to company names
        cik_to_company_dict = dict(zip(index_fin["cik"], index_fin["company"]))
        cik_to_company_dict.update(dict(zip(index_mda["cik"], index_mda["company"])))
        cik_to_company_dict.update(dict(zip(index_aaer["cik"], index_aaer["company"])))

        # Remove company columns after creating the dictionary
        index_aaer.drop(columns=["company"], inplace=True)
        index_mda.drop(columns=["company"], inplace=True)
        index_fin.drop(columns=["company"], inplace=True)

        # Merge dataframes
        logging.info("Merging indexes")
        index_merged = pd.merge(
            index_aaer, index_mda, on=["cik", "year", "quarter"], how="outer"
        )
        index_merged = pd.merge(index_merged, index_fin, on=["cik", "year", "quarter"])
        index_merged.drop_duplicates(inplace=True)
        index_merged["is_fraud"] = index_merged.aaer_id.notna()

        # Filter for rows with MDA data
        index_merged = index_merged[index_merged["mda_quarter_id"].notna()]

        # Enrich with serial fraud information
        index_merged = self._enrich_with_serial_fraud_info(index_merged)

        # Count reports per CIK
        index_merged["cik_reports_count"] = index_merged.groupby("cik")[
            "quarter"
        ].transform("count")

        return index_merged, cik_to_company_dict

    def _enrich_with_serial_fraud_info(self, index_merged):
        """
        Enrich the merged dataset with serial fraud information.

        Args:
            index_merged (pd.DataFrame): Merged dataset with fraud indicators

        Returns:
            pd.DataFrame: Dataset enriched with serial fraud information
        """
        # Identify and process serial frauds
        logging.info("Enriching with serial fraud features")
        index_merged_with_aaer = index_merged[index_merged.is_fraud == True][
            ["cik", "year", "quarter"]
        ]
        index_merged_with_aaer["serial_count"] = 1
        index_merged_with_aaer["total_serial_count"] = 1
        index_merged_with_aaer = index_merged_with_aaer.sort_values(
            by=["cik", "year", "quarter"]
        ).drop_duplicates()

        current_count = 1
        current_series_start = 0
        current_series_end = 0

        def isConsecutive(d1, d2):
            y1, q1 = d1
            y2, q2 = d2
            if y1 == y2:
                if q1 == q2 - 1:
                    return True
            elif y1 == y2 - 1:
                if q1 == 4 and q2 == 1:
                    return True
            return False

        for i in tqdm(
            range(1, index_merged_with_aaer.shape[0]), "Counting Serial frauds"
        ):
            is_end_of_series = False
            if (
                index_merged_with_aaer.iloc[i]["cik"]
                == index_merged_with_aaer.iloc[i - 1]["cik"]
            ):
                y1 = int(index_merged_with_aaer.iloc[i - 1]["year"])
                q1 = int(index_merged_with_aaer.iloc[i - 1]["quarter"][-1])
                y2 = int(index_merged_with_aaer.iloc[i]["year"])
                q2 = int(index_merged_with_aaer.iloc[i]["quarter"][-1])

                if isConsecutive((y1, q1), (y2, q2)):
                    current_count += 1
                    current_series_end = i
                    index_merged_with_aaer.iloc[
                        i, index_merged_with_aaer.columns.get_loc("serial_count")
                    ] = current_count
                else:
                    is_end_of_series = True
            else:
                is_end_of_series = True

            if is_end_of_series:
                index_merged_with_aaer.iloc[
                    current_series_start : current_series_end + 1,
                    index_merged_with_aaer.columns.get_loc("total_serial_count"),
                ] = current_count

                current_count = 1
                current_series_start = i
                current_series_end = i

        # Merge the serial fraud information back
        index_merged = index_merged.merge(
            index_merged_with_aaer, on=["cik", "year", "quarter"], how="left"
        )

        return index_merged

    def downsample_non_frauds(self, index_merged, non_fraud_pct):
        """
        Downsample non-fraudulent cases to create a more balanced dataset.

        Args:
            index_merged (pd.DataFrame): Merged dataset containing both fraud and non-fraud cases
            non_fraud_pct (float): Target percentage of non-fraud samples

        Returns:
            pd.DataFrame: Balanced dataset with downsampled non-fraud cases
            int: Number of fraud samples
            int: Number of non-fraud samples
        """
        # Split fraud and non-fraud data
        index_merged_with_fraud = index_merged.query("is_fraud==True")
        index_merged_no_fraud = index_merged.query("is_fraud==False")

        # Calculate target size for non-fraud samples
        size_frauds = len(index_merged_with_fraud)
        target_size_non_frauds = int(
            (non_fraud_pct / (100 - non_fraud_pct)) * size_frauds
        )
        logging.info(f"Downsampling non frauds to : {target_size_non_frauds}")
        # Sample non-fraud data
        sampled_non_fraud = self.sample_non_fraud(
            index_merged_with_fraud, index_merged_no_fraud, target_size_non_frauds
        )

        # Combine fraud and sampled non-fraud data
        index_final = pd.concat([index_merged_with_fraud, sampled_non_fraud])
        logging.info(f"Size of the final index: {len(index_final)}")

        # Log year distribution statistics
        year_distribution = index_final.groupby("year")["is_fraud"].agg(
            ["mean", "count"]
        )
        logging.info("Fraud ratio by year in the final dataset:")
        for year, row in year_distribution.iterrows():
            logging.info(
                f"Year {year}: {row['mean']:.2%} fraud ratio, {row['count']} samples"
            )

        # Count fraud and non-fraud samples
        n_fraud_samples = len(index_final[index_final.is_fraud == True])
        n_non_fraud_samples = len(index_final[index_final.is_fraud == False])

        pct_fraud = n_fraud_samples / (n_fraud_samples + n_non_fraud_samples)
        logging.info(
            f"Overall percentage fraud samples: {pct_fraud:.2%}, {n_fraud_samples} / {n_fraud_samples+n_non_fraud_samples}"
        )

        return index_final, n_fraud_samples, n_non_fraud_samples

    def sample_non_fraud(
        self, index_merged_with_fraud, index_merged_no_fraud, target_size_non_frauds
    ):
        """
        Sample non-fraud cases while preserving the distribution of subgroups defined by 'year' and 'sicagg'.

        Args:
            index_merged_with_fraud: DataFrame containing only fraud cases
            index_merged_no_fraud: DataFrame containing only non-fraud cases
            target_size_non_frauds: Target number of non-fraud samples to select

        Returns:
            DataFrame of sampled non-fraud cases
        """
        logging.info(
            f"Sampling non-fraud cases, preserving year and sicagg distribution"
        )

        sampled_non_fraud_list = []
        total_sampled = 0

        # Calculate the required number of samples per group
        num_fraud_groups = index_merged_with_fraud.groupby(["year", "sicagg"]).size()
        num_non_fraud_groups = index_merged_no_fraud.groupby(["year", "sicagg"]).size()

        downsampling_factor = target_size_non_frauds / len(index_merged_no_fraud)
        # Iterate through each group
        for (year, sicagg), group_size in num_non_fraud_groups.items():
            # Calculate the target size for this group
            group_target_size = int(group_size * downsampling_factor)
            group_target_size = min(
                group_target_size, group_size
            )  # Ensure not sampling more than available

            # Sample from the group
            group_data = index_merged_no_fraud[
                (index_merged_no_fraud["year"] == year)
                & (index_merged_no_fraud["sicagg"] == sicagg)
            ]
            if not group_data.empty and group_target_size > 0:
                sampled_group = group_data.sample(
                    n=min(group_target_size, len(group_data))
                )
                sampled_non_fraud_list.append(sampled_group)
                total_sampled += len(sampled_group)

        # Concatenate the sampled groups
        sampled_non_fraud = pd.concat(sampled_non_fraud_list)

        logging.info(f"Sampled {total_sampled} non-fraud cases")
        return sampled_non_fraud

    def load_all_index_data(self, AAER_INDEX_PATH, MDA_INDEX_PATH, FIN_INDEX_PATH):
        """
        Load AAER, MDA, and financial data from disk and prepare them for processing.

        Args:
            AAER_INDEX_PATH: Path to the AAER index file
            MDA_INDEX_PATH: Path to the MDA index file
            FIN_INDEX_PATH: Path to the financial index file

        Returns:
            tuple: (index_aaer, index_mda, index_fin)
        """
        index_aaer = pd.read_csv(AAER_INDEX_PATH)
        logging.info("AAER index loaded")

        index_mda = pd.read_csv(MDA_INDEX_PATH, sep="|", na_values="None")
        logging.info("MDA index loaded")

        index_fin = pd.read_csv(FIN_INDEX_PATH, index_col=False)
        index_fin.drop(columns=["Unnamed: 0"], inplace=True)
        index_fin = index_fin.query(f"year < {MAX_YEAR}")

        logging.info("Financials index loaded")

        index_aaer["quarter"] = index_aaer["fiscal_quarter"].apply(lambda x: x[4:])
        index_mda["quarter"] = index_mda["quarter"].apply(lambda x: x[4:])

        return index_aaer, index_mda, index_fin

    def split_datasets(self, index_final, test_size=None):
        """
        Central method to choose and apply the appropriate dataset splitting strategy.
        Override this method in subclasses to change the splitting strategy.

        Args:
            index_final: The full dataset to split
            test_size: The proportion of data to use for testing

        Returns:
            train_index, test_index: The split datasets
        """
        raise NotImplementedError("split datasets not implemented for this class")

    def preprocess_final_dataset_index(
        self, index_aaer, index_mda, index_fin, do_train_test_splits=True
    ):
        """
        Preprocess and prepare the final dataset index by merging data sources,
        downsampling non-fraud cases, and splitting into train/test sets.

        Args:
            index_aaer: AAER index data
            index_mda: MDA index data
            index_fin: Financial index data

        Returns:
            tuple: (index_final, train_index, test_index)
        """
        seed_everything(SEED_PREPROCESSING)

        # 1. Merge the data
        index_merged, cik_to_company_dict = self.merge_data(
            index_aaer, index_mda, index_fin
        )

        # 2. Downsample non-fraud cases
        index_final, n_fraud_samples, n_non_fraud_samples = self.downsample_non_frauds(
            index_merged, self.NON_FRAUD_PCT
        )

        # 3. Save the final dataset
        dataset_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        index_final.to_csv(
            dataset_dir / "all_index.csv",
            index=False,
        )
        logging.info(f"Full dataset index saved to {dataset_dir / 'all_index.csv'}")

        # 4. Split the dataset into train and test sets
        if do_train_test_splits or True:
            train_index, test_index = self.split_datasets(
                index_final, test_size=self.DEFAULT_TEST_SIZE
            )
            # 5. Generate statistics
            self._generate_and_save_statistics(
                index_final,
                train_index,
                test_index,
                n_fraud_samples,
                n_non_fraud_samples,
                cik_to_company_dict,
            )

        return index_final

    def _generate_and_save_statistics(
        self,
        index_final,
        train_index,
        test_index,
        n_fraud_samples,
        n_non_fraud_samples,
        cik_to_company_dict,
    ):
        """
        Generate and save statistics about the dataset.

        Args:
            index_final: The complete dataset
            train_index: The training dataset
            test_index: The testing dataset
            n_fraud_samples: Number of fraud samples
            n_non_fraud_samples: Number of non-fraud samples
            cik_to_company_dict: Dictionary mapping CIKs to company names
        """
        # Load SIC code information
        sic_index_file = SICAGG_INDEX_FILE
        sic_index = pd.read_csv(sic_index_file, usecols=["sicagg", "industry_title"])
        sic_index_dict = sic_index.set_index("sicagg")["industry_title"].to_dict()

        # Add industry information to datasets
        index_final_with_industry = index_final.copy()
        index_final_with_industry["industry"] = index_final.sicagg.apply(
            lambda x: sic_index_dict.get(x, "Unknown")
        )

        # Calculate industry distributions for the entire dataset
        sic_dist = index_final_with_industry.industry.value_counts().to_dict()
        sic_dist_fraud = (
            index_final_with_industry.query("is_fraud==True")
            .industry.value_counts()
            .to_dict()
        )
        sic_dist_no_fraud = (
            index_final_with_industry.query("is_fraud==False")
            .industry.value_counts()
            .to_dict()
        )

        # Calculate industry distributions for the train set
        index_final_with_industry_train = train_index.copy()
        index_final_with_industry_train["industry"] = train_index.sicagg.apply(
            lambda x: sic_index_dict.get(x, "Unknown")
        )
        sic_dist_train = (
            index_final_with_industry_train.industry.value_counts().to_dict()
        )
        sic_dist_fraud_train = (
            index_final_with_industry_train.query("is_fraud==True")
            .industry.value_counts()
            .to_dict()
        )
        sic_dist_no_fraud_train = (
            index_final_with_industry_train.query("is_fraud==False")
            .industry.value_counts()
            .to_dict()
        )

        # Calculate industry distributions for the test set
        index_final_with_industry_test = test_index.copy()
        index_final_with_industry_test["industry"] = test_index.sicagg.apply(
            lambda x: sic_index_dict.get(x, "Unknown")
        )
        sic_dist_test = index_final_with_industry_test.industry.value_counts().to_dict()
        sic_dist_fraud_test = (
            index_final_with_industry_test.query("is_fraud==True")
            .industry.value_counts()
            .to_dict()
        )
        sic_dist_no_fraud_test = (
            index_final_with_industry_test.query("is_fraud==False")
            .industry.value_counts()
            .to_dict()
        )

        # Calculate fraud label distributions

        index_final_fraud = index_final.query("is_fraud==True")

        index_final_no_dup_cik = index_final.drop_duplicates("cik")

        # Calculate fraud distribution by year for entire dataset
        year_fraud_stats = self._calculate_year_fraud_stats(index_final)

        # Calculate fraud distribution by year for train set
        train_year_fraud_stats = self._calculate_year_fraud_stats(train_index)

        # Calculate fraud distribution by year for test set
        test_year_fraud_stats = self._calculate_year_fraud_stats(test_index)

        # Compile all statistics into a dictionary
        global_stats = {
            "n_samples": len(index_final),
            "n_companies": index_final["cik"].nunique(),
            "n_fraud_samples": n_fraud_samples,
            "n_non_fraud_samples": n_non_fraud_samples,
            "fraud_pct": round(
                100 * len(index_final[index_final.is_fraud == True]) / len(index_final),
                2,
            ),
            "split_stats": {
                "train_samples": len(train_index),
                "train_fraud_samples": len(train_index[train_index.is_fraud == True]),
                "train_non_fraud_samples": len(
                    train_index[train_index.is_fraud == False]
                ),
                "train_fraud_percentage": round(
                    100
                    * len(train_index[train_index.is_fraud == True])
                    / len(train_index)
                ),
                "test_samples": len(test_index),
                "test_fraud_samples": len(test_index[test_index.is_fraud == True]),
                "test_non_fraud_samples": len(test_index[test_index.is_fraud == False]),
                "test_fraud_percentage": round(
                    100 * len(test_index[test_index.is_fraud == True]) / len(test_index)
                ),
            },
            "year_fraud_distribution": year_fraud_stats,
            "year_fraud_distribution_train": train_year_fraud_stats,
            "year_fraud_distribution_test": test_year_fraud_stats,
            "sic_distribution": {
                "global": sic_dist,
                "non_fraud": sic_dist_no_fraud,
                "fraud": sic_dist_fraud,
            },
            "sic_distribution_train": {
                "global": sic_dist_train,
                "non_fraud": sic_dist_no_fraud_train,
                "fraud": sic_dist_fraud_train,
            },
            "sic_distribution_test": {
                "global": sic_dist_test,
                "non_fraud": sic_dist_no_fraud_test,
                "fraud": sic_dist_fraud_test,
            },
            **{
                feature: {
                    "min": float(index_final[feature].min()),
                    "max": float(index_final[feature].max()),
                    "mean": float(index_final[feature].mean()),
                    "std": float(index_final[feature].std()),
                    "median": float(index_final[feature].median()),
                }
                for feature in EXTENDED_FINANCIAL_FEATURES_COUNT_COLS
                if feature in index_final.columns
            },
            "report_per_cik": {
                "min": int(index_final_no_dup_cik.cik_reports_count.min()),
                "max": int(index_final_no_dup_cik.cik_reports_count.max()),
                "mean": float(index_final_no_dup_cik.cik_reports_count.mean()),
                "std": float(index_final_no_dup_cik.cik_reports_count.std()),
                "median": float(index_final_no_dup_cik.cik_reports_count.median()),
            },
            "mistatement_counts": index_final[LIST_MISTATEMENT_TYPE_RENAMED]
            .sum()
            .to_dict(),
            "mistatement_counts_percentage": (
                index_final[LIST_MISTATEMENT_TYPE_RENAMED].sum()
                / index_final[LIST_MISTATEMENT_TYPE_RENAMED].sum().sum()
            ).to_dict(),
            "mistatement_counts_percentage_fraud": (
                index_final_fraud[LIST_MISTATEMENT_TYPE_RENAMED].sum()
                / index_final_fraud[LIST_MISTATEMENT_TYPE_RENAMED].sum().sum()
            ).to_dict(),
            "serial_fraud_distribution": index_final_fraud.total_serial_count.value_counts().to_dict()
            if "total_serial_count" in index_final_fraud.columns
            else {},
        }

        # Save statistics
        with open(
            PREPROCESSED_PATH_EXTENDED / f"{self.DATASET_VERSION}/global_stats.yaml",
            "w",
        ) as f:
            pyaml.dump(global_stats, f)
        logging.info(
            f"Global dataset stats saved to {PREPROCESSED_PATH_EXTENDED / f'{self.DATASET_VERSION}/global_stats.yaml'}"
        )

        # Save CIK to company mapping
        with open(
            PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION / "cik_to_company.yaml",
            "w",
        ) as f:
            pyaml.dump(cik_to_company_dict, f)
        logging.info(
            f"CIK to company mapping saved to {PREPROCESSED_PATH_EXTENDED/ self.DATASET_VERSION / 'cik_to_company.yaml'}"
        )

    def _calculate_year_fraud_stats(self, dataset):
        """
        Calculate fraud statistics by year for a given dataset.

        Args:
            dataset: DataFrame containing the dataset to analyze

        Returns:
            dict: Dictionary with year-wise fraud statistics
        """
        year_fraud_stats = {}
        for year in sorted(dataset["year"].unique()):
            year_data = dataset[dataset["year"] == year]
            year_fraud_stats[
                str(year)
            ] = {  # Convert year to string for YAML compatibility
                "total": int(len(year_data)),
                "fraud": int(len(year_data[year_data.is_fraud == True])),
                "non_fraud": int(len(year_data[year_data.is_fraud == False])),
                "fraud_pct": round(
                    100
                    * len(year_data[year_data.is_fraud == True])
                    / max(1, len(year_data)),
                    2,
                ),
            }
        return year_fraud_stats

    def run(self, AAER_INDEX_PATH, MDA_INDEX_PATH, FIN_INDEX_PATH):
        """
        Main entry point to run the full preprocessing pipeline.

        Args:
            AAER_INDEX_PATH: Path to the AAER index file
            MDA_INDEX_PATH: Path to the MDA index file
            FIN_INDEX_PATH: Path to the financial index file
        """
        configure_logger(
            Path(f"sec_global_preprocessing_{time.strftime('%Y%m%d-%H%M%S')}.log"),
            logging.INFO,
        )
        begin = datetime.now()

        index_aaer, index_mda, index_fin = self.load_all_index_data(
            AAER_INDEX_PATH, MDA_INDEX_PATH, FIN_INDEX_PATH
        )

        self.preprocess_final_dataset_index(index_aaer, index_mda, index_fin)

        logging.info("Preprocessing completed")
        logging.info("Stats")
        duration = datetime.now() - begin
        logging.info(f"Process duration:{duration}")

    ###########################################
    # Dataset Splitting Methods
    ###########################################

    # Splitting function random (v5)
    def split_and_save_datasets_random(self, index_final, test_size=0.1):
        """
        Split the final index into train and test sets using stratified sampling by fraud status.
        """
        logging.info(
            f"Splitting dataset Randomly into train and test sets (test_size={test_size})"
        )

        # Use the split_dataset_randomly function from utils
        train_index, test_index = split_dataset_randomly(
            index_final, test_size=test_size
        )

        # Create directory if it doesn't exist
        dataset_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save the splits
        train_index.to_csv(
            dataset_dir / "train_index.csv",
            index=False,
        )
        test_index.to_csv(
            dataset_dir / "test_index.csv",
            index=False,
        )

        logging.info(
            f"Train set: {len(train_index)} samples, {len(train_index[train_index.is_fraud == True])} fraud"
        )
        logging.info(
            f"Test set: {len(test_index)} samples, {len(test_index[test_index.is_fraud == True])} fraud"
        )
        return train_index, test_index

    # Splitting function CIK-unbiased (v4)
    def split_and_save_datasets_cik_unbiased(self, index_final, test_size=0.1):
        """
        Split the final index into train and test sets by company to prevent data leakage,
        while preserving fraud/non-fraud ratio and maintaining industry distribution.

        This split ensures that all reports from a company are either in the training set
        or in the test set, but not split between them.
        """
        logging.info(
            f"Splitting dataset by company (CIK-unbiased) with test_size={test_size} while preserving ratios"
        )

        # Use the split_dataset_by_cik function from utils
        train_index, test_index = split_dataset_by_cik(index_final, test_size=test_size)

        # Create directory if it doesn't exist
        dataset_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save the train and test indices
        train_index.to_csv(
            dataset_dir / "train_index.csv",
            index=False,
        )
        test_index.to_csv(
            dataset_dir / "test_index.csv",
            index=False,
        )

        return train_index, test_index

    # Splitting function with time split
    def split_and_save_dataset_by_time(self, index_final, test_size=0.2):
        """
        Split the dataset based on time, ensuring all training samples come from periods
        before any test samples, with fraud ratio in test proportional to train based on test_size.

        Args:
            index_final: The full dataset to split
            test_size: The proportion of data to use for testing (most recent data)

        Returns:
            train_index, test_index: The split datasets
        """
        logging.info(f"Splitting dataset by time with test_size={test_size}")

        # Use the split_dataset_by_time function from utils
        train_index, test_index = split_dataset_by_time(
            index_final,
            test_size=test_size,
        )

        # Create directory if it doesn't exist
        dataset_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save the datasets
        train_index.to_csv(
            dataset_dir / "train_index.csv",
            index=False,
        )
        test_index.to_csv(
            dataset_dir / "test_index.csv",
            index=False,
        )

        return train_index, test_index

    ###########################################
    # K-Fold Cross Validation Methods
    ###########################################

    def split_dataset_with_fold(self, index_final, k=5):
        """
        Base method for K-fold cross-validation splitting.

        Args:
            index_final: DataFrame containing the dataset to split
            k: Number of folds

        Returns:
            dict: Results of the k-fold splitting
        """
        raise NotImplementedError(
            "Split dataset by fold not implemented for this class. Implement in subclasses."
        )

    def run_k_folds(self, aaer_index_path, mda_index_path, fin_index_path, k=5):
        """
        Run k-fold splitting using one or more splitting strategies.

        Args:
            AAER_INDEX_PATH: Path to the AAER index file
            MDA_INDEX_PATH: Path to the MDA index file
            FIN_INDEX_PATH: Path to the financial index file
            k: Number of folds to create
            split_by_cik: Whether to create a CIK-based split
            split_random: Whether to crFeate a random split
            split_by_time: Whether to create a time-based split (only supports k=1)

        Returns:
            dict: Dictionary with results from each splitting method
        """
        configure_logger(
            Path(
                f"sec_global_preprocessing_kfolds_{time.strftime('%Y%m%d-%H%M%S')}.log"
            ),
            logging.INFO,
        )
        begin = datetime.now()

        # Load the data like in the run method
        index_aaer, index_mda, index_fin = self.load_all_index_data(
            aaer_index_path, mda_index_path, fin_index_path
        )

        # Process the data to get the final index
        index_final = self.preprocess_final_dataset_index(
            index_aaer, index_mda, index_fin, do_train_test_splits=False
        )

        results = {}

        if not PREPROCESSED_PATH_EXTENDED.exists():
            PREPROCESSED_PATH_EXTENDED.mkdir(parents=True, exist_ok=True)

        # Create version-specific directory
        version_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        if not version_dir.exists():
            version_dir.mkdir(parents=True, exist_ok=True)

        # Ensure report_id column exists
        if "report_id" not in index_final.columns:
            logging.info("Adding report_id column based on index")
            index_final = index_final.reset_index().rename(
                columns={"index": "report_id"}
            )

        folds_data = self.split_dataset_with_fold(index_final=index_final, k=k)

        # Create a combined statistics file
        stats_summary = {
            "dataset_version": self.DATASET_VERSION,
            "k_folds": k,
            "total_samples": len(index_final),
            "fraud_samples": int(index_final["is_fraud"].sum()),
            "non_fraud_samples": int(len(index_final) - index_final["is_fraud"].sum()),
            "fraud_ratio": float(index_final["is_fraud"].mean()),
            "timestamp": datetime.now().isoformat(),
            "fold_stats": folds_data["subsets_stats"],
        }

        # Add industry distribution if available
        if "sicagg" in index_final.columns:
            stats_summary["industry_distribution"] = (
                index_final["sicagg"].value_counts().to_dict()
            )

        with open(
            PREPROCESSED_PATH_EXTENDED / f"{self.DATASET_VERSION}/kfolds_summary.yaml",
            "w",
        ) as f:
            pyaml.dump(stats_summary, f)

        logging.info("K-fold splitting completed")
        duration = datetime.now() - begin
        logging.info(f"Process duration: {duration}")

        logging.info(
            f"K-fold splitting completed. Summary saved to {self.DATASET_VERSION}_kfolds_summary.yaml"
        )
        return results

    def split_and_save_dataset_kfolds_by_time(self, index_final, k=1, test_size=0.1):
        """
        Splits dataset using time-based splitting. This method only supports a single fold (k=1)
        but follows the same structure as other k-fold methods for consistency.

        Args:
            index_final: DataFrame containing the dataset to split
            k: Must be 1 (included for API consistency)
            test_size: The proportion of data to use for testing (most recent data)

        Returns:
            dict: Dictionary with fold information and stats
        """
        if k != 1:
            raise Exception(f"Time-based splitting only supports k=1, {k} provided")

        logging.info(f"Performing time-based splitting with test_size={test_size}")

        # Create output directory
        root_fold_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        fold_dir = root_fold_dir / f"fold_1"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Use the split_dataset_by_time function from utils
        train_index, test_index = split_dataset_by_time(
            index_final, test_size=test_size, keep_fraud_ratio=True
        )

        # Save the datasets
        train_index.to_csv(fold_dir / "train.csv", index=False)
        test_index.to_csv(fold_dir / "test.csv", index=False)

        # Log statistics
        train_fraud_count = len(train_index[train_index.is_fraud == True])
        test_fraud_count = len(test_index[test_index.is_fraud == True])

        logging.info(f"Time-based split results:")
        logging.info(
            f"  Train set: {len(train_index)} samples, {train_fraud_count} fraud ({train_fraud_count/len(train_index):.2%})"
        )
        logging.info(
            f"  Test set: {len(test_index)} samples, {test_fraud_count} fraud ({test_fraud_count/len(test_index):.2%})"
        )

        # Calculate fold metadata
        train_years = sorted(train_index["year"].unique().tolist())
        test_years = sorted(test_index["year"].unique().tolist())

        logging.info(f"  Train years: {min(train_years)}-{max(train_years)}")
        logging.info(f"  Test years: {min(test_years)}-{max(test_years)}")

        # Create results dictionary similar to other k-fold methods
        subsets = {"0": test_index}

        train_industry_dist = train_index["sicagg"].value_counts().to_dict()
        test_industry_dist = test_index["sicagg"].value_counts().to_dict()

        train_years_dist = train_index["year"].value_counts().to_dict()
        test_years_dist = test_index["year"].value_counts().to_dict()

        folds_stats = {
            "method": "time_based",
            "test_size": test_size,
            "1": {
                "train": {
                    "samples": len(train_index),
                    "fraud": int(train_index["is_fraud"].sum()),
                    "non_fraud": int(len(train_index) - train_index["is_fraud"].sum()),
                    "fraud_ratio": float(train_index["is_fraud"].mean()),
                    "years": train_years,
                    "industry_distribution": train_industry_dist,
                    "years_distribution": train_years_dist,
                },
                "test": {
                    "samples": len(test_index),
                    "fraud": int(test_index["is_fraud"].sum()),
                    "non_fraud": int(len(test_index) - test_index["is_fraud"].sum()),
                    "fraud_ratio": float(test_index["is_fraud"].mean()),
                    "years": test_years,
                    "industry_distribution": test_industry_dist,
                    "years_distribution": test_years_dist,
                },
            },
        }

        # Save fold statistics
        with open(fold_dir / "time_split_stats.yaml", "w") as f:
            pyaml.dump(folds_stats, f)

        return {"subsets": subsets, "subsets_stats": folds_stats}

    def split_and_save_dataset_kfolds_by_cik(self, index_final, k=5):
        """
        Splits dataset into k folds using CIK-based splitting to ensure no company overlap between folds.

        Args:
            index_final: DataFrame containing the dataset to split
            k: Number of folds to create

        Returns:
            dict: Dictionary with fold information and stats
        """
        from researchpkg.anomaly_detection.models.utils import (
            generate_k_folds_subsets_by_cik,
        )

        logging.info(f"Splitting dataset into {k} folds using CIK-based splitting")

        # Create output directory
        root_fold_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        root_fold_dir.mkdir(parents=True, exist_ok=True)

        # Generate k folds with no company overlap
        folds_data = generate_k_folds_subsets_by_cik(index_final, k)

        # Save each fold as a separate file and create train/val splits
        for fold_id, fold_df in folds_data["subsets"].items():
            # Save this fold as validation set
            fold_dir = root_fold_dir / f"fold_{fold_id}"
            fold_dir.mkdir(exist_ok=True)
            fold_df.to_csv(fold_dir / f"test.csv", index=False)

            # Create training set by combining all other folds
            train_dfs = [
                df for fid, df in folds_data["subsets"].items() if fid != fold_id
            ]
            train_df = pd.concat(train_dfs)

            # Save training set
            train_df.to_csv(fold_dir / f"train.csv", index=False)

            # Log statistics
            logging.info(f"Fold {fold_id}:")
            logging.info(
                f"  Validation: {len(fold_df)} samples, {fold_df['is_fraud'].sum()} fraud, "
                f"{len(fold_df['cik'].unique())} companies"
            )
            logging.info(
                f"  Train: {len(train_df)} samples, {train_df['is_fraud'].sum()} fraud, "
                f"{len(train_df['cik'].unique())} companies"
            )

        # Save fold statistics
        with open(root_fold_dir / "folds_stats.yaml", "w") as f:
            pyaml.dump(folds_data["subsets_stats"], f)

        logging.info(f"K-fold split (CIK-based) saved to {root_fold_dir}")
        return folds_data

    def split_and_save_dataset_kfolds_random(self, index_final, k=5):
        """
        Splits dataset into k folds using random splitting.

        Args:
            index_final: DataFrame containing the dataset to split
            k: Number of folds to create

        Returns:
            dict: Dictionary with fold information and stats
        """
        from researchpkg.anomaly_detection.models.utils import (
            generate_k_folds_random,
        )

        logging.info(f"Splitting dataset into {k} folds using random splitting")

        # Create output directory
        root_fold_dir = PREPROCESSED_PATH_EXTENDED / self.DATASET_VERSION
        root_fold_dir.mkdir(parents=True, exist_ok=True)

        # Generate k random folds
        folds_data = generate_k_folds_random(index_final, k)

        # Save each fold as a separate file and create train/val splits
        for fold_id, fold_df in folds_data["subsets"].items():
            # Create fold directory
            fold_dir = root_fold_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Save this fold as validation set
            fold_df.to_csv(fold_dir / "test.csv", index=False)

            # Create training set by combining all other folds
            train_dfs = [
                df for fid, df in folds_data["subsets"].items() if fid != fold_id
            ]
            train_df = pd.concat(train_dfs)

            # Save training set
            train_df.to_csv(fold_dir / f"train.csv", index=False)

            # Log statistics
            logging.info(f"Random Fold {fold_id}:")
            logging.info(
                f"  Test set: {len(fold_df)} samples, {fold_df['is_fraud'].sum()} fraud"
            )
            logging.info(
                f"  Train set: {len(train_df)} samples, {train_df['is_fraud'].sum()} fraud"
            )

            if "cik" in fold_df.columns:
                test_ciks = set(fold_df["cik"].unique())
                train_ciks = set(train_df["cik"].unique())
                overlap = test_ciks.intersection(train_ciks)

                logging.info(f"  Unique companies in test: {len(test_ciks)}")
                logging.info(f"  Unique companies in train: {len(train_ciks)}")
                logging.info(f"  Company overlap: {len(overlap)} companies")

        # Save fold statistics
        with open(root_fold_dir / "folds_stats.yaml", "w") as f:
            pyaml.dump(folds_data["subsets_stats"], f)

        logging.info(f"K-fold split (random) saved to {root_fold_dir}")
        return folds_data
