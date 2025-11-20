import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    GLABEL_MAPPING,
    PREPROCESSED_PATH_NOAUG,
    SEED_PREPROCESSING,
    SIC_INDEX_FILE,
    SICAGG_INDEX_FILE,
)
from researchpkg.anomaly_detection.preprocessing.noaug.sec_financial_preprocessing_quarterly_noaug import (
    NOAUG_FINANCIAL_FEATURES_COUNT_COLS,
)
from researchpkg.utils import configure_logger


class NoAugPreprocessing:
    NON_FRAUD_PCT = None
    DATASET_NAME = None

    def __init__(self):
        self.label_to_glabel_dict = {}
        for glabel in GLABEL_MAPPING:
            for l in GLABEL_MAPPING[glabel]:
                self.label_to_glabel_dict[l] = glabel

    def load_all_index_data(self, AAER_INDEX_PATH, MDA_INDEX_PATH, FIN_INDEX_PATH):
        index_aaer = pd.read_excel(AAER_INDEX_PATH)
        index_aaer.rename(columns={"tags": "labels"}, inplace=True)
        index_aaer.drop(columns=["Unnamed: 0"], inplace=True)
        logging.info("AAER index loaded")

        index_mda = pd.read_excel(MDA_INDEX_PATH, na_values="None")
        logging.info("MDA index loaded")

        index_fin = pd.read_csv(FIN_INDEX_PATH, index_col=False)
        index_fin.drop(columns=["Unnamed: 0"], inplace=True)
        logging.info("Financials index loaded")

        index_aaer["quarter"] = index_aaer["fiscal_quarter"].apply(lambda x: x[4:])
        index_mda["quarter"] = index_mda["quarter"].apply(lambda x: x[4:])

        return index_aaer, index_mda, index_fin

    def split_and_save_datasets(self, index_final, test_size=0.1):
        """
        Split the final index into train and test sets using stratified sampling by fraud status.
        """
        logging.info(
            f"Splitting dataset into train and test sets (test_size={test_size})"
        )

        # Stratify by is_fraud to maintain the same fraud/non-fraud ratio in both sets
        train_index, test_index = train_test_split(
            index_final,
            test_size=test_size,
            random_state=SEED_PREPROCESSING,
            stratify=index_final["is_fraud"],
        )

        train_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_train_index.csv",
            index=False,
        )
        test_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_test_index.csv",
            index=False,
        )

        logging.info(
            f"Train set: {len(train_index)} samples, {len(train_index[train_index.is_fraud == True])} fraud"
        )
        logging.info(
            f"Test set: {len(test_index)} samples, {len(test_index[test_index.is_fraud == True])} fraud"
        )

        return train_index, test_index

    def split_and_save_datasets_cik_unbiased(self, index_final, test_size=0.1):
        """
        Split the final index into train and test sets by company to prevent data leakage.

        This split ensures that all reports from a company are either in the training set
        or in the test set, but not split between them.
        """
        logging.info(
            f"Splitting dataset by company (CIK-unbiased) with test_size={test_size}"
        )

        # Group at the company level
        unique_companies = (
            index_final.groupby("cik")["is_fraud"].max().reset_index(name="any_fraud")
        )

        # Stratify by the aggregated fraud indicator
        train_companies, test_companies = train_test_split(
            unique_companies,
            test_size=test_size,
            random_state=SEED_PREPROCESSING,
            stratify=unique_companies["any_fraud"],
        )

        intersect = set(train_companies["cik"]).intersection(test_companies["cik"])
        assert (
            len(intersect) == 0
        ), f"No company should be in both train and test set. Found : {len(intersect)} companies in both: {intersect}"

        train_index = index_final[index_final.cik.isin(train_companies.cik)]
        test_index = index_final[index_final.cik.isin(test_companies.cik)]

        # Save the train and test indices
        train_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_train_index.csv",
            index=False,
        )
        test_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_test_index.csv",
            index=False,
        )

        # Log the split statistics
        n_train_samples = len(train_index)
        n_train_fraud = len(train_index[train_index.is_fraud == True])
        n_train_non_fraud = len(train_index[train_index.is_fraud == False])
        n_train_companies = len(train_companies)
        n_train_fraud_companies = len(
            train_companies[train_companies.any_fraud == True]
        )

        n_test_samples = len(test_index)
        n_test_fraud = len(test_index[test_index.is_fraud == True])
        n_test_non_fraud = len(test_index[test_index.is_fraud == False])
        n_test_companies = len(test_companies)
        n_test_fraud_companies = len(test_companies[test_companies.any_fraud == True])

        logging.info(
            f"Train set: {n_train_samples} samples, {n_train_fraud} fraud ({n_train_fraud/n_train_samples:.2%}), "
            f"{n_train_companies} companies, {n_train_fraud_companies} fraud companies"
        )
        logging.info(
            f"Test set: {n_test_samples} samples, {n_test_fraud} fraud ({n_test_fraud/n_test_samples:.2%}), "
            f"{n_test_companies} companies, {n_test_fraud_companies} fraud companies"
        )

        return train_index, test_index

    def split_and_save_dataset_ciks_timesplit(self, index_final, test_size=0.1):
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

        # Create a time column for easier sorting
        index_final = index_final.copy()
        index_final["year_int"] = index_final["year"].astype(int)
        index_final["quarter_int"] = index_final["quarter"].apply(lambda x: int(x[-1]))
        index_final["time_value"] = (
            index_final["year_int"] * 10 + index_final["quarter_int"]
        )

        # Sort by time value
        index_final = index_final.sort_values("time_value")

        # Identify unique time periods
        time_periods = (
            index_final.groupby("time_value")[["is_fraud", "cik"]]
            .agg(
                {
                    "is_fraud": "sum",  # Count of fraud cases per period
                    "cik": "count",  # Total count of records per period
                }
            )
            .reset_index()
        )
        time_periods.rename(
            columns={"is_fraud": "fraud_count", "cik": "total_count"}, inplace=True
        )
        time_periods = time_periods.sort_values("time_value")

        # Calculate cumulative counts
        time_periods["cum_total"] = time_periods["total_count"].cumsum()
        time_periods["cum_fraud"] = time_periods["fraud_count"].cumsum()

        # Total counts
        total_records = time_periods["total_count"].sum()
        total_fraud = time_periods["fraud_count"].sum()

        # Target train/test split for fraud cases
        target_train_fraud = total_fraud * (1 - test_size)

        # Find the cutoff point where fraud ratio is closest to desired
        cutoff_time = time_periods[
            time_periods["cum_fraud"] >= target_train_fraud
        ].iloc[0]["time_value"]

        # Split the data
        train_index = index_final[index_final["time_value"] < cutoff_time].copy()
        test_index = index_final[index_final["time_value"] >= cutoff_time].copy()

        # Drop the temporary columns used for sorting
        train_index = train_index.drop(
            ["year_int", "quarter_int", "time_value"], axis=1
        )
        test_index = test_index.drop(["year_int", "quarter_int", "time_value"], axis=1)

        # Save the datasets
        train_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_train_index.csv",
            index=False,
        )
        test_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_test_index.csv",
            index=False,
        )

        # Calculate statistics
        n_train_samples = len(train_index)
        n_train_fraud = len(train_index[train_index.is_fraud == True])
        n_train_companies = len(train_index["cik"].unique())
        n_train_fraud_companies = len(
            train_index[train_index.is_fraud == True]["cik"].unique()
        )

        n_test_samples = len(test_index)
        n_test_fraud = len(test_index[test_index.is_fraud == True])
        n_test_companies = len(test_index["cik"].unique())
        n_test_fraud_companies = len(
            test_index[test_index.is_fraud == True]["cik"].unique()
        )

        # Calculate fraud ratios and test ratio
        train_fraud_ratio = (
            n_train_fraud / n_train_samples if n_train_samples > 0 else 0
        )
        test_fraud_ratio = n_test_fraud / n_test_samples if n_test_samples > 0 else 0
        actual_test_size = n_test_samples / (n_train_samples + n_test_samples)
        fraud_test_ratio = (
            n_test_fraud / (n_train_fraud + n_test_fraud)
            if (n_train_fraud + n_test_fraud) > 0
            else 0
        )

        # Log the split information
        cutoff_year = cutoff_time // 10
        cutoff_quarter = cutoff_time % 10

        earliest_train = (
            f"{int(train_index['year'].min())}-Q{int(train_index['quarter'].min()[-1])}"
            if n_train_samples > 0
            else "N/A"
        )
        latest_train = (
            f"{int(train_index['year'].max())}-Q{int(train_index['quarter'].max()[-1])}"
            if n_train_samples > 0
            else "N/A"
        )
        earliest_test = (
            f"{int(test_index['year'].min())}-Q{int(test_index['quarter'].min()[-1])}"
            if n_test_samples > 0
            else "N/A"
        )
        latest_test = (
            f"{int(test_index['year'].max())}-Q{int(test_index['quarter'].max()[-1])}"
            if n_test_samples > 0
            else "N/A"
        )

        logging.info(f"Time split cutoff: {cutoff_year}Q{cutoff_quarter}")
        logging.info(
            f"Time periods - Train: {earliest_train} to {latest_train}, Test: {earliest_test} to {latest_test}"
        )
        logging.info(
            f"Target test_size: {test_size:.2%}, Actual: {actual_test_size:.2%}"
        )
        logging.info(
            f"Fraud split ratio: {fraud_test_ratio:.2%} (should be close to {test_size:.2%})"
        )
        logging.info(
            f"Train set: {n_train_samples} samples, {n_train_fraud} fraud ({train_fraud_ratio:.2%}), "
            f"{n_train_companies} companies, {n_train_fraud_companies} fraud companies"
        )
        logging.info(
            f"Test set: {n_test_samples} samples, {n_test_fraud} fraud ({test_fraud_ratio:.2%}), "
            f"{n_test_companies} companies, {n_test_fraud_companies} fraud companies"
        )

        # Check if companies overlap between train and test (expected since this is a time split)
        train_companies = set(train_index["cik"])
        test_companies = set(test_index["cik"])
        overlap_companies = train_companies.intersection(test_companies)
        overlap_percentage = (
            len(overlap_companies) / len(train_companies.union(test_companies))
            if len(train_companies.union(test_companies)) > 0
            else 0
        )
        logging.info(
            f"Companies in both train and test sets: {len(overlap_companies)} ({overlap_percentage:.2%} of all companies)"
        )

        return train_index, test_index

    def preprocess_final_dataset_index(self, index_aaer, index_mda, index_fin):
        index_aaer["company"] = index_aaer.company.str.lower()
        index_mda["company"] = index_mda.company.str.lower()
        index_fin["company"] = index_fin.company.str.lower()

        cik_to_company_dict = dict(zip(index_fin["cik"], index_fin["company"]))
        cik_to_company_dict.update(dict(zip(index_mda["cik"], index_mda["company"])))
        cik_to_company_dict.update(dict(zip(index_aaer["cik"], index_aaer["company"])))

        index_aaer.drop(columns=["company"], inplace=True)
        index_mda.drop(columns=["company"], inplace=True)
        index_fin.drop(columns=["company"], inplace=True)

        index_aaer["glabels"] = index_aaer["labels"].apply(
            lambda x: ";".join(
                set([self.label_to_glabel_dict[t] for t in x.split(";")])
            )
        )

        index_aaer["num_glabels"] = index_aaer["glabels"].apply(
            lambda x: len(x.split(";")) if type(x) == str else 0
        )

        logging.info("Merging indexes")
        index_merged = pd.merge(
            index_aaer, index_mda, on=["cik", "year", "quarter"], how="outer"
        )
        index_merged = pd.merge(index_merged, index_fin, on=["cik", "year", "quarter"])
        index_merged.drop_duplicates(inplace=True)
        index_merged["is_fraud"] = index_merged.aaer_no.notna()

        index_merged = index_merged[index_merged["mda_quarter_id"].notna()]

        logging.info("Enrich features")
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

        index_merged = index_merged.merge(
            index_merged_with_aaer, on=["cik", "year", "quarter"], how="left"
        )

        index_merged["cik_reports_count"] = index_merged.groupby("cik")[
            "quarter"
        ].transform("count")

        index_merged_with_fraud = index_merged.query("is_fraud==True")
        index_merged_no_fraud = index_merged.query("is_fraud==False")

        size_frauds = len(index_merged_with_fraud)
        target_size_non_frauds = int(
            (self.NON_FRAUD_PCT / (100 - self.NON_FRAUD_PCT)) * size_frauds
        )

        # Instead of random sampling across all non-fraud cases, we'll sample by year
        index_merged_with_fraud = index_merged.query("is_fraud==True")
        index_merged_no_fraud = index_merged.query("is_fraud==False")

        size_frauds = len(index_merged_with_fraud)
        target_size_non_frauds = int(
            (self.NON_FRAUD_PCT / (100 - self.NON_FRAUD_PCT)) * size_frauds
        )

        # Calculate fraud distribution by year
        fraud_by_year = index_merged_with_fraud.groupby("year").size()
        all_years = sorted(index_merged["year"].unique())

        logging.info(
            "Sampling non-fraud cases by year to maintain consistent fraud ratio"
        )

        # Sample non-fraud for each year while maintaining the overall target ratio
        sampled_non_fraud_dfs = []

        for year in all_years:
            year_fraud_count = fraud_by_year.get(year, 0)
            year_non_fraud = index_merged_no_fraud[
                index_merged_no_fraud["year"] == year
            ]

            if year_fraud_count == 0 or len(year_non_fraud) == 0:
                # If no fraud cases this year or no non-fraud cases, skip or handle specially
                if len(year_non_fraud) > 0:
                    # Include a small sample of non-fraud from this year
                    year_sample_size = min(
                        len(year_non_fraud), int(target_size_non_frauds * 0.01)
                    )  # 1% of target
                    if year_sample_size > 0:
                        year_sample = year_non_fraud.sample(
                            n=year_sample_size, random_state=SEED_PREPROCESSING
                        )
                        sampled_non_fraud_dfs.append(year_sample)
                continue

            # Calculate year target non-fraud based on fraud distribution and overall target
            year_target_non_fraud = int(
                (year_fraud_count / size_frauds) * target_size_non_frauds
            )

            # Adjust if we don't have enough non-fraud for this year
            year_sample_size = min(len(year_non_fraud), year_target_non_fraud)

            if year_sample_size > 0:
                year_sample = year_non_fraud.sample(
                    n=year_sample_size, random_state=SEED_PREPROCESSING
                )
                sampled_non_fraud_dfs.append(year_sample)

                logging.info(
                    f"Year {year}: {year_fraud_count} fraud, sampled {year_sample_size} non-fraud "
                    + f"(ratio: {year_fraud_count/(year_fraud_count+year_sample_size):.2%})"
                )

        # Combine all sampled non-fraud
        sampled_non_fraud = pd.concat(sampled_non_fraud_dfs)

        # Check if we need to adjust to hit the target non-fraud count
        remaining_needed = target_size_non_frauds - len(sampled_non_fraud)

        if remaining_needed > 0:
            # Get remaining non-fraud cases (those not already sampled)
            remaining_non_fraud = index_merged_no_fraud[
                ~index_merged_no_fraud.index.isin(sampled_non_fraud.index)
            ]

            # If we still have non-sampled cases and need more
            if len(remaining_non_fraud) > 0:
                sample_size = min(len(remaining_non_fraud), remaining_needed)
                additional_sample = remaining_non_fraud.sample(
                    n=sample_size, random_state=SEED_PREPROCESSING
                )
                sampled_non_fraud = pd.concat([sampled_non_fraud, additional_sample])
                logging.info(
                    f"Added {sample_size} additional non-fraud samples to reach target"
                )

        index_final = pd.concat([index_merged_with_fraud, sampled_non_fraud])
        logging.info(f"Size of the final index: {len(index_final)}")

        # Verify the year distribution of fraud/non-fraud
        year_distribution = index_final.groupby("year")["is_fraud"].agg(
            ["mean", "count"]
        )
        logging.info("Fraud ratio by year in the final dataset:")
        for year, row in year_distribution.iterrows():
            logging.info(
                f"Year {year}: {row['mean']:.2%} fraud ratio, {row['count']} samples"
            )

        n_fraud_samples = len(index_final[index_final.is_fraud == True])
        n_non_fraud_samples = len(index_final[index_final.is_fraud == False])

        pct_fraud = n_fraud_samples / (n_fraud_samples + n_non_fraud_samples)
        logging.info(
            f"Overall percentage fraud samples: {pct_fraud:.2%}, {n_fraud_samples} / {n_fraud_samples+n_non_fraud_samples}"
        )

        if not PREPROCESSED_PATH_NOAUG.exists():
            PREPROCESSED_PATH_NOAUG.mkdir(parents=True, exist_ok=True)
        index_final.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_index.csv",
            index=False,
        )
        logging.info(
            f"Full dataset index saved to {PREPROCESSED_PATH_NOAUG / f'{self.DATASET_NAME}index.csv'}"
        )

        train_index, test_index = self.split_and_save_datasets(
            index_final, test_size=0.1
        )

        sic_index_file = SICAGG_INDEX_FILE
        sic_index = pd.read_csv(sic_index_file, usecols=["sicagg", "industry_title"])
        sic_index_dict = sic_index.set_index("sicagg")["industry_title"].to_dict()

        index_final_with_industry = index_final.copy()
        index_final_with_industry["industry"] = index_final.sicagg.apply(
            lambda x: sic_index_dict[x]
        )

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

        # Same distribution but for train and test
        index_final_with_industry_train = train_index.copy()
        index_final_with_industry_train["industry"] = train_index.sicagg.apply(
            lambda x: sic_index_dict[x]
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
        index_final_with_industry_test = test_index.copy()
        index_final_with_industry_test["industry"] = test_index.sicagg.apply(
            lambda x: sic_index_dict[x]
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

        # Same distribution but for train and test
        index_final_with_industry_train = train_index.copy()
        index_final_with_industry_train["industry"] = train_index.sicagg.apply(
            lambda x: sic_index_dict[x]
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
        index_final_with_industry_test = test_index.copy()
        index_final_with_industry_test["industry"] = test_index.sicagg.apply(
            lambda x: sic_index_dict[x]
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

        glabels_count = {}
        index_final_fraud = index_final.query("is_fraud==True")
        for glabel_list in index_final_fraud.glabels.values:
            for l in glabel_list.split(";"):
                glabels_count[l] = 1 + glabels_count.get(l, 0)
        glabels_count = dict(sorted(glabels_count.items(), key=lambda x: x[1]))

        index_final_no_dup_cik = index_final.drop_duplicates("cik")

        global_stats = {
            "n_samples": len(index_final),
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
                "test_samples": len(test_index),
                "test_fraud_samples": len(test_index[test_index.is_fraud == True]),
                "test_non_fraud_samples": len(test_index[test_index.is_fraud == False]),
            },
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
                    "min": index_final[feature].min(),
                    "max": index_final[feature].max(),
                    "mean": index_final[feature].mean(),
                    "std": index_final[feature].std(),
                    "median": index_final[feature].median(),
                }
                for feature in NOAUG_FINANCIAL_FEATURES_COUNT_COLS
            },
            "report_per_cik": {
                "min": index_final_no_dup_cik.cik_reports_count.min(),
                "max": index_final_no_dup_cik.cik_reports_count.max(),
                "mean": index_final_no_dup_cik.cik_reports_count.mean(),
                "std": index_final_no_dup_cik.cik_reports_count.std(),
                "median": index_final_no_dup_cik.cik_reports_count.std(),
            },
            "serial_fraud_distribution": index_final_fraud.total_serial_count.value_counts().to_dict(),
            "fraud_labels_distribution": glabels_count,
        }

        with open(PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}.yaml", "w") as f:
            pyaml.dump(global_stats, f)
        logging.info(
            f"Global dataset stats saved to {PREPROCESSED_PATH_NOAUG / f'{self.DATASET_NAME}_stats.yaml'}"
        )

        with open(PREPROCESSED_PATH_NOAUG / "cik_to_company.yaml", "w") as f:
            pyaml.dump(cik_to_company_dict, f)
        logging.info(
            f"CIK to company mapping saved to {PREPROCESSED_PATH_NOAUG / 'cik_to_company.yaml'}"
        )

        return index_final, train_index, test_index

    def run(self, AAER_INDEX_PATH, MDA_INDEX_PATH, FIN_INDEX_PATH):
        configure_logger(
            Path(f"sec_global_preprocessing_{time.strftime('%Y%m%d-%H%M%S')}.log"),
            logging.INFO,
        )
        begin = datetime.now()

        index_aaer, index_mda, index_fin = self.load_all_index_data(
            AAER_INDEX_PATH, MDA_INDEX_PATH, FIN_INDEX_PATH
        )

        index_final, train_index, test_index = self.preprocess_final_dataset_index(
            index_aaer, index_mda, index_fin
        )

        train_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_train_index.csv",
            index=False,
        )
        test_index.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_test_index.csv",
            index=False,
        )
        index_final.to_csv(
            PREPROCESSED_PATH_NOAUG / f"{self.DATASET_NAME}_index.csv",
            index=False,
        )

        logging.info("Preprocessing completed")
        logging.info("Stats")
        duration = datetime.now() - begin
        logging.info(f"Process duration:{duration}")
