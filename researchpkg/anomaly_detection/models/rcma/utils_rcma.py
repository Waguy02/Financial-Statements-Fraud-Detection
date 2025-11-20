import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_EXTENDED,
    PREPROCESSED_PATH,
)
from researchpkg.anomaly_detection.models.utils import (
    get_train_test_splitter,
    load_cross_validation_path,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
)

MDA_PATH_SUMMARIZED = PREPROCESSED_PATH / "SEC_MDA_SUMMARIZED" / "quarterly"
FULL_FINANCIAL_PATH = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"


def load_mda_content(mda_quarter_id):
    """Loads MDA content from a file."""
    mda_file = MDA_PATH_SUMMARIZED / f"{mda_quarter_id}.txt"
    if not mda_file.exists():
        logging.warning(f"MDA file {mda_file} does not exist.")
        return "No MDA content available."
    with open(mda_file, "r", encoding="utf-8") as file:
        return file.read()


def get_last_financial_data(df, full_financials_df):
    """
    Efficiently get the *most recent* previous quarter's financial data.

    Args:
        df (pd.DataFrame): DataFrame with 'cik', 'year', 'quarter' columns.
        full_financials_df (pd.DataFrame): DataFrame with full financial data.

    Returns:
        numpy array of the same size as df[FEATURES]
        numpy array of the same size as df[FEATURES] with current ones if there are some past ones
    """
    logging.info("Getting the most recent previous quarter's financials...")
    sic_agg = df["sicagg"].unique()
    full_financials_df = full_financials_df[full_financials_df["sicagg"].isin(sic_agg)]

    # Create a combined key for merging
    df["key"] = (
        df["cik"].astype(str)
        + "_"
        + df["year"].astype(str)
        + "_"
        + df["quarter"].astype(str)
    )
    full_financials_df["key"] = (
        full_financials_df["cik"].astype(str)
        + "_"
        + full_financials_df["year"].astype(str)
        + "_"
        + full_financials_df["quarter"].astype(str)
    )

    # Shift the keys to get the *previous* quarter's key
    full_financials_df["prev_key"] = full_financials_df.groupby("cik")["key"].shift(1)

    # Make a lookup map for fast feature assignment
    feature_map = (
        full_financials_df.dropna(subset=["prev_key"])
        .drop_duplicates(subset=["prev_key"], keep="first")
        .set_index("prev_key")[EXTENDED_FINANCIAL_FEATURES]
        .to_dict("index")
    )

    # Store results here
    final_data = np.zeros((len(df), len(EXTENDED_FINANCIAL_FEATURES)))

    # Fetch data, defaulting to current features
    num_missing = 0
    # Use enumerate to track the index
    for i in tqdm(range(len(df)), desc="Fetching previous financial data", unit="row"):
        row = df.iloc[i]
        # Get the previous data point using the key
        prev_data_point = feature_map.get(row["key"])

        # Default copy (current row)
        final_data[i, :] = row[EXTENDED_FINANCIAL_FEATURES].values

        # try fetching preious data is previous
        if prev_data_point:
            final_data[i, :] = np.array(list(prev_data_point.values()))
        else:
            num_missing += 1

    logging.info(
        f"Missing previous financial data for {num_missing} rows over {len(df)}"
    )
    return final_data


def get_average_industry_level_feature(full_financials_df):
    """Calculates average industry-level features."""
    df_sicagg = full_financials_df[
        ["sicagg", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES
    ].copy()  # Avoid modifying original

    df_sicagg = (
        df_sicagg.groupby(["sicagg", "year"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    df_sicagg = df_sicagg.drop_duplicates(subset=["sicagg", "year"], keep="first")
    return df_sicagg


def get_top5_industry_level_feature(full_financials_df):
    """Calculates top 5 industry-level features."""
    df_sicagg = full_financials_df[
        ["cik", "sicagg", "year", "quarter"] + EXTENDED_FINANCIAL_FEATURES
    ].copy()  #

    df_sicagg = (
        df_sicagg.groupby(["cik", "sicagg", "year"])
        .agg({f: "max" for f in EXTENDED_FINANCIAL_FEATURES})
        .reset_index()
        .drop(columns=["cik"])
    )

    df_sicagg = df_sicagg.sort_values(
        by=["sicagg", "year", "Assets"], ascending=[True, True, False]
    )
    df_sicagg = (
        df_sicagg.groupby(["sicagg", "year"], dropna=False)
        .head(5)
        .reset_index(drop=True)
    )
    df_sicagg = (
        df_sicagg.groupby(["sicagg", "year"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    df_sicagg = df_sicagg.drop_duplicates(subset=["sicagg", "year"], keep="first")
    return df_sicagg


def compute_rcma_group_ratios_features(
    X_data, prev_data, average_industry_level_data, top5_industry_level_data
):
    """Computes RCMA group ratios features."""
    EXTENDED_FINANCIAL_FEATURES_NUMERIC = EXTENDED_FINANCIAL_FEATURES

    logging.info("Computing RCMA feature groups...")
    n_samples = len(X_data)
    n_features = len(EXTENDED_FINANCIAL_FEATURES_NUMERIC)
    n_groups = 7

    X_rcma = np.zeros((n_samples, n_features, n_groups))

    # Group 1: Individual financial features (G1)
    X_rcma[:, :, 0] = X_data.values

    # Group 2 & 3: Organizational features (G2 & G3)
    if prev_data is None or len(prev_data) == 0:
        logging.warning(
            "No prior data available to compute organizational features. Filling with organizational data with zeros."
        )
        X_rcma[:, :, 1:3] = 0
    else:
        for idx in range(n_samples):

            # Get Last quarte feature if is avaliable
            last_quarter_features = prev_data[idx, :]

            # Series for vector operations

            current_features = X_data.iloc[idx]
            X_rcma[idx, :, 1] = (
                (current_features - last_quarter_features).fillna(0).values
            )
            X_rcma[idx, :, 2] = (
                (current_features / last_quarter_features)
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
                .values
            )

    # Group 4 & 5: Average industry-level features (G4 & G5)
    X_rcma[:, :, 3] = (
        X_data.values
        - average_industry_level_data[EXTENDED_FINANCIAL_FEATURES_NUMERIC].values
    )

    X_rcma[:, :, 4] = (
        (
            X_data.values
            / average_industry_level_data[EXTENDED_FINANCIAL_FEATURES_NUMERIC]
        )
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )

    # Group 6 & 7: Top-5 industry-level features (G6 & G7)
    X_rcma[:, :, 5] = (
        X_data.values
        - top5_industry_level_data[EXTENDED_FINANCIAL_FEATURES_NUMERIC].values
    )

    X_rcma[:, :, 6] = (
        (X_data.values / top5_industry_level_data[EXTENDED_FINANCIAL_FEATURES_NUMERIC])
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )

    logging.info("RCMA feature groups computed.")

    # if raw features are zero anywhre, set the rcma features to zero
    X_rcma[X_data.values == 0] = 0
    return X_rcma


# Remaining code is the same
def load_rcma_dataset(
    dataset_version,
    train_path=None,
    test_path=None,
    val_path=None,
    full_financial_path=None,
    fold_id=None,
):
    """Loads the RCMA dataset."""
    logging.info(f"Loading dataset version: {dataset_version}")

    # Determine paths based on dataset version
    if train_path is None or test_path is None:
        train_path, test_path = load_cross_validation_path(
            {
                "dataset_version": dataset_version,
                "cross_validation": True,
                "fold_id": fold_id,
                "dataset_name": dataset_version.replace("_kfolds", ""),
            }
        )

    # Determine feature set based on dataset version
    if dataset_version.startswith(("company_isolated_splitting", "time_splitting", "time_splitting")):
        from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
            EXTENDED_FINANCIAL_FEATURES,
            EXTENDED_FINANCIAL_FEATURES_COUNT_COLS,
        )

        features = EXTENDED_FINANCIAL_FEATURES
        count_cols = EXTENDED_FINANCIAL_FEATURES_COUNT_COLS

        if full_financial_path is None:
            from researchpkg.anomaly_detection.config import FINANCIALS_DIR_EXTENDED

            full_financial_path = (
                FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
            )
    else:
        raise ValueError(f"Unsupported dataset version: {dataset_version}")

    # Load full financial data
    logging.info(f"Loading full financials data from {full_financial_path}")
    full_df = pd.read_csv(full_financial_path)
    full_df = full_df[["cik", "year", "quarter"] + features]

    # Helper function to merge financial data
    def merge_with_financials(df):
        df = df.drop(columns=count_cols, errors="ignore")  # Safely drop count columns
        df = df.merge(full_df, on=["cik", "year", "quarter"], how="left")
        return df

    # Load training data
    logging.info(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    train_df = merge_with_financials(train_df)

    assert "sicagg_x" not in train_df.columns
    assert "sicagg" in train_df.columns

    # Handle validation data
    if val_path is not None:
        logging.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path)
        val_df = merge_with_financials(val_df)
    else:
        # Create validation split from training data
        splitter = get_train_test_splitter({"dataset_version": dataset_version})
        train_df, val_df = splitter(train_df, test_size=0.1)

    # Load test data
    logging.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    test_df = merge_with_financials(test_df)

    # Define target variables
    y_train = train_df["is_fraud"].astype(int)
    y_val = val_df["is_fraud"].astype(int)
    y_test = test_df["is_fraud"].astype(int)

    mda_id_train = train_df["mda_quarter_id"].values.tolist()
    mda_id_val = val_df["mda_quarter_id"].values.tolist()
    mda_id_test = test_df["mda_quarter_id"].values.tolist()

    mda_content_train = [load_mda_content(mda_id) for mda_id in mda_id_train]
    mda_content_val = [load_mda_content(mda_id) for mda_id in mda_id_val]
    mda_content_test = [load_mda_content(mda_id) for mda_id in mda_id_test]

    feature_cols = [col for col in features if col in train_df.columns]

    X_train_pre_rcma = train_df[feature_cols]
    X_val_pre_rcma = val_df[feature_cols]
    X_test_pre_rcma = test_df[feature_cols]

    # Prepare previous data for group ratios calculation
    full_financials_df = pd.read_csv(FULL_FINANCIAL_PATH)

    train_df["quarter_num"] = train_df["quarter"].map(
        {"q1": 1, "q2": 2, "q3": 3, "q4": 4}
    )
    val_df["quarter_num"] = val_df["quarter"].map({"q1": 1, "q2": 2, "q3": 3, "q4": 4})
    test_df["quarter_num"] = test_df["quarter"].map(
        {"q1": 1, "q2": 2, "q3": 3, "q4": 4}
    )

    full_financials_df["quarter_num"] = full_financials_df["quarter"].map(
        {"q1": 1, "q2": 2, "q3": 3, "q4": 4}
    )

    logging.info("Getting last financials...")

    # Add the key before call
    train_df["key"] = (
        train_df["cik"].astype(str)
        + "_"
        + train_df["year"].astype(str)
        + "_"
        + train_df["quarter"].astype(str)
    )
    val_df["key"] = (
        val_df["cik"].astype(str)
        + "_"
        + val_df["year"].astype(str)
        + "_"
        + val_df["quarter"].astype(str)
    )
    test_df["key"] = (
        test_df["cik"].astype(str)
        + "_"
        + test_df["year"].astype(str)
        + "_"
        + test_df["quarter"].astype(str)
    )
    full_financials_df["key"] = (
        full_financials_df["cik"].astype(str)
        + "_"
        + full_financials_df["year"].astype(str)
        + "_"
        + full_financials_df["quarter"].astype(str)
    )

    train_prev_data = get_last_financial_data(train_df, full_financials_df)
    val_prev_data = get_last_financial_data(val_df, full_financials_df)
    test_prev_data = get_last_financial_data(test_df, full_financials_df)

    logging.info(f"Calculating average sicagg level data...")
    average_industry_level_data = get_average_industry_level_feature(full_financials_df)
    train_sicagg = train_df[["sicagg", "year", "quarter"]]
    val_sicagg = val_df[["sicagg", "year", "quarter"]]
    test_sicagg = test_df[["sicagg", "year", "quarter"]]

    train_sicagg = train_sicagg.merge(
        average_industry_level_data, on=["sicagg", "year"], how="left"
    )
    val_sicagg = val_sicagg.merge(
        average_industry_level_data, on=["sicagg", "year"], how="left"
    )
    test_sicagg = test_sicagg.merge(
        average_industry_level_data, on=["sicagg", "year"], how="left"
    )

    logging.info(f"Calculating top 5 sicagg level data...")
    top5_industry_level_data = get_top5_industry_level_feature(full_financials_df)
    train_sicagg_top5 = train_df[["sicagg", "year"]]
    val_sicagg_top5 = val_df[["sicagg", "year"]]
    test_sicagg_top5 = test_df[["sicagg", "year"]]

    train_sicagg_top5 = train_sicagg_top5.merge(
        top5_industry_level_data, on=["sicagg", "year"], how="left"
    )
    val_sicagg_top5 = val_sicagg_top5.merge(
        top5_industry_level_data, on=["sicagg", "year"], how="left"
    )
    test_sicagg_top5 = test_sicagg_top5.merge(
        top5_industry_level_data, on=["sicagg", "year"], how="left"
    )

    logging.info(f"Computing rcma Features...")
    X_train = compute_rcma_group_ratios_features(
        X_train_pre_rcma, train_prev_data, train_sicagg, train_sicagg_top5
    )
    X_val = compute_rcma_group_ratios_features(
        X_val_pre_rcma, val_prev_data, val_sicagg, val_sicagg_top5
    )
    X_test = compute_rcma_group_ratios_features(
        X_test_pre_rcma, test_prev_data, test_sicagg, test_sicagg_top5
    )

    logging.info(f"Selected {len(feature_cols)} features for training")

    # Handle missing values
    for name, data in [
        ("train", X_train_pre_rcma),
        ("validation", X_val_pre_rcma),
        ("test", X_test_pre_rcma),
    ]:
        if data.isna().sum().sum() > 0:
            logging.warning(
                f"Found {data.isna().sum().sum()} missing values in {name} data. Filling with zeros."
            )
            data.fillna(0, inplace=True)

    return {
        "train": {
            "x_num": X_train,
            "x_mda": mda_content_train,
            "y": y_train,
        },
        "val": {
            "x_num": X_val,
            "x_mda": mda_content_val,
            "y": y_val,
        },
        "test": {
            "x_num": X_test,
            "x_mda": mda_content_test,
            "y": y_test,
        },
    }


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("Testing the RCMA dataset loading")
    dataset = load_rcma_dataset("company_isolated_splitting", fold_id=1)
    print("Train dataset shape:", dataset["train"]["x_num"].shape)
    print("Validation dataset shape:", dataset["val"]["x_num"].shape)
    print("Test dataset shape:", dataset["test"]["x_num"].shape)

    print("Train MDA content shape:", len(dataset["train"]["x_mda"]))
    print("Validation MDA content shape:", len(dataset["val"]["x_mda"]))
    print("Test MDA content shape:", len(dataset["test"]["x_mda"]))

    print("Train labels shape:", dataset["train"]["y"].shape)
    print("Validation labels shape:", dataset["val"]["y"].shape)
    print("Test labels shape:", dataset["test"]["y"].shape)
