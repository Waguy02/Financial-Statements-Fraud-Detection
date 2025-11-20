"""
SEC Financial Preprocessing without Data imputation using Random Forest (RF)
"""
import fcntl
import json
import logging
import multiprocessing
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyaml
import sklearn.neighbors._base
import tqdm
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed

from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_DECHOW,
    FINANCIALS_DIR_EXTENDED,
    MAX_CORE_USAGE,
    MAX_TAG_DEPTH,
    PREPROCESSED_PATH_EXTENDED,
    SEC_FILENAMES,
    SEC_FINANCIALS_RAW_DATASET_PATH,
    SEC_TAXONOMY,
    SEC_TAXONOMY_DATA_DIR,
    SEC_TAXONOMY_VERSION,
    SIC1_EXCLUDED,
)
from researchpkg.anomaly_detection.preprocessing.utils import (
    get_ith_label,
    get_sicagg,
    normalize_tlabel,
    save_dataset_config,
)
from researchpkg.industry_classification.utils.gaap_taxonomy_parser import (
    CalculationTree,
    CalculationTreeType,
)
from researchpkg.utils import configure_logger

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base

RUN_TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------------------------
# GROUP OF BASIC CONSTANTS
# ------------------------------------------------------------------------------
STMT_BS = "BS"
STMT_IS = "IS"
STMT_CF = "CF"
UOM_USD = "USD"
CRDR_CREDIT = "c"
QUARTER_RANGE = 4
SEC_FINANCIALS_QTR_FILENAME_PREFIX = "sec_financials_quarterly_"

# ------------------------------------------------------------------------------
# COLUMN NAME CONSTANTS
# ------------------------------------------------------------------------------
COL_COMPANY = "company"
COL_CIK = "cik"
COL_YEAR = "year"
COL_QUARTER = "quarter"
COL_PERIOD = "period"
COL_SICAGG = "sicagg"
COL_SIC = "sic"
COL_N_TAGS = "n_tags"
COL_N_TAGS_AUGMENTED = "n_tags_augmented"
COL_N_TAGS_TOTAL = "n_tags_total"
COL_N_IMPORTANT_TAGS = "n_important_tags"

COL_N_FEATURES = "n_extended_features"
COL_N_DECHOW_FEATURES = "n_dechow_features"

COL_FISCAL_YEAR_QUARTER = "fiscal_year_quarter"


# ------------------------------------------------------------------------------
# BALANCE SHEET TAG CONSTANTS
# ------------------------------------------------------------------------------
ACCOUNTS_PAYABLE_TAG = "AccountsPayableCurrentAndNoncurrent"
ASSETS_TAG = "Assets"
CASH_TAG = "CashCashEquivalentsAndShortTermInvestments"
INVENTORY_NET_TAG = "InventoryNet"
PROPERTY_PLANT_EQUIPMENT_NET_TAG = "PropertyPlantAndEquipmentNet"
INTANGIBLE_ASSETS_NET_TAG = "IntangibleAssetsNetIncludingGoodwill"
RETAINED_EARNINGS_TAG = "RetainedEarningsAccumulatedDeficit"
COMMON_STOCK_TAG = "CommonStockValue"
PREFERRED_STOCK_TAG = "PreferredStockValue"
GOODWILL_TAG = "Goodwill"
CURRENT_ASSETS_TAG = "AssetsCurrent"
CURRENT_LIABILITIES_TAG = "LiabilitiesCurrent"
TOTAL_LIABILITIES_TAG = "Liabilities"
SHORT_TERM_DEBT_TAG = "DebtCurrent"
ADDITIONAL_PAID_IN_CAPITAL_TAG = "AdditionalPaidInCapital"
AOCI_TAG = "AccumulatedOtherComprehensiveIncomeLossNetOfTax"
TREASURY_STOCK_TAG = "TreasuryStockValue"
TEMPORARY_EQUITY_TAG = (
    "TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests"
)
RECEIVABLE_FROM_SHAREHOLDERS_TAG = (
    "ReceivableFromShareholdersOrAffiliatesForIssuanceOfCapitalStock"
)
MINORITY_INTEREST_TAG = "MinorityInterest"
UNEARNED_ESOP_SHARES_TAG = "UnearnedESOPShares"
COMMON_STOCK_HELD_BY_SUBSIDIARY_TAG = "CommonStockHeldBySubsidiary"
LONG_TERM_DEBT_CURRENT_TAG = "LongTermDebtCurrent"
LONG_TERM_DEBT_NONCURRENT_TAG = "LongTermDebtNoncurrent"
ACCOUNT_RECEIVABLES_CURRENT_TAG = "AccountsReceivableNetCurrent"
ACCOUNT_RECEIVABLES_NON_CURRENT_TAG = "AccountsReceivableNetNoncurrent"
AMORTIZATION_OF_INTANGIBLES_ASSETS = "AmortizationOfIntangibleAssets"

SHORT_TERM_INVESTMENTS_TAG = "ShortTermInvestments"
LONG_TERM_INVESTMENTS_TAG = "LongTermInvestments"
TAXES_PAYABLE_TAG = "TaxesPablesCurrent"
PROCEEDS_FROM_LOANS_TAG = "ProceedsFromLoans"


# ------------------------------------------------------------------------------
# INCOME STATEMENT TAG CONSTANTS
# ------------------------------------------------------------------------------
REVENUES_TAG = "Revenues"
COST_OF_REVENUE_TAG = "CostOfRevenue"
OPERATING_INCOME_LOSS_TAG = "OperatingIncomeLoss"
INTEREST_EXPENSE_TAG = "InterestAndDebtExpense"
SELLING_GENERAL_ADMINISTRATIVE_EXPENSE_TAG = "SellingGeneralAndAdministrativeExpense"
NET_INCOME_LOSS_TAG = "NetIncomeLoss"
GROSS_PROFIT_TAG = "GrossProfit"
OPERATING_EXPENSES_TAG = "OperatingExpenses"
INCOME_LOSS_FROM_CONTINUING_OPERATIONS = "IncomeLossFromContinuingOperations"
DEFERRED_TAX_LIABILITIES_TAX_DEFERRED_INCOME = "DeferredTaxLiabilitiesTaxDeferredIncome"
# ------------------------------------------------------------------------------
# CASH FLOW STATEMENT TAG CONSTANTS
# ------------------------------------------------------------------------------
NET_CASH_FROM_OPERATIONS_TAG = "NetCashProvidedByUsedInOperatingActivities"
NET_CASH_FROM_FINANCING_TAG = "NetCashProvidedByUsedInFinancingActivities"
NET_CASH_FROM_INVESTING_TAG = "NetCashProvidedByUsedInInvestingActivities"
DEFERRED_TAX_ASSET_INCOME = "DeferredTaxAssetsDeferredIncome"
DEFERRED_TAX_LIABILITY_EXPENSE = "DeferredTaxLiabilitiesDeferredExpense"
DEPRECIATION_AND_AMORTIZATION_TAG = "DepreciationAndAmortization"


IMPORTANT_TAGS = [
    ACCOUNTS_PAYABLE_TAG,
    ASSETS_TAG,
    CASH_TAG,
    INVENTORY_NET_TAG,
    PROPERTY_PLANT_EQUIPMENT_NET_TAG,
    INTANGIBLE_ASSETS_NET_TAG,
    RETAINED_EARNINGS_TAG,
    COMMON_STOCK_TAG,
    PREFERRED_STOCK_TAG,
    GOODWILL_TAG,
    CURRENT_ASSETS_TAG,
    CURRENT_LIABILITIES_TAG,
    TOTAL_LIABILITIES_TAG,
    SHORT_TERM_DEBT_TAG,
    ADDITIONAL_PAID_IN_CAPITAL_TAG,
    AOCI_TAG,
    TREASURY_STOCK_TAG,
    TEMPORARY_EQUITY_TAG,
    RECEIVABLE_FROM_SHAREHOLDERS_TAG,
    MINORITY_INTEREST_TAG,
    UNEARNED_ESOP_SHARES_TAG,
    COMMON_STOCK_HELD_BY_SUBSIDIARY_TAG,
    LONG_TERM_DEBT_CURRENT_TAG,
    LONG_TERM_DEBT_NONCURRENT_TAG,
    ACCOUNT_RECEIVABLES_CURRENT_TAG,
    ACCOUNT_RECEIVABLES_NON_CURRENT_TAG,
    AMORTIZATION_OF_INTANGIBLES_ASSETS,
    REVENUES_TAG,
    COST_OF_REVENUE_TAG,
    OPERATING_INCOME_LOSS_TAG,
    INTEREST_EXPENSE_TAG,
    SELLING_GENERAL_ADMINISTRATIVE_EXPENSE_TAG,
    NET_INCOME_LOSS_TAG,
    GROSS_PROFIT_TAG,
    OPERATING_EXPENSES_TAG,
    INCOME_LOSS_FROM_CONTINUING_OPERATIONS,
    DEFERRED_TAX_LIABILITIES_TAX_DEFERRED_INCOME,
    NET_CASH_FROM_OPERATIONS_TAG,
    NET_CASH_FROM_FINANCING_TAG,
    NET_CASH_FROM_INVESTING_TAG,
    DEFERRED_TAX_ASSET_INCOME,
    DEFERRED_TAX_LIABILITY_EXPENSE,
    DEPRECIATION_AND_AMORTIZATION_TAG,
]

# ------------------------------------------------------------------------------
# AGGREGATE TAG CONSTANTS
# ------------------------------------------------------------------------------
LONG_TERM_DEBT_AGG = "agg_LONG_TERM_DEBT"
EQUITY_AGG = "agg_EQUITY"
TOTAL_DEBT_AGG = "agg_TOTAL_DEBT"
DEFERRED_TAX_EXPENSE_AGG = "agg_DEF_TAX_EXPENSE"
ACCRUALS_AGG = "agg_ACCRUALS"
EBIT_AGG = "agg_EBIT"
EBITDA_AGG = "agg_EBIDTA"
NET_CASH_FLOW_AGG = "agg_NET_CASH_FLOW"
ACCOUNT_RECEIVABLES_AGG = "agg_ACCOUNT_RECEIVABLES"

AGGREGATE_FEATURES = [
    LONG_TERM_DEBT_AGG,
    EQUITY_AGG,
    TOTAL_DEBT_AGG,
    DEFERRED_TAX_EXPENSE_AGG,
    ACCRUALS_AGG,
    EBIT_AGG,
    EBITDA_AGG,
    NET_CASH_FLOW_AGG,
    ACCOUNT_RECEIVABLES_AGG,
]


# DECHOW MODEL 1 Features
DECHOW_RSST_ACCRUALS = "DC_RSST_Accruals"
DECHOW_CH_REC = "DC_CH_REC"
DECHOW_CH_INV = "DC_CH_INV"
DECHOW_SOFT_ASSETS = "DC_SOFT_ASSETS"
DECHOW_CH_CASHSALES = "DC_CH_CASHSALES"
DECHOW_CH_ROA = "DC_CH_ROA"
DECHOW_ISSUANCE = "DC_ISSUANCE"


# INTERMEDIATE VARIABLES FOR DECHOW MODEL
DECHOW_DELTA_WC = "DC_DELTA_WC"
DECHOW_NCO = "DC_NCO"
DECHOW_FIN = "DC_FIN"

DECHOW_FEATURES = [
    DECHOW_RSST_ACCRUALS,
    DECHOW_CH_REC,
    DECHOW_CH_INV,
    DECHOW_SOFT_ASSETS,
    DECHOW_CH_CASHSALES,
    DECHOW_CH_ROA,
    DECHOW_ISSUANCE,
]


DECHOW_FEATURES_SHORT_DESCRIPTIONS = {
    DECHOW_RSST_ACCRUALS: "RSST Accrual: Measures discretionary accruals based on the Reverse-Salomon-Teoh (RSST) model",
    DECHOW_CH_REC: "Change in Accounts Receivable scaled by total assets",
    DECHOW_CH_INV: "Change in Inventory scaled by total assets",
    DECHOW_SOFT_ASSETS: "Ratio of Intangible Assets and Goodwill to Total Assets, indicating the proportion of 'soft' or less tangible assets",
    DECHOW_CH_CASHSALES: "Change in Cash Sales. Sales minus changes in accounts receivable.",
    DECHOW_CH_ROA: "Change in Return on Assets, indicating the trend in a company's profitability relative to its assets",
    DECHOW_ISSUANCE: "Net debt and equity issuance scaled by total assets, reflecting the extent of financing activities",
}


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return 0
    return numerator / denominator


def safe_sum(*args):
    if any(pd.isna(arg) or arg == 0 for arg in args):
        return 0
    return sum(args)


def native_sum(*args):
    """
    Custom sum function to handle NaN and zero values.
    Returns 0 if any argument is NaN or zero.
    """
    return sum(args)


def get_tag_value(tag, row, prev=False):
    prefix = "prev_" if prev else ""
    return row.get(f"{prefix}{tag}", 0)


def get_tag_avg_value(tag, row):
    prev_value = get_tag_value(tag, row, prev=True)
    current_value = get_tag_value(tag, row)

    if prev_value == 0 or current_value == 0:
        return 0
    return safe_sum(prev_value, current_value) / 2


def get_tag_diff_value(tag, row):
    prev_value = get_tag_value(tag, row, prev=True)
    current_value = get_tag_value(tag, row)

    if prev_value == 0 or current_value == 0:
        return 0
    if prev_value == current_value:
        return 0.000001  # to avoid zero because zero===nan

    return current_value - prev_value


def load_quarter_dataset(directory: Path) -> tuple[dict, dict]:
    dataset = {}
    dataset_name = directory.name
    dataset_info = {"year": dataset_name[:4], "quarter": dataset_name[4:]}

    for filename in SEC_FILENAMES:
        dataset_file = directory / f"{filename}.txt"
        if filename == "sub":
            dataset[filename] = pd.read_csv(
                dataset_file, sep="\t", low_memory=False, dtype={"sic": str}
            )
        else:
            dataset[filename] = pd.read_csv(dataset_file, sep="\t", low_memory=False)

    return dataset, dataset_info


def extract_quarter_dataset(directory: Path) -> pd.DataFrame:
    dataset, _ = load_quarter_dataset(directory)
    df_sub, df_tag, df_num, df_pre = (
        dataset["sub"],
        dataset["tag"],
        dataset["num"],
        dataset["pre"],
    )

    df_pre = df_pre.drop_duplicates(subset=["adsh", "tag", "stmt"])
    df_tag = df_tag[(df_tag.datatype == "monetary") & (df_tag.custom == 0)]
    df_tag = df_tag[["tag", "version", "datatype", "custom", "crdr", "tlabel"]]
    df_tag = df_tag[df_tag.version.str.contains("gaap")]
    df_tag.dropna(subset=["tag", "crdr", "datatype", "tlabel"], inplace=True)
    df_tag["value_sign"] = df_tag["crdr"].apply(
        lambda x: -1 if x.lower() == CRDR_CREDIT else 1
    )

    df_num = df_num.dropna(subset=["value", "adsh"])
    # only data without segments"
    df_num = df_num[df_num.segments.isna()]  # Only global data ie non segments"
    df_num = df_num.query(
        "(qtrs==1) or (qtrs==0)"
    )  # Only point in time of quarters data
    df_num = df_num.sort_values(by=["adsh", "tag", "ddate"]).drop_duplicates(
        subset=["adsh", "tag"], keep="last"
    )

    for i in range(1, QUARTER_RANGE + 1):
        df_tag[f"label{i}"] = df_tag.tlabel.apply(lambda x: get_ith_label(x, i))

    df_sub_min = df_sub[
        [
            "adsh",
            "cik",
            "name",
            "sic",
            "countryba",
            "bas1",
            "form",
            "fy",
            "fp",
            "period",
        ]
    ].fillna("0")
    df_sub_min["cik"] = df_sub_min["cik"].astype(str)
    df_sub_min["sic3"] = df_sub_min.sic.apply(lambda x: x[:3]).astype(int)
    df_sub_min["sic2"] = df_sub_min.sic.apply(lambda x: x[:2]).astype(int)
    df_sub_min["sic1"] = df_sub_min.sic.apply(lambda x: x[:1]).astype(int)
    df_sub_min["sicagg"] = df_sub_min.sic2.apply(lambda x: get_sicagg(x))
    df_sub_min["period"] = df_sub_min["period"].astype(int)
    df_sub_min.rename(columns={"name": "company"}, inplace=True)
    df_sub_min = df_sub_min[~df_sub_min.sic1.isin(SIC1_EXCLUDED)]

    df = pd.merge(df_num, df_pre, on=["adsh", "tag", "version"])
    df = pd.merge(df, df_tag, on=["tag", "version"])
    df = pd.merge(df, df_sub_min, on=["adsh"])

    df = df.sort_values(by=["adsh", "tag", "version"]).drop_duplicates(
        subset=["adsh", "tag"], keep="last"
    )

    df = df[df.stmt.isin([STMT_BS, STMT_IS, STMT_CF])]

    # update fp 'fy' to 'q4'
    df["fp"] = df["fp"].replace({"fy": "q4"})

    # Fixing fiscal years in datasets
    df["year"] = df["fy"].astype(int)
    df["quarter"] = df["fp"]
    df["quarter"] = df["quarter"].str.lower()

    df = df.query("(quarter in ['q1', 'q2', 'q3', 'q4','fy']) & (year!=0)")
    df["quarter"] = df["quarter"].replace({"fy": "q4"})

    df = df[df.uom == UOM_USD]
    df["tlabel"] = df.tlabel.apply(lambda t: normalize_tlabel(t))
    df["tag_depth"] = df.tlabel.apply(lambda t: len(t.split(",")))

    df["nline_bs"] = (
        df.query("(stmt == 'BS')&(value!=0)")
        .groupby("adsh")["tag"]
        .transform("nunique")
    )

    return df


def process_quarter_dataset(directory: Path):
    dataset = extract_quarter_dataset(directory)
    dataset = dataset.sort_values(["adsh", "tag"])[
        [
            "company",
            "cik",
            "year",
            "quarter",
            "period",
            "sicagg",
            "sic",
            "tag",
            "value",
            "crdr",
        ]
    ]

    if not FINANCIALS_DIR_DECHOW.exists():
        FINANCIALS_DIR_DECHOW.mkdir()
    dataset.to_csv(
        FINANCIALS_DIR_DECHOW
        / f"{SEC_FINANCIALS_QTR_FILENAME_PREFIX}{directory.name}_dechow.csv",
        index=False,
    )


def extract_financial_data(root_dir: Path, start_year: int, end_year: int) -> None:
    all_directories = list(
        [directory for directory in root_dir.glob("*") if directory.is_dir()]
    )

    all_directories = list(
        filter(
            lambda d: start_year <= int(d.name[:4]) <= end_year,
            all_directories,
        )
    )

    logging.info(f"{len(all_directories)} datasets to process")

    njobs = min(MAX_CORE_USAGE, multiprocessing.cpu_count())
    njobs = min(njobs, len(all_directories))

    if not (FINANCIALS_DIR_DECHOW).exists():
        (FINANCIALS_DIR_DECHOW).mkdir(parents=True)
    tags_index_csvfile = FINANCIALS_DIR_DECHOW / "sec_tags_index_dechow.csv"
    tags_index_csvfile_versionned = (
        FINANCIALS_DIR_DECHOW / "sec_tags_index_versionned_dechow.csv"
    )
    tags_index = CalculationTree.get_full_tag_index_df(max_level=MAX_TAG_DEPTH)
    tags_index["tlabel"] = tags_index["tag"].apply(normalize_tlabel)
    tags_index.sort_values(["stmt", "depth"], inplace=True)

    tags_index.to_csv(tags_index_csvfile_versionned, index=False)

    tags_index.drop_duplicates(
        subset=[
            "tag",
        ]
    ).to_csv(tags_index_csvfile, index=False)

    Parallel(n_jobs=njobs)(
        delayed(process_quarter_dataset)(directory)
        for directory in tqdm.tqdm(all_directories, "Extracting dataset")
    )


def impute_financial_data(root_dir: Path, start_year, end_year):
    all_directories = [
        directory
        for directory in root_dir.glob("*")
        if directory.is_dir() and (start_year <= int(directory.stem[:4]) <= end_year)
    ]

    bs_tree = CalculationTree.build_taxonomy_tree(
        SEC_TAXONOMY_DATA_DIR,
        SEC_TAXONOMY,
        SEC_TAXONOMY_VERSION,
        type=CalculationTreeType.BALANCE_SHEET,
    )
    is_tree = CalculationTree.build_taxonomy_tree(
        SEC_TAXONOMY_DATA_DIR,
        SEC_TAXONOMY,
        SEC_TAXONOMY_VERSION,
        type=CalculationTreeType.INCOME_STATEMENT,
    )
    cf_tree = CalculationTree.build_taxonomy_tree(
        SEC_TAXONOMY_DATA_DIR,
        SEC_TAXONOMY,
        SEC_TAXONOMY_VERSION,
        type=CalculationTreeType.CASH_FLOW_STATEMENT,
    )

    tags_index_csvfile = FINANCIALS_DIR_DECHOW / "sec_tags_index_dechow.csv"
    tags_index = pd.read_csv(tags_index_csvfile)
    tag_columns = tags_index["tag"].tolist()
    tags_set = set(tag_columns)

    def taxonomy_infer(directory):
        configure_logger(
            Path(f"sec_financial_preprocessing_dechow_{RUN_TIMESTAMP}.log"),
            logging.INFO,
        )
        year_quarter_name = directory.name
        partial_csv_file = (
            FINANCIALS_DIR_DECHOW
            / f"sec_financials_quarterly_{year_quarter_name}_dechow.csv"
        )

        partial_df = pd.read_csv(partial_csv_file)
        partial_df = pivot_financial_data(partial_df)

        missing_tags = list(tags_set - set(partial_df.columns.tolist()))
        logging.info(f"Number of missing tags : {len(missing_tags)}")
        partial_df.loc[:, missing_tags] = 0

        initial_size = len(partial_df)
        partial_df = compute_missing_financial_data(
            partial_df, tags_index, bs_tree, is_tree, cf_tree
        )

        final_size = len(partial_df)

        if len(partial_df) > 0:
            partial_df.to_csv(partial_csv_file, index=False)
            logging.info(
                f"Computing missing data for {year_quarter_name} done. "
                f"Removed {initial_size - final_size} of {initial_size} rows with too many missing values"
            )
            return partial_df[tag_columns].values
        else:
            logging.info(f"All rows removed for {year_quarter_name}. File is empty")
            partial_csv_file.unlink()
            return None

    njobs = min(MAX_CORE_USAGE, multiprocessing.cpu_count())
    processed_data_list = Parallel(n_jobs=njobs)(
        delayed(taxonomy_infer)(directory)
        for directory in tqdm.tqdm(all_directories, "Taxonomy infer data")
    )

    processed_data_list = [data for data in processed_data_list if data is not None]


def merge_financial_data(root_dir: Path, start_year, end_year):

    all_directories = list(
        [
            directory
            for directory in root_dir.glob("*")
            if directory.is_dir()
            and (start_year <= int(directory.stem[:4]) <= end_year)
        ]
    )
    all_directories = list(set(all_directories))
    dataset_csvfile = FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv"

    tags_index_csvfile = FINANCIALS_DIR_DECHOW / "sec_tags_index_dechow.csv"
    tags_index = pd.read_csv(tags_index_csvfile)
    dechow_features_index = DECHOW_FEATURES

    columns_order = (
        [
            COL_COMPANY,
            COL_CIK,
            COL_YEAR,
            COL_QUARTER,
            COL_PERIOD,
            COL_SICAGG,
            COL_SIC,
            COL_N_DECHOW_FEATURES,
        ]
        + list(sorted(tags_index.tag.unique().tolist()))
        + dechow_features_index
    )

    if dataset_csvfile.exists():
        dataset_csvfile.unlink()

    def merge_single_directory(directory):
        tqdm.tqdm.pandas()
        configure_logger(
            Path(f"sec_financial_preprocessing_dechow_{RUN_TIMESTAMP}.log"),
            logging.INFO,
        )
        extended_index_file = (
            FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly_no_financials.csv"
        )
        logging.info(f"Loading extended index from {extended_index_file}")
        extended_index = pd.read_csv(
            extended_index_file, usecols=[COL_CIK, COL_YEAR, COL_QUARTER]
        )

        year_quarter_name = directory.name

        year = int(year_quarter_name[:4])
        quarter = int(year_quarter_name[-1])

        if quarter == 1:
            prev_year_quarter_name = f"{year-1}q{4}"
        else:
            prev_year_quarter_name = f"{year}q{quarter-1}"

        partial_csv_file = (
            FINANCIALS_DIR_DECHOW
            / f"sec_financials_quarterly_{year_quarter_name}_dechow.csv"
        )

        prev_partial_csv_file = (
            FINANCIALS_DIR_DECHOW
            / f"sec_financials_quarterly_{prev_year_quarter_name}_dechow.csv"
        )
        if not partial_csv_file.exists():
            logging.warning(f"Financial data not found for directory: {directory}")
            return

        if not prev_partial_csv_file.exists():
            logging.info(
                f"Previous quarter of {year_quarter_name} data does not exists."
                f"{prev_partial_csv_file} missing. Skipping"
            )
            return

        partial_df = pd.read_csv(partial_csv_file)

        # Load previous quarter data and update quarter and year fields
        prev_partial_df = pd.read_csv(prev_partial_csv_file)

        prev_partial_df.drop(columns=[COL_YEAR, COL_QUARTER, COL_PERIOD], inplace=True)

        # reformat back prev data
        prev_partial_df = prev_partial_df.merge(
            partial_df[[COL_CIK, COL_YEAR, COL_QUARTER, COL_PERIOD]], on=COL_CIK
        )

        if len(partial_df) == 0:
            logging.info(
                f"Empty data for quarter {year_quarter_name} missing. SKipping."
            )
            return
        if len(prev_partial_df) == 0:
            logging.info(
                f"Empty data for prev quarter of {year_quarter_name} missing. Skipping"
            )
            return

        partial_df = partial_df[partial_df.cik.isin(prev_partial_df.cik.unique())]
        prev_partial_df = prev_partial_df[
            prev_partial_df.cik.isin(partial_df.cik.unique())
        ]

        partial_df = compute_aggregate_financial_features(partial_df)
        prev_partial_df = compute_aggregate_financial_features(prev_partial_df)

        # remove existing prev
        prev_partial_df = prev_partial_df.rename(
            columns={
                col: f"prev_{col}"
                for col in prev_partial_df.columns
                if col
                not in [
                    COL_COMPANY,
                    COL_CIK,
                    COL_YEAR,
                    COL_QUARTER,
                    COL_PERIOD,
                    COL_SICAGG,
                    COL_SIC,
                ]
            }
        )
        partial_df = pd.merge(
            partial_df,
            prev_partial_df,
            on=[
                COL_COMPANY,
                COL_CIK,
                COL_YEAR,
                COL_QUARTER,
                COL_PERIOD,
                COL_SICAGG,
                COL_SIC,
            ],
        )
        partial_df = partial_df.fillna(0)

        # Remove rows that does not match the extended index
        partial_df = partial_df.merge(
            extended_index, on=[COL_CIK, COL_YEAR, COL_QUARTER]
        )

        partial_df = compute_dechow_features(partial_df)
        partial_df = partial_df[columns_order]

        for col in [COL_PERIOD, COL_YEAR]:
            partial_df[col] = partial_df[col].astype(int)

        if not dataset_csvfile.exists():
            partial_df.to_csv(dataset_csvfile, header=True, index=False)
        else:
            with open(dataset_csvfile, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                partial_df.to_csv(f, header=False, index=False, mode="a")
                fcntl.flock(f, fcntl.LOCK_UN)

    logging.info("Merging quarterly datasets")
    Parallel(n_jobs=MAX_CORE_USAGE, prefer="processes")(
        delayed(merge_single_directory)(directory)
        for directory in tqdm.tqdm(all_directories, "Merging quarterly datasets")
    )

    logging.info("Merging partial datasets completed.")


def pivot_financial_data(data: pd.DataFrame) -> pd.DataFrame:
    pivoted_df = data.pivot_table(
        index=["company", "cik", "year", "quarter", "period", "sicagg", "sic"],
        columns="tag",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted_df.columns = [
        col if isinstance(col, str) else "_".join(map(str, col))
        for col in pivoted_df.columns
    ]
    pivoted_df = pivoted_df.copy()
    pivoted_df[COL_N_TAGS] = pivoted_df.iloc[:, 5:].notnull().sum(axis=1)
    return pivoted_df


def compute_missing_financial_data(
    data: pd.DataFrame,
    tag_index: pd.DataFrame,
    bs_tree: CalculationTree,
    is_tree: CalculationTree,
    cf_tree: CalculationTree,
) -> pd.DataFrame:
    header_columns = [
        "company",
        "cik",
        "year",
        "quarter",
        "period",
        "sicagg",
        "sic",
        COL_N_TAGS,
    ]
    all_tags = set(tag_index.tag.unique().tolist())

    def compute_missing_data_single_row(row: pd.Series) -> pd.Series:
        initial_row_values = row.to_dict()
        new_row_values = {
            k: v
            for k, v in initial_row_values.items()
            if k in all_tags and v != 0 and v != None
        }

        initial_size = len(new_row_values)
        new_row_values = bs_tree.compute_missing_values(
            new_row_values, max_depth=MAX_TAG_DEPTH
        )

        new_row_values = is_tree.compute_missing_values(
            new_row_values, max_depth=MAX_TAG_DEPTH
        )

        new_row_values = cf_tree.compute_missing_values(
            new_row_values, max_depth=MAX_TAG_DEPTH
        )

        new_row_values = {k: v for k, v in new_row_values.items() if k in all_tags}

        new_row_values[COL_N_TAGS_AUGMENTED] = len(new_row_values) - initial_size

        new_row_values.update({k: initial_row_values[k] for k in header_columns})

        for k in all_tags:
            if k not in new_row_values:
                new_row_values[k] = 0.0

        return pd.Series(new_row_values)

    data.fillna(0.0, inplace=True)
    logging.info(" Calculate missing financial data using taxonomy tree")
    tqdm.tqdm.pandas()
    updated_data = data.progress_apply(compute_missing_data_single_row, axis=1)
    if len(updated_data) > 0:
        logging.info(
            f"  {updated_data['n_tags_augmented'].mean()} tags augmented in average"
        )
    else:
        updated_data[COL_N_TAGS_AUGMENTED] = []
    return updated_data


def compute_aggregate_financial_features(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing aggregate financial features")

    def calculate_total_equity(row):
        return safe_sum(
            abs(get_tag_value(COMMON_STOCK_TAG, row)),
            abs(get_tag_value(PREFERRED_STOCK_TAG, row)),
            get_tag_value(ADDITIONAL_PAID_IN_CAPITAL_TAG, row),
            get_tag_value(RETAINED_EARNINGS_TAG, row),
            get_tag_value(AOCI_TAG, row),
            -get_tag_value(TREASURY_STOCK_TAG, row),
            -get_tag_value(TEMPORARY_EQUITY_TAG, row),
            -get_tag_value(RECEIVABLE_FROM_SHAREHOLDERS_TAG, row),
            -get_tag_value(MINORITY_INTEREST_TAG, row),
            get_tag_value(UNEARNED_ESOP_SHARES_TAG, row),
            get_tag_value(COMMON_STOCK_HELD_BY_SUBSIDIARY_TAG, row),
        )

    def calculate_total_debt(row):
        return safe_sum(
            get_tag_value(SHORT_TERM_DEBT_TAG, row),
            get_tag_value(LONG_TERM_DEBT_AGG, row),
        )

    def calculate_deferred_tax_expense(row):
        return safe_sum(
            get_tag_value(DEFERRED_TAX_LIABILITY_EXPENSE, row),
            -get_tag_value(DEFERRED_TAX_ASSET_INCOME, row),
        )

    def calculate_accruals(row):
        return safe_sum(
            get_tag_value(NET_INCOME_LOSS_TAG, row),
            -get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
        )

    def calculate_ebit(row):
        return safe_sum(
            get_tag_value(REVENUES_TAG, row),
            -get_tag_value(COST_OF_REVENUE_TAG, row),
            -get_tag_value(OPERATING_EXPENSES_TAG, row),
        )

    def calculate_ebitda(row):
        return safe_sum(
            calculate_ebit(row),
            get_tag_value(DEPRECIATION_AND_AMORTIZATION_TAG, row),
        )

    def calculate_net_cash_flow(row):
        return safe_sum(
            get_tag_value(NET_CASH_FROM_FINANCING_TAG, row),
            get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
            get_tag_value(NET_CASH_FROM_INVESTING_TAG, row),
        )

    def calculate_accounts_receivables(row):
        return safe_sum(
            get_tag_value(ACCOUNT_RECEIVABLES_CURRENT_TAG, row),
            get_tag_value(ACCOUNT_RECEIVABLES_NON_CURRENT_TAG, row),
        )

    data[EQUITY_AGG] = data.apply(calculate_total_equity, axis=1)
    data[LONG_TERM_DEBT_AGG] = data.apply(
        lambda row: safe_sum(
            get_tag_value(LONG_TERM_DEBT_CURRENT_TAG, row),
            get_tag_value(LONG_TERM_DEBT_NONCURRENT_TAG, row),
        ),
        axis=1,
    )
    data[TOTAL_DEBT_AGG] = data.apply(calculate_total_debt, axis=1)
    data[DEFERRED_TAX_EXPENSE_AGG] = data.apply(calculate_deferred_tax_expense, axis=1)
    data[ACCRUALS_AGG] = data.apply(calculate_accruals, axis=1)
    data[EBIT_AGG] = data.apply(calculate_ebit, axis=1)
    data[EBITDA_AGG] = data.apply(calculate_ebitda, axis=1)
    data[NET_CASH_FLOW_AGG] = data.apply(calculate_net_cash_flow, axis=1)
    data[ACCOUNT_RECEIVABLES_AGG] = data.apply(calculate_accounts_receivables, axis=1)

    return data


def compute_dechow_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dechow features.
    DECHOW_RSST_ACCRUALS = "DC_RSST_Accruals"
    DECHOW_CH_REC = "DC_CH_REC"
    DECHOW_CH_INV = "DC_CH_INV"
    DECHOW_SOFT_ASSETS  = "DC_SOFT_ASSETS"
    DECHOW_CH_CASHSALES = "DC_CH_CASHSALES"
    DECHOW_CH_ROA = "DC_CH_ROA"
    DECHOW_ISSUANCE = "DC_ISSUANCE

    Predicted Value=−7.893
                +0.790×RSST Accruals
                +2.518×Change in Receivables
                +1.191×Change in Inventory
                +1.979×Soft Assets
                +0.171×Change in Cash Sales
                −0.932×Change in ROA+1.029×Actual Issuance

    """
    logging.info("Computing Dechow features")

    def calculate_dechow_features(row):

        # WC_accruals  =  [[ΔCurrent Assets (DATA 4) – ΔCash and Short-term Investments (DATA 1)]– [ΔCurrent Liabilities (DATA 5) ΔDebt in Current Liabilities (DATA 34) – ΔTaxes Payable (DATA 71)] /Average total assets

        wc = native_sum(
            get_tag_value(CURRENT_ASSETS_TAG, row),
            -get_tag_value(CASH_TAG, row),
            -get_tag_value(SHORT_TERM_INVESTMENTS_TAG, row),
            -get_tag_value(CURRENT_LIABILITIES_TAG, row),
            +get_tag_value(SHORT_TERM_DEBT_TAG, row),
        )

        wc_prev = native_sum(
            get_tag_value(CURRENT_ASSETS_TAG, row, prev=True),
            -get_tag_value(CASH_TAG, row, prev=True),
            -get_tag_value(SHORT_TERM_INVESTMENTS_TAG, row, prev=True),
            +get_tag_value(CURRENT_LIABILITIES_TAG, row, prev=True),
        )

        delta_dwc = native_sum(wc, -wc_prev)

        # delta_NCO
        # NCO = [Total Assets (DATA 6) Current Assets (DATA 4) - Investments and Advances (DATA 32)] – [Total Liabilities (DATA 181) – Current Liabilities (DATA 5) – Long-term Debt (DATA 9)]
        nco = native_sum(
            get_tag_value(ASSETS_TAG, row),
            -get_tag_value(CURRENT_ASSETS_TAG, row),
            -get_tag_value(SHORT_TERM_INVESTMENTS_TAG, row),
            -get_tag_value(LONG_TERM_INVESTMENTS_TAG, row),
            -get_tag_value(TOTAL_LIABILITIES_TAG, row),
            get_tag_value(CURRENT_LIABILITIES_TAG, row),
            get_tag_value(LONG_TERM_DEBT_AGG, row),
        )

        nco_prev = native_sum(
            get_tag_value(ASSETS_TAG, row, prev=True),
            -get_tag_value(CURRENT_ASSETS_TAG, row, prev=True),
            -get_tag_value(SHORT_TERM_INVESTMENTS_TAG, row, prev=True),
            -get_tag_value(LONG_TERM_INVESTMENTS_TAG, row, prev=True),
            -get_tag_value(TOTAL_LIABILITIES_TAG, row, prev=True),
            get_tag_value(CURRENT_LIABILITIES_TAG, row, prev=True),
            get_tag_value(LONG_TERM_DEBT_AGG, row, prev=True),
        )
        delta_nco = native_sum(nco, -nco_prev)

        # delta_FIN
        # FIN = [Short-term Investments (DATA 193) + Long-term Investments (DATA 32)] – [Long-term Debt (DATA 9) + Debt in Current Liabilities (DATA 34) + Preferred Stock (DATA 130)];
        fin = native_sum(
            get_tag_value(SHORT_TERM_INVESTMENTS_TAG, row),
            get_tag_value(LONG_TERM_INVESTMENTS_TAG, row),
            -get_tag_value(LONG_TERM_DEBT_AGG, row),
            -get_tag_value(SHORT_TERM_DEBT_TAG, row),
            -get_tag_value(PREFERRED_STOCK_TAG, row),
        )

        fin_prev = native_sum(
            get_tag_value(SHORT_TERM_INVESTMENTS_TAG, row, prev=True),
            get_tag_value(LONG_TERM_INVESTMENTS_TAG, row, prev=True),
            -get_tag_value(LONG_TERM_DEBT_AGG, row, prev=True),
            -get_tag_value(SHORT_TERM_DEBT_TAG, row, prev=True),
            -get_tag_value(PREFERRED_STOCK_TAG, row, prev=True),
        )
        delta_fin = native_sum(fin, -fin_prev)

        # rsst_accruals =  (∆WC + ∆NCO + ∆FIN)/Average total assets
        rsst_accruals = safe_divide(
            native_sum(delta_dwc, delta_nco, delta_fin),
            get_tag_avg_value(ASSETS_TAG, row),
        )

        # Changes_in_receivables : Delta_Receivables/ Average Assets
        ch_rec = safe_divide(
            get_tag_diff_value(ACCOUNT_RECEIVABLES_AGG, row),
            get_tag_avg_value(ASSETS_TAG, row),
        )

        # Changes_in_inventories : Delta_Inventory/ Average Assets
        ch_inv = safe_divide(
            get_tag_diff_value(INVENTORY_NET_TAG, row),
            get_tag_avg_value(ASSETS_TAG, row),
        )

        # Soft_assets = (Total assets (DATA 6) - PP&E (DATA 8) – Cash and cash  equivalent (DATA 1))/Total assets (DATA 6)
        soft_assets = safe_divide(
            native_sum(
                get_tag_value(ASSETS_TAG, row),
                -get_tag_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row),
                -get_tag_value(CASH_TAG, row),
            ),
            get_tag_value(ASSETS_TAG, row),
        )

        # Changes_in_cash_sales = Percentage change in cash sales [Sales(DATA 12)-∆Accounts  Receivables (DATA 2)]
        ch_cash_sales = native_sum(
            get_tag_value(REVENUES_TAG, row),
            -get_tag_diff_value(ACCOUNT_RECEIVABLES_AGG, row),
        )

        # [Earningst (DATA 18)/Average total assetst ] - [Earningst-1 (DATA 18)/Average total assetst-1]
        ch_roa = native_sum(
            safe_divide(
                get_tag_value(NET_INCOME_LOSS_TAG, row),
                get_tag_avg_value(ASSETS_TAG, row),
            ),
            -safe_divide(
                get_tag_value(NET_INCOME_LOSS_TAG, row, prev=True),
                get_tag_value(ASSETS_TAG, row, prev=True),
            ),
        )

        # Actual issuance : Indicates whether the company issued new shares or new debt in the current period.
        # Check for increase in common stock, preferred stock, or additional paid-in capital
        equity_issuance = (
            (get_tag_diff_value(COMMON_STOCK_TAG, row) > 0)
            or (get_tag_diff_value(PREFERRED_STOCK_TAG, row) > 0)
            or (get_tag_diff_value(ADDITIONAL_PAID_IN_CAPITAL_TAG, row) > 0)
        )

        # Check for increase in total debt or long-term debt specifically
        debt_issuance = (
            (get_tag_diff_value(TOTAL_DEBT_AGG, row) > 0)
            or (get_tag_diff_value(LONG_TERM_DEBT_AGG, row) > 0)
            or (get_tag_value(PROCEEDS_FROM_LOANS_TAG, row) > 0)
        )  # Use the explicitly provided tag

        issuance = 1 if equity_issuance or debt_issuance else 0

        return pd.Series(
            {
                DECHOW_RSST_ACCRUALS: rsst_accruals,
                DECHOW_CH_REC: ch_rec,
                DECHOW_CH_INV: ch_inv,
                DECHOW_SOFT_ASSETS: soft_assets,
                DECHOW_CH_CASHSALES: ch_cash_sales,
                DECHOW_CH_ROA: ch_roa,
                DECHOW_ISSUANCE: issuance,
            }
        )

    tqdm.tqdm.pandas()
    data = data.fillna(0)
    data[
        [
            DECHOW_RSST_ACCRUALS,
            DECHOW_CH_REC,
            DECHOW_CH_INV,
            DECHOW_SOFT_ASSETS,
            DECHOW_CH_CASHSALES,
            DECHOW_CH_ROA,
            DECHOW_ISSUANCE,
        ]
    ] = data.progress_apply(calculate_dechow_features, axis=1)
    data[COL_N_DECHOW_FEATURES] = data[DECHOW_FEATURES].ne(0).sum(axis=1)
    return data


def clean_dataset_files(root_dir):
    """
    Deleted all the partial files
    """
    all_directories = list(
        [directory for directory in root_dir.glob("*") if directory.is_dir()]
    )
    logging.info("Cleaning Partial CSV files")
    for directory in all_directories:
        year_quarter_name = directory.name
        partial_csv_file = (
            FINANCIALS_DIR_DECHOW
            / f"sec_financials_quarterly_{year_quarter_name}_dechow.csv"
        )
        if partial_csv_file.exists():
            partial_csv_file.unlink()


def save_dataset_stats():
    logging.info("Saving dataset stats")

    main_columns = [
        COL_COMPANY,
        COL_CIK,
        COL_YEAR,
        COL_QUARTER,
        COL_PERIOD,
        COL_SICAGG,
        COL_SIC,
        COL_N_DECHOW_FEATURES,
    ]

    dataset_csvfile = FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_dechow.csv"
    main_df = pd.read_csv(dataset_csvfile, usecols=main_columns)

    # Get the nubmer of tags for the tags index
    tags_index_csvfile = FINANCIALS_DIR_DECHOW / "sec_tags_index_dechow.csv"
    tags_index = pd.read_csv(tags_index_csvfile)

    def get_col_stat(col_name):
        return {
            "avg": main_df[col_name].mean().item(),
            "min": main_df[col_name].min().item(),
            "max": main_df[col_name].max().item(),
            "std": main_df[col_name].std().item(),
            "median": main_df[col_name].median().item(),
        }

    stats_dict = {
        "counts": {
            "n_reports": main_df[["cik", "year", "quarter"]].drop_duplicates().shape[0],
            "n_companies": main_df["cik"].nunique(),
        },
        "summary_sic_agg": main_df["sicagg"].value_counts().to_dict(),
        "summary_sic": main_df["sic"].value_counts().to_dict(),
        "n_dechow_features": len(DECHOW_FEATURES),
        "dechow_count_stats": get_col_stat(COL_N_DECHOW_FEATURES),
    }
    save_dataset_config(str(FINANCIALS_DIR_DECHOW), **stats_dict)

    main_df.to_csv(
        FINANCIALS_DIR_DECHOW / "sec_financials_quarterly_no_financials_dechow.csv"
    )


if __name__ == "__main__":

    configure_logger(
        Path(f"sec_financial_preprocessing_dechow_{RUN_TIMESTAMP}.log"), logging.INFO
    )
    begin = datetime.now()
    start_year = 2009
    end_year = 2024
    extract_financial_data(Path(SEC_FINANCIALS_RAW_DATASET_PATH), start_year, end_year)
    impute_financial_data(Path(SEC_FINANCIALS_RAW_DATASET_PATH), start_year, end_year)
    merge_financial_data(Path(SEC_FINANCIALS_RAW_DATASET_PATH), start_year, end_year)
    clean_dataset_files(Path(SEC_FINANCIALS_RAW_DATASET_PATH))
    save_dataset_stats()
    logging.info("Preprocessing completed")
    duration = datetime.now() - begin
    logging.info(f"Process duration:{duration}")
