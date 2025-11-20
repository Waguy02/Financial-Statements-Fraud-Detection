import json
import logging
import os
import re
from typing import Any, Dict, List

import pyaml

from researchpkg.anomaly_detection.config import VLLM_ENDPOINTS_CONFIG_PATH

# ------------------------------------------------------------------------------
# 1) CONSTANTS & DEFINITIONS
# ------------------------------------------------------------------------------
DEFAULT_SIC_AGG = "0"
# SIC Code Groups
SIC_10 = {"10", "11", "12", "13", "14"}
SIC_15 = {"15", "16", "17"}
SIC_20 = {
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
}
SIC_40 = {"40", "41", "42", "43", "44", "45", "46", "47", "48", "49"}
SIC_50 = {"50", "51"}
SIC_52 = {"52", "53", "54", "55", "56", "57", "58", "59"}
SIC_60 = {"60", "61", "62", "63", "64", "65", "67"}
SIC_70 = {
    "70",
    "72",
    "73",
    "75",
    "76",
    "78",
    "79",
    "80",
    "81",
    "82",
    "83",
    "84",
    "85",
    "86",
    "87",
    "89",
}
SIC_90 = {"90", "91", "92", "93", "94", "95", "96", "97", "98", "99"}


def normalize_tlabel(tlabel):
    """
    Normalize a tlabel by lowercasing and replacing "and" by ",".

    :param tlabel: A tlabel string
    :return: A normalized tlabel string
    """

    return tlabel.lower().replace(" and ", ", ")


def get_ith_label(tlabel, i):
    """
    Get the ith label of a tlabel.

    :param tlabel: A tlabel string
    :param i: The index of the label to extract
    """
    all_labels = tlabel.split(",")
    if i > len(all_labels):
        return ""
    else:
        return all_labels[i - 1]


def normalize_cik_code(cik: str) -> str:
    """
    Normalize the length of CIK to 10 characters
    """

    return "0" * (10 - len(cik)) + cik


def get_ratios_index():
    return [
        "ratio_GrossProfitMargin",
        "ratio_OperatingMargin",
        "ratio_NetProfitMargin",
        "ratio_EBITMargin",
        "ratio_EBITDAMargin",
        "ratio_CashFlowMargin",
        "ratio_ReturnOnAssets",
        "ratio_ReturnOnEquity",
        "ratio_CurrentRatio",
        "ratio_QuickRatio",
        "ratio_CashRatio",
        "ratio_WorkingCapitalToTotalAssets",
        "ratio_DebtToAssetsRatio",
        "ratio_DebtToEquityRatio",
        "ratio_InterestCoverageRatio",
        "ratio_TotalLiabilitiesToAssets",
        "ratio_AssetTurnover",
        "ratio_FixedAssetTurnover",
        "ratio_ReceivablesTurnover",
        "ratio_InventoryTurnover",
        "ratio_SalesTurnover",
        "ratio_EquityMultiplier",
        "ratio_SGARatio",
        "ratio_GoodwilltoAssets",
        "ratio_CashFlowToDebtRatio",
        "ratio_CashFlowFinancingActivities",
        "ratio_CashFlowOperatingActivities",
        "ratio_EquityRatio",
        "ratio_OperatingCashFlowToCurrentLiabilities",
        "ratio_CashFlowToRevenue",
        "ratio_CashFlowCoverageRatio",
        "ratio_NetWorkingCapital",
        "ratio_LongTermDebtToEquity",
        "ratio_DegreeOfFinancialLeverage",
        "ratio_InvestedCapitalRatio",
        "ratio_CashToTotalAsset",
        "ratio_DebtServiceCoverage",
        "ratio_FinancialLeverageIndex",
        "ratio_TimesInterestEarnedRatio",
        "ratio_CurrentAssetToRevenues",
        "ratio_CurrentLiabilitiesToRevenues",
        "ratio_ShortTermDebtToRevenue",
        "ratio_IntangibleAssetToRevenue",
        "ratio_LongtermLeverage",
        "ratio_CFF",
    ]


def get_aggregates_index():
    return [
        "agg_EQUITY",
        "agg_LONG_TERM_DEBT",
        "agg_TOTAL_DEBT",
        "agg_DEF_TAX_EXPENSE",
        "agg_ACCRUALS",
        "agg_EBIT",
        "agg_EBIDTA",
        "agg_NET_CASH_FLOW",
        "agg_ACCOUNT_RECEIVABLES",
    ]


def get_differential_features_index():
    return [
        "diff_WC_Accruals",
        "diff_Inventories",
        "diff_Receivables",
        "diff_CashSales",
        "diff_CashMargin",
        "diff_DefTaxExpense",
        "diff_Earnings",
        "diff_AverageAssets",
    ]


def get_benish_features_index():
    return [
        "Beneish_PROBM",
        "Beneish_ACCRUALS",
        "Beneish_DSR",
        "Beneish_GMI",
        "Beneish_AQI",
        "Beneish_SGI",
        "Beneish_DEPI",
        "Beneish_SGAI",
        "Beneish_LEVI",
    ]


def save_dataset_config(
    dataset_dir,
    filename="dataset_config",
    **kwargs,
):
    """
    Save the config of the dataset in a  yaml file.

    :param global_exp_dir:Root directory of the  dataset
    :param min_year: Min year of the dataset
    :param max_year: Max year of the dataset
    :param sic1_used: List of sic1 used (not excluded)
    :param nb_cik: Number of companies in the dataset
    :param max_sub_per_cik: Max number of submission per cik(company)
    :param top_k_tags: Number of top tags to consider
    :param tags_persheet_count_threshold: Threshold of number of appearance of tags to consider
    :param dataset_size: Size of the dataset
    :param train_dataset_size: Size of the train dataset
    :param val_dataset_size: Size of the validation dataset
    """
    config = kwargs
    config["global_exp_dir"] = dataset_dir
    config_file = os.path.join(dataset_dir, f"{filename}.yaml")
    import pyaml

    with open(config_file, "w") as f:
        pyaml.yaml.dump(config, f, default_flow_style=None, indent=4)


def clean_mda_content(mda_content, clean_small_lines=False):
    mda_content = re.sub("&#[0-9a-fA-F]+;", "", mda_content)  # Remove HTML entities

    # Remove too short paragraphs (less than 30 characters)
    if clean_small_lines:
        mda_content = re.sub(r"(?m)^[^\n]{0,30}\n", "", mda_content)

    # Remove all empty lines
    mda_content = re.sub(r"\n{2,}", "\n", mda_content)

    # Remove too long spaces
    mda_content = re.sub(r"\s+", " ", mda_content).strip()

    return mda_content


def get_sicagg(sic2: str) -> str:
    sic2 = str(sic2)
    if sic2.startswith("0"):
        return "0"
    if sic2 in SIC_10:
        return "10"
    if sic2 in SIC_15:
        return "15"
    if sic2 in SIC_20:
        return "20"
    if sic2 in SIC_40:
        return "40"
    if sic2 in SIC_50:
        return "50"
    if sic2 in SIC_52:
        return "52"
    if sic2 in SIC_60:
        return "60"
    if sic2 in SIC_70:
        return "70"
    if sic2 in SIC_90:
        return "90"
    return DEFAULT_SIC_AGG


def load_vllm_configs(config_deduplication=1) -> List[Dict[str, Any]]:
    """
    Loads VLLM endpoint configurations from a JSON file.
    config_deduplication: Generate virtual copy of each config 1 means no duplication, 2 means 2 copies, etc.
    """
    if not VLLM_ENDPOINTS_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"VLLM endpoints config file not found: {VLLM_ENDPOINTS_CONFIG_PATH}"
        )

    with open(VLLM_ENDPOINTS_CONFIG_PATH, "r", encoding="utf-8") as f:
        configs = json.load(f)

    active_configs = [
        cfg
        for cfg in configs
        if not cfg.get("disabled", False) and cfg.get("endpoint") and cfg.get("model")
    ]
    if not active_configs:
        raise ValueError(
            "No active VLLM endpoint configurations found in the JSON file."
        )

    if config_deduplication > 1:
        output_configs = []
        for config in active_configs:
            # Create multiple copies of the config if requested
            for i in range(config_deduplication):
                output_configs.append(
                    {
                        "endpoint": config["endpoint"],
                        "model": config["model"],
                        "name": f"{config['name']}_{i + 1}",
                    }
                )
    else:
        output_configs = active_configs

    logging.info(f"Loaded {len(active_configs)} active VLLM endpoint configurations.")

    if config_deduplication > 1:
        logging.info(
            f"Each configuration has been duplicated {config_deduplication} times for parallel processing."
        )

    return output_configs
