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
    FINANCIALS_DIR_EXTENDED,
    MAX_CORE_USAGE,
    MAX_TAG_DEPTH,
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
COL_N_AGGREGATES = "n_aggregates"
COL_N_DIFF_FEATURES = "n_diff_features"
COL_N_BENISH_FEATURES = "n_benish_features"
COL_N_RATIOS = "n_ratios"
COL_N_FEATURES = "n_extended_features"

COL_FISCAL_YEAR_QUARTER = "fiscal_year_quarter"

EXTENDED_FINANCIAL_FEATURES_COUNT_COLS = [
    COL_N_IMPORTANT_TAGS,
    COL_N_AGGREGATES,
    COL_N_DIFF_FEATURES,
    COL_N_BENISH_FEATURES,
    COL_N_RATIOS,
    COL_N_FEATURES,
]


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


# ------------------------------------------------------------------------------
# DIFFERENTIAL FEATURE CONSTANTS
# ------------------------------------------------------------------------------
DIFF_WC_ACCRUALS = "diff_WC_Accruals"
DIFF_INVENTORIES = "diff_Inventories"
DIFF_RECEIVABLES = "diff_Receivables"
DIFF_CASH_SALES = "diff_CashSales"
DIFF_CASH_MARGIN = "diff_CashMargin"
DIFF_DEF_TAX_EXPENSE = "diff_DefTaxExpense"
DIFF_EARNINGS = "diff_Earnings"
DIFF_AVERAGE_ASSETS = "diff_AverageAssets"

DIFF_REVENUES = "diff_Revenues"
DIFF_CASH = "diff_Cash"
DIFF_EBIT = "diff_EBIT"
DIFF_EBITDA = "diff_EBITDA"
DIFF_NET_CASH_FLOW = "diff_NetCashFlow"
DIFF_DEPRECIATION = "diff_Depreciation"
DIFF_ASSETS = "diff_Assets"
DIFF_EQUITY = "diff_Equity"

DIFF_FEATURES = [
    DIFF_WC_ACCRUALS,
    DIFF_INVENTORIES,
    DIFF_RECEIVABLES,
    DIFF_CASH_SALES,
    DIFF_CASH_MARGIN,
    DIFF_DEF_TAX_EXPENSE,
    DIFF_EARNINGS,
    DIFF_AVERAGE_ASSETS,
    DIFF_REVENUES,
    DIFF_CASH,
    DIFF_EBIT,
    DIFF_EBITDA,
    DIFF_NET_CASH_FLOW,
    DIFF_DEPRECIATION,
    DIFF_ASSETS,
    DIFF_EQUITY,
]

# ------------------------------------------------------------------------------
# RATIO FEATURE CONSTANTS
# ------------------------------------------------------------------------------
RATIO_GROSS_PROFIT_MARGIN = "ratio_GrossProfitMargin"
RATIO_OPERATING_MARGIN = "ratio_OperatingMargin"
RATIO_NET_PROFIT_MARGIN = "ratio_NetProfitMargin"
RATIO_EBIT_MARGIN = "ratio_EBITMargin"
RATIO_EBITDA_MARGIN = "ratio_EBITDAMargin"
RATIO_CASH_FLOW_MARGIN = "ratio_CashFlowMargin"
RATIO_RETURN_ON_ASSETS = "ratio_ReturnOnAssets"
RATIO_RETURN_ON_EQUITY = "ratio_ReturnOnEquity"
RATIO_CURRENT_RATIO = "ratio_CurrentRatio"
RATIO_QUICK_RATIO = "ratio_QuickRatio"
RATIO_CASH_RATIO = "ratio_CashRatio"
RATIO_WC_TO_TOTAL_ASSETS = "ratio_WorkingCapitalToTotalAssets"
RATIO_DEBT_TO_ASSETS = "ratio_DebtToAssetsRatio"
RATIO_DEBT_TO_EQUITY = "ratio_DebtToEquityRatio"
RATIO_INTEREST_COVERAGE = "ratio_InterestCoverageRatio"
RATIO_TOTAL_LIAB_TO_ASSETS = "ratio_TotalLiabilitiesToAssets"
RATIO_ASSET_TURNOVER = "ratio_AssetTurnover"
RATIO_FIXED_ASSET_TURNOVER = "ratio_FixedAssetTurnover"
RATIO_RECEIVABLES_TURNOVER = "ratio_ReceivablesTurnover"
RATIO_INVENTORY_TURNOVER = "ratio_InventoryTurnover"
RATIO_SALES_TURNOVER = "ratio_SalesTurnover"
RATIO_EQUITY_MULTIPLIER = "ratio_EquityMultiplier"
RATIO_SGA_RATIO = "ratio_SGARatio"
RATIO_GOODWILL_TO_ASSETS = "ratio_GoodwilltoAssets"
RATIO_CASH_FLOW_TO_DEBT = "ratio_CashFlowToDebtRatio"
RATIO_CF_FINANCING_ACTIVITIES = "ratio_CashFlowFinancingActivities"
RATIO_CF_OPERATING_ACTIVITIES = "ratio_CashFlowOperatingActivities"
RATIO_EQUITY = "ratio_EquityRatio"
RATIO_OP_CF_TO_CURRENT_LIAB = "ratio_OperatingCashFlowToCurrentLiabilities"
RATIO_CASH_FLOW_TO_REVENUE = "ratio_CashFlowToRevenue"
RATIO_CASH_FLOW_COVERAGE = "ratio_CashFlowCoverageRatio"
RATIO_NET_WORKING_CAPITAL = "ratio_NetWorkingCapital"
RATIO_LT_DEBT_TO_EQUITY = "ratio_LongTermDebtToEquity"
RATIO_DEG_OF_FIN_LEVERAGE = "ratio_DegreeOfFinancialLeverage"
RATIO_INVESTED_CAPITAL = "ratio_InvestedCapitalRatio"
RATIO_CASH_TO_TOTAL_ASSET = "ratio_CashToTotalAsset"
RATIO_DEBT_SERVICE_COVERAGE = "ratio_DebtServiceCoverage"
RATIO_FIN_LEVERAGE_INDEX = "ratio_FinancialLeverageIndex"
RATIO_TIMES_INTEREST_EARNED = "ratio_TimesInterestEarnedRatio"
RATIO_CURRENT_ASSET_TO_REVENUES = "ratio_CurrentAssetToRevenues"
RATIO_CURRENT_LIAB_TO_REVENUES = "ratio_CurrentLiabilitiesToRevenues"
RATIO_SHORT_TERM_DEBT_TO_REVENUE = "ratio_ShortTermDebtToRevenue"
RATIO_INTANGIBLE_ASSET_TO_REVENUE = "ratio_IntangibleAssetToRevenue"
RATIO_LONGTERM_LEVERAGE = "ratio_LongtermLeverage"
RATIO_CFF = "ratio_CFF"

RATIO_FEATURES = [
    RATIO_GROSS_PROFIT_MARGIN,
    RATIO_OPERATING_MARGIN,
    RATIO_NET_PROFIT_MARGIN,
    RATIO_EBIT_MARGIN,
    RATIO_EBITDA_MARGIN,
    RATIO_CASH_FLOW_MARGIN,
    RATIO_RETURN_ON_ASSETS,
    RATIO_RETURN_ON_EQUITY,
    RATIO_CURRENT_RATIO,
    RATIO_QUICK_RATIO,
    RATIO_CASH_RATIO,
    RATIO_WC_TO_TOTAL_ASSETS,
    RATIO_DEBT_TO_ASSETS,
    RATIO_DEBT_TO_EQUITY,
    RATIO_INTEREST_COVERAGE,
    RATIO_TOTAL_LIAB_TO_ASSETS,
    RATIO_ASSET_TURNOVER,
    RATIO_FIXED_ASSET_TURNOVER,
    RATIO_RECEIVABLES_TURNOVER,
    RATIO_INVENTORY_TURNOVER,
    RATIO_SALES_TURNOVER,
    RATIO_EQUITY_MULTIPLIER,
    RATIO_SGA_RATIO,
    RATIO_GOODWILL_TO_ASSETS,
    RATIO_CASH_FLOW_TO_DEBT,
    RATIO_CF_FINANCING_ACTIVITIES,
    RATIO_CF_OPERATING_ACTIVITIES,
    RATIO_EQUITY,
    RATIO_OP_CF_TO_CURRENT_LIAB,
    RATIO_CASH_FLOW_TO_REVENUE,
    RATIO_CASH_FLOW_COVERAGE,
    RATIO_NET_WORKING_CAPITAL,
    RATIO_LT_DEBT_TO_EQUITY,
    RATIO_DEG_OF_FIN_LEVERAGE,
    RATIO_INVESTED_CAPITAL,
    RATIO_CASH_TO_TOTAL_ASSET,
    RATIO_DEBT_SERVICE_COVERAGE,
    RATIO_FIN_LEVERAGE_INDEX,
    RATIO_TIMES_INTEREST_EARNED,
    RATIO_CURRENT_ASSET_TO_REVENUES,
    RATIO_CURRENT_LIAB_TO_REVENUES,
    RATIO_SHORT_TERM_DEBT_TO_REVENUE,
    RATIO_INTANGIBLE_ASSET_TO_REVENUE,
    RATIO_LONGTERM_LEVERAGE,
    RATIO_CFF,
]


# ------------------------------------------------------------------------------
# BENEISH FEATURE CONSTANTS
# ------------------------------------------------------------------------------
BENEISH_PROBM = "Beneish_PROBM"
BENEISH_ACCRUALS = "Beneish_ACCRUALS"
BENEISH_DSR = "Beneish_DSR"
BENEISH_GMI = "Beneish_GMI"
BENEISH_AQI = "Beneish_AQI"
BENEISH_SGI = "Beneish_SGI"
BENEISH_DEPI = "Beneish_DEPI"
BENEISH_SGAI = "Beneish_SGAI"
BENEISH_LEVI = "Beneish_LEVI"

BENEISH_FEATURES = [
    BENEISH_PROBM,
    BENEISH_ACCRUALS,
    BENEISH_DSR,
    BENEISH_GMI,
    BENEISH_AQI,
    BENEISH_SGI,
    BENEISH_DEPI,
    BENEISH_SGAI,
    BENEISH_LEVI,
]

EXTENDED_FINANCIAL_FEATURES = (
    IMPORTANT_TAGS
    + AGGREGATE_FEATURES
    + DIFF_FEATURES
    + RATIO_FEATURES
    + BENEISH_FEATURES
)


EXTENDED_FEATURES_DESCRIPTION_DICT = {
    # Important Tags
    ACCOUNTS_PAYABLE_TAG: "Accounts Payables: Total amount owed to suppliers for goods or services purchased on credit.",
    ASSETS_TAG: "Total Assets: Sum of all assets owned by the company.",
    CASH_TAG: "Cash and Short-term Investments: Liquid assets readily convertible to cash.",
    INVENTORY_NET_TAG: "Inventory Net: Value of raw materials, work-in-progress, and finished goods.",
    PROPERTY_PLANT_EQUIPMENT_NET_TAG: "Property, Plant, and Equipment: Net value of fixed assets.",
    INTANGIBLE_ASSETS_NET_TAG: "Intangible Assets Net Including Goodwill: Value of intangible assets like patents, trademarks, and goodwill.",
    RETAINED_EARNINGS_TAG: "Retained Earnings: Accumulated profits retained in the business.",
    COMMON_STOCK_TAG: "Common Stock: Total value of common stock issued by the company.",
    PREFERRED_STOCK_TAG: "Preferred Stock: Total value of preferred stock issued by the company.",
    GOODWILL_TAG: "Goodwill: Excess of purchase price over the fair value of identifiable net assets acquired.",
    CURRENT_ASSETS_TAG: "Current Assets: Sum of assets expected to be converted to cash within one year.",
    CURRENT_LIABILITIES_TAG: "Current Liabilities: Sum of obligations due within one year.",
    TOTAL_LIABILITIES_TAG: "Total Liabilities: Sum of all obligations owed by the company.",
    SHORT_TERM_DEBT_TAG: "Short-Term Debt: Obligations due within one year.",
    ADDITIONAL_PAID_IN_CAPITAL_TAG: "Additional Paid-in Capital: Excess of proceeds over par value of stock issued.",
    AOCI_TAG: "Accumulated Other Comprehensive Income: Changes in equity not reflected in net income.",
    TREASURY_STOCK_TAG: "Treasury Stock: Shares of the company's own stock that have been repurchased.",
    TEMPORARY_EQUITY_TAG: "Temporary Equity: Equity that is redeemable or otherwise temporary in nature.",
    RECEIVABLE_FROM_SHAREHOLDERS_TAG: "Receivable from Shareholders: Amounts owed to the company by its shareholders.",
    MINORITY_INTEREST_TAG: "Minority Interest: Portion of equity in a subsidiary not attributable to the parent company.",
    UNEARNED_ESOP_SHARES_TAG: "Unearned ESOP Shares: Shares held by an Employee Stock Ownership Plan that have not yet been earned by employees.",
    COMMON_STOCK_HELD_BY_SUBSIDIARY_TAG: "Common Stock Held by Subsidiary: Shares of the company's common stock held by a subsidiary.",
    LONG_TERM_DEBT_CURRENT_TAG: "Long-Term Debt (Current Portion): Portion of long-term debt due within one year.",
    LONG_TERM_DEBT_NONCURRENT_TAG: "Long-Term Debt (Non-Current Portion): Portion of long-term debt due beyond one year.",
    ACCOUNT_RECEIVABLES_CURRENT_TAG: "Accounts Receivable (Current): Amounts due from customers within one year.",
    ACCOUNT_RECEIVABLES_NON_CURRENT_TAG: "Accounts Receivable (Non-Current): Amounts due from customers beyond one year.",
    AMORTIZATION_OF_INTANGIBLES_ASSETS: "Amortization of Intangibles: Expense recognized for amortizing intangible assets.",
    REVENUES_TAG: "Revenues: Total income generated from the sale of goods or services.",
    COST_OF_REVENUE_TAG: "Cost of Revenue: Direct costs attributable to the production of goods sold.",
    OPERATING_INCOME_LOSS_TAG: "Operating Income: Profit from core business operations.",
    INTEREST_EXPENSE_TAG: "Interest Expense: Cost incurred for borrowed funds.",
    SELLING_GENERAL_ADMINISTRATIVE_EXPENSE_TAG: "Selling, General, and Administrative Expense: Costs related to selling, marketing, and managing the business.",
    NET_INCOME_LOSS_TAG: "Net Income: Profit after all expenses and taxes.",
    GROSS_PROFIT_TAG: "Gross Profit: Revenue less cost of goods sold.",
    OPERATING_EXPENSES_TAG: "Operating Expenses: Costs incurred in the normal course of business operations.",
    INCOME_LOSS_FROM_CONTINUING_OPERATIONS: "Income from Continuing Operations: Profit from ongoing business activities.",
    DEFERRED_TAX_LIABILITIES_TAX_DEFERRED_INCOME: "Deferred Tax Liabilities: Tax liabilities or assets that result from temporary differences.",
    NET_CASH_FROM_OPERATIONS_TAG: "Net Cash from Operations: Cash generated from core business activities.",
    NET_CASH_FROM_FINANCING_TAG: "Net Cash from Financing: Cash from debt, equity, and dividends.",
    NET_CASH_FROM_INVESTING_TAG: "Net Cash from Investing: Cash from buying or selling assets.",
    DEFERRED_TAX_ASSET_INCOME: "Deferred Tax Asset: The future tax benefits from existing temporary differences and carry-forwards",
    DEFERRED_TAX_LIABILITY_EXPENSE: "Deferred Tax Liability: The future tax obligations from existing temporary differences and carry-forwards",
    DEPRECIATION_AND_AMORTIZATION_TAG: "Depreciation and Amortization: Expense reflecting asset value decline.",
    # Aggregate Features
    LONG_TERM_DEBT_AGG: "Long-term Debt: Obligations due beyond one year.",
    EQUITY_AGG: "Equity: The owner's stake in the company, calculated as assets minus liabilities.",
    TOTAL_DEBT_AGG: "Total Debt: The sum of short-term and long-term debt.",
    DEFERRED_TAX_EXPENSE_AGG: "Deferred Tax Expense: The change in deferred tax assets and liabilities during the period.",
    ACCRUALS_AGG: "Accruals: Adjustments for revenues that have been earned but not yet recorded, and for expenses that have been incurred but not yet recorded.",
    EBIT_AGG: "Earnings Before Interest and Taxes (EBIT): A measure of a company's profitability that excludes interest and tax expenses.",
    EBITDA_AGG: "Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA): A measure of a company's profitability that excludes interest, tax, depreciation, and amortization expenses.",
    NET_CASH_FLOW_AGG: "Net Cash Flow: The total amount of cash flowing into and out of a company.",
    ACCOUNT_RECEIVABLES_AGG: "Accounts Receivables: The amount of money owed to a company by its customers.",
    # Differential Features
    DIFF_WC_ACCRUALS: "Change in Working Capital Accruals: The change in current assets minus the change in current liabilities, excluding cash.",
    DIFF_INVENTORIES: "Change in Inventories: The change in the value of a company's inventories.",
    DIFF_RECEIVABLES: "Change in Receivables: The change in the amount of money owed to a company by its customers.",
    DIFF_CASH_SALES: "Change in Cash Sales: The change in a company's sales that were paid for in cash.",
    DIFF_CASH_MARGIN: "Change in Cash Margin: The change in a company's cash flow as a percentage of sales.",
    DIFF_DEF_TAX_EXPENSE: "Change in Deferred Tax Expense: The change in a company's deferred tax expense.",
    DIFF_EARNINGS: "Change in Earnings: The change in a company's net income.",
    DIFF_AVERAGE_ASSETS: "Average assets in the period (previous + current /2)",
    DIFF_REVENUES: "Change in Revenues: The change in a company's total income from sales.",
    DIFF_CASH: "Change in Cash: The change in a company's cash balance.",
    DIFF_EBIT: "Change in EBIT: The change in a company's earnings before interest and taxes.",
    DIFF_EBITDA: "Change in EBITDA: The change in a company's earnings before interest, taxes, depreciation, and amortization.",
    DIFF_NET_CASH_FLOW: "Change in Net Cash Flow: The change in a company's total amount of cash flowing into and out of the business.",
    DIFF_DEPRECIATION: "Change in Depreciation: The change in a company's depreciation expense.",
    DIFF_ASSETS: "Change in Assets: The change in a company's total assets.",
    DIFF_EQUITY: "Change in Equity: The change in the owner's stake in the company.",
    # Ratio Features
    RATIO_GROSS_PROFIT_MARGIN: "Gross Profit Margin: Gross profit as a percentage of revenue, indicating production efficiency.",
    RATIO_OPERATING_MARGIN: "Operating Margin: Operating income as a percentage of revenue, reflecting core business profitability.",
    RATIO_NET_PROFIT_MARGIN: "Net Profit Margin: Net income as a percentage of revenue, reflecting overall profitability after all costs.",
    RATIO_EBIT_MARGIN: "EBIT Margin: Earnings before interest and taxes as a percentage of revenue",
    RATIO_EBITDA_MARGIN: "EBITDA Margin: Earnings before interest, tax, depreciation, and amortization as a percentage of revenue",
    RATIO_CASH_FLOW_MARGIN: "Cash Flow Margin: Operating cash flow as a percentage of revenue.",
    RATIO_RETURN_ON_ASSETS: "Return on Assets (ROA): Net income as a percentage of total assets.",
    RATIO_RETURN_ON_EQUITY: "Return on Equity (ROE): Net income as a percentage of shareholder equity.",
    RATIO_CURRENT_RATIO: "Current Ratio: Current assets divided by current liabilities, measuring short-term liquidity.",
    RATIO_QUICK_RATIO: "Quick Ratio: (Current Assets - Inventory) / Current Liabilities, measuring immediate liquidity.",
    RATIO_CASH_RATIO: "Cash Ratio: Cash and cash equivalents divided by current liabilities, measuring immediate solvency.",
    RATIO_WC_TO_TOTAL_ASSETS: "Working Capital to Total Assets: Working capital (current assets - current liabilities) as a percentage of total assets, measuring efficiency.",
    RATIO_DEBT_TO_ASSETS: "Debt to Assets Ratio: Total debt divided by total assets, indicating financial leverage.",
    RATIO_DEBT_TO_EQUITY: "Debt to Equity Ratio: Total debt divided by shareholder equity, indicating financial risk.",
    RATIO_INTEREST_COVERAGE: "Interest Coverage Ratio: EBIT divided by interest expense, measuring ability to pay interest.",
    RATIO_TOTAL_LIAB_TO_ASSETS: "Total Liabilities to Assets: Total liabilities divided by total assets, assessing solvency.",
    RATIO_ASSET_TURNOVER: "Asset Turnover: Revenue divided by average total assets, indicating asset utilization.",
    RATIO_FIXED_ASSET_TURNOVER: "Fixed Asset Turnover: Revenue divided by average fixed assets, indicating fixed asset efficiency.",
    RATIO_RECEIVABLES_TURNOVER: "Receivables Turnover: Revenue divided by average accounts receivable, indicating credit collection efficiency.",
    RATIO_INVENTORY_TURNOVER: "Inventory Turnover: Cost of goods sold divided by average inventory, indicating inventory management efficiency.",
    RATIO_SALES_TURNOVER: "Sales Turnover: Total revenue generated from sales as percentage of average inventories.",
    RATIO_EQUITY_MULTIPLIER: "Equity Multiplier: Total assets divided by shareholder equity, indicating financial leverage.",
    RATIO_SGA_RATIO: "SG&A Ratio: Selling, general, and administrative expenses as a percentage of revenue, measuring operating efficiency.",
    RATIO_GOODWILL_TO_ASSETS: "Goodwill to Assets: Goodwill as a percentage of total assets, measuring intangible value.",
    RATIO_CASH_FLOW_TO_DEBT: "Cash Flow to Debt Ratio: Operating cash flow divided by total debt, measuring ability to repay debt.",
    RATIO_CF_FINANCING_ACTIVITIES: "Cash Flow from Financing: Percentage of cash flow from debt, equity, and dividends.",
    RATIO_CF_OPERATING_ACTIVITIES: "Cash Flow from Operations: Percentage of cash generated from core business.",
    RATIO_EQUITY: "EquityRatio: Percentage of assets that are financed by equity.",
    RATIO_OP_CF_TO_CURRENT_LIAB: "Operating Cash Flow To Current Liabilities: Measures the ability to pay off current liabilities with operating cash flow",
    RATIO_CASH_FLOW_TO_REVENUE: "Cash Flow to Revenue: Percentage of revenue which is cash",
    RATIO_CASH_FLOW_COVERAGE: "Cash Flow Coverage Ratio: Percentage of cash flow that is enough to cover the debt",
    RATIO_NET_WORKING_CAPITAL: "NetWorking Capital: Measures liquidity of the company",
    RATIO_LT_DEBT_TO_EQUITY: "Long Term Debt To Equity: Shows how a company finance its assets with debt vs equity",
    RATIO_DEG_OF_FIN_LEVERAGE: "Degree Of Financial Leverage: Measures the sensitivity of a company's earning per share to the change in EBIT",
    RATIO_INVESTED_CAPITAL: "Invested Capital Ratio: Used to see how much capital is tied up in inventory or plant equipment",
    RATIO_CASH_TO_TOTAL_ASSET: "Cash To Total Asset: Measures the proportion of a company’s assets that are in the form of cash",
    RATIO_DEBT_SERVICE_COVERAGE: "Debt Service Coverage: Measures a firm’s ability to use its earnings to meet its debt obligations",
    RATIO_FIN_LEVERAGE_INDEX: "Financial LeverageIndex: Used to determine the degree to which a business uses debt to finance its operations",
    RATIO_TIMES_INTEREST_EARNED: "Times InterestEarnedRatio: Used to measure a company’s ability to pay its debt obligations",
    RATIO_CURRENT_ASSET_TO_REVENUES: "Current Asset To Revenues: Assesses the efficiency with which a business is using its current assets to generate revenue",
    RATIO_CURRENT_LIAB_TO_REVENUES: "Current Liabilities To Revenues: A financial ratio that compares a company's current liabilities to its revenue",
    RATIO_SHORT_TERM_DEBT_TO_REVENUE: "Short TermDebt To Revenue: Used to analyze a company’s ability to pay off its short-term debts with revenue",
    RATIO_INTANGIBLE_ASSET_TO_REVENUE: "Intangible Asset ToRevenue: Used to assess how well a company is leveraging its intangible assets to generate revenue",
    RATIO_LONGTERM_LEVERAGE: "LongtermLeverage: Measures proportion of capital from debt and equity",
    RATIO_CFF: "Cash Flow Financing Ratio: (Financing Activities/Net Cash Flow)/Average Total Assets.",
    # Beneish Features
    BENEISH_PROBM: "Beneish M-score calculation: A  proxy probability of earnings manipulation.",
    BENEISH_ACCRUALS: "Accruals: Adjustments for revenues that have been earned but not yet recorded, and for expenses that have been incurred but not yet recorded relative to totals assets.",
    BENEISH_DSR: "Days' Sales in Receivables Index: Measures relative change in collection period",
    BENEISH_GMI: "Gross Margin Index: Measures relative change in gross margin.",
    BENEISH_AQI: "Asset Quality Index: Measures change in non-current assets relative to total assets.",
    BENEISH_SGI: "Sales Growth Index: Measures relative change in sales revenue.",
    BENEISH_DEPI: "Depreciation Index: Measures relative change in depreciation rate.",
    BENEISH_SGAI: "SG&A Index: Measures change in (selling, general, and administrative expenses relative to sales):[(SGA Expense t)/Sales t]/[(SGA Expense t−1)/Sales t−1].",
    BENEISH_LEVI: "Leverage Index: Measures relative change in leverage.",
}

EXTENDED_FEATURES_SHORT_DESCRIPTION_DICT = {
    key: value.split(":")[0]
    for key, value in EXTENDED_FEATURES_DESCRIPTION_DICT.items()
}


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return 0
    return numerator / denominator


def safe_sum(*args):
    if any(pd.isna(arg) or arg == 0 for arg in args):
        return 0
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

    if not FINANCIALS_DIR_EXTENDED.exists():
        FINANCIALS_DIR_EXTENDED.mkdir()
    dataset.to_csv(
        FINANCIALS_DIR_EXTENDED
        / f"{SEC_FINANCIALS_QTR_FILENAME_PREFIX}{directory.name}.csv",
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

    if not (FINANCIALS_DIR_EXTENDED).exists():
        (FINANCIALS_DIR_EXTENDED).mkdir(parents=True)
    tags_index_csvfile = FINANCIALS_DIR_EXTENDED / "sec_tags_index.csv"
    tags_index_csvfile_versionned = (
        FINANCIALS_DIR_EXTENDED / "sec_tags_index_versionned.csv"
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

    tags_index_csvfile = FINANCIALS_DIR_EXTENDED / "sec_tags_index.csv"
    tags_index = pd.read_csv(tags_index_csvfile)
    tag_columns = tags_index["tag"].tolist()
    tags_set = set(tag_columns)

    def taxonomy_infer(directory):
        configure_logger(
            Path(f"sec_financial_preprocessing_no_rf{RUN_TIMESTAMP}.log"), logging.INFO
        )
        year_quarter_name = directory.name
        partial_csv_file = (
            FINANCIALS_DIR_EXTENDED
            / f"sec_financials_quarterly_{year_quarter_name}.csv"
        )

        partial_df = pd.read_csv(partial_csv_file)
        partial_df = pivot_financial_data(partial_df)

        missing_tags = list(tags_set - set(partial_df.columns.tolist()))
        logging.info(f"Number of missing tags : {len(missing_tags)}")
        partial_df.loc[:, missing_tags] = 0

        partial_df = compute_missing_financial_data(
            partial_df, tags_index, bs_tree, is_tree, cf_tree
        )

        # Filter rows with too many missing values
        initial_size = len(partial_df)
        partial_df = partial_df[
            partial_df[IMPORTANT_TAGS].eq(0).sum(axis=1) < (len(IMPORTANT_TAGS) * 0.75)
        ]
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
    dataset_csvfile = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"

    tags_index_csvfile = FINANCIALS_DIR_EXTENDED / "sec_tags_index.csv"
    tags_index = pd.read_csv(tags_index_csvfile)

    aggregates_index = AGGREGATE_FEATURES
    differential_features_index = DIFF_FEATURES
    beneish_features_index = BENEISH_FEATURES
    ratios_index = RATIO_FEATURES

    columns_order = (
        [
            COL_COMPANY,
            COL_CIK,
            COL_YEAR,
            COL_QUARTER,
            COL_PERIOD,
            COL_SICAGG,
            COL_SIC,
            COL_N_TAGS,
            COL_N_TAGS_AUGMENTED,
            COL_N_TAGS_TOTAL,
            COL_N_IMPORTANT_TAGS,
            COL_N_RATIOS,
            COL_N_DIFF_FEATURES,
            COL_N_AGGREGATES,
            COL_N_BENISH_FEATURES,
            COL_N_FEATURES,
        ]
        # + list(sorted(tags_index.tag.unique().tolist()))
        + list(aggregates_index)
        + list(differential_features_index)
        + list(ratios_index)
        + list(beneish_features_index)
    )

    if dataset_csvfile.exists():
        dataset_csvfile.unlink()

    def merge_single_directory(directory):
        tqdm.tqdm.pandas()
        configure_logger(
            Path(f"sec_financial_preprocessing_{RUN_TIMESTAMP}.log"), logging.INFO
        )

        year_quarter_name = directory.name

        year = int(year_quarter_name[:4])
        quarter = int(year_quarter_name[-1])

        if quarter == 1:
            prev_year_quarter_name = f"{year-1}q{4}"
        else:
            prev_year_quarter_name = f"{year}q{quarter-1}"

        partial_csv_file = (
            FINANCIALS_DIR_EXTENDED
            / f"sec_financials_quarterly_{year_quarter_name}.csv"
        )

        prev_partial_csv_file = (
            FINANCIALS_DIR_EXTENDED
            / f"sec_financials_quarterly_{prev_year_quarter_name}.csv"
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

        partial_df = compute_differential_financial_features(partial_df)
        partial_df = compute_ratios(partial_df)
        partial_df = compute_beineish_features(partial_df)

        partial_df[COL_N_TAGS_TOTAL] = (
            partial_df[COL_N_TAGS] + partial_df[COL_N_TAGS_AUGMENTED]
        )

        partial_df[COL_N_IMPORTANT_TAGS] = partial_df[IMPORTANT_TAGS].ne(0).sum(axis=1)

        partial_df[COL_N_FEATURES] = (
            +partial_df[COL_N_RATIOS]
            + partial_df[COL_N_DIFF_FEATURES]
            + partial_df[COL_N_AGGREGATES]
            + partial_df[COL_N_BENISH_FEATURES]
            + partial_df[COL_N_IMPORTANT_TAGS]
        )

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

    data[COL_N_AGGREGATES] = data[AGGREGATE_FEATURES].ne(0).sum(axis=1)

    return data


def compute_differential_financial_features(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing differential financial features")

    data[DIFF_WC_ACCRUALS] = data.apply(
        lambda row: safe_sum(
            get_tag_diff_value(CURRENT_ASSETS_TAG, row),
            -get_tag_diff_value(CURRENT_LIABILITIES_TAG, row),
            -get_tag_diff_value(CASH_TAG, row),
        ),
        axis=1,
    )

    data[DIFF_INVENTORIES] = data.apply(
        lambda row: safe_divide(
            get_tag_diff_value(INVENTORY_NET_TAG, row),
            get_tag_avg_value(ASSETS_TAG, row),
        ),
        axis=1,
    )

    data[DIFF_RECEIVABLES] = data.apply(
        lambda row: safe_divide(
            get_tag_diff_value(ACCOUNT_RECEIVABLES_AGG, row),
            get_tag_avg_value(ASSETS_TAG, row),
        ),
        axis=1,
    )

    data[DIFF_CASH_SALES] = data.apply(
        lambda row: safe_sum(
            safe_divide(
                get_tag_value(REVENUES_TAG, row),
                get_tag_avg_value(INVENTORY_NET_TAG, row),
            ),
            safe_divide(
                get_tag_value(REVENUES_TAG, row, prev=True),
                get_tag_avg_value(INVENTORY_NET_TAG, row),
            ),
            -get_tag_diff_value(ACCOUNT_RECEIVABLES_AGG, row),
        )
        / 2,
        axis=1,
    )

    data[DIFF_CASH_MARGIN] = data.apply(
        lambda row: safe_divide(
            safe_sum(
                get_tag_value(COST_OF_REVENUE_TAG, row),
                -get_tag_diff_value(INVENTORY_NET_TAG, row),
                get_tag_diff_value(ACCOUNT_RECEIVABLES_AGG, row),
            ),
            get_tag_value(DIFF_CASH_SALES, row),
        ),
        axis=1,
    )

    data[DIFF_DEF_TAX_EXPENSE] = data.apply(
        lambda row: safe_divide(
            get_tag_diff_value(DEFERRED_TAX_EXPENSE_AGG, row),
            get_tag_value(ASSETS_TAG, row, prev=True),
        ),
        axis=1,
    )

    data[DIFF_EARNINGS] = data.apply(
        lambda row: safe_divide(
            get_tag_diff_value(NET_INCOME_LOSS_TAG, row),
            get_tag_avg_value(ASSETS_TAG, row),
        ),
        axis=1,
    )

    data[DIFF_AVERAGE_ASSETS] = data.apply(
        lambda row: get_tag_avg_value(ASSETS_TAG, row),
        axis=1,
    )

    data[DIFF_REVENUES] = data.apply(lambda row: get_tag_diff_value(REVENUES_TAG, row))

    data[DIFF_CASH] = data.apply(lambda row: get_tag_diff_value(CASH_TAG, row))

    data[DIFF_EBIT] = data.apply(lambda row: get_tag_diff_value(EBIT_AGG, row))

    data[DIFF_EBITDA] = data.apply(lambda row: get_tag_diff_value(EBITDA_AGG, row))

    data[DIFF_NET_CASH_FLOW] = data.apply(
        lambda row: get_tag_diff_value(NET_CASH_FLOW_AGG, row)
    )

    data[DIFF_DEPRECIATION] = data.apply(
        lambda row: get_tag_diff_value(DEPRECIATION_AND_AMORTIZATION_TAG, row)
    )

    data[DIFF_ASSETS] = data.apply(lambda row: get_tag_diff_value(ASSETS_TAG, row))

    data[DIFF_EQUITY] = data.apply(lambda row: get_tag_diff_value(EQUITY_AGG, row))

    data[COL_N_DIFF_FEATURES] = data[DIFF_FEATURES].ne(0).sum(axis=1)

    return data


def compute_ratios(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing financial ratios")

    ratios_list = [
        (
            RATIO_GROSS_PROFIT_MARGIN,
            "1. Gross margin",
            lambda row: safe_divide(
                get_tag_value(GROSS_PROFIT_TAG, row), get_tag_value(REVENUES_TAG, row)
            ),
        ),
        (
            RATIO_OPERATING_MARGIN,
            "2. OperatingMargin",
            lambda row: safe_divide(
                get_tag_value(OPERATING_INCOME_LOSS_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_NET_PROFIT_MARGIN,
            "3. NetProfitMargin",
            lambda row: safe_divide(
                get_tag_value(NET_INCOME_LOSS_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        # EBIT MARGIN
        (
            RATIO_EBIT_MARGIN,
            "4. EBITMargin",
            lambda row: safe_divide(
                get_tag_value(EBIT_AGG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        # EBITDA MARGIN
        (
            RATIO_EBITDA_MARGIN,
            "5. EBITDA Margin",
            lambda row: safe_divide(
                get_tag_value(EBITDA_AGG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        # CASH FLOW MARGIN
        (
            RATIO_CASH_FLOW_MARGIN,
            "6. CashFlowMargin",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_RETURN_ON_ASSETS,  # ROA
            "7. ReturnOnAssets",
            lambda row: safe_divide(
                get_tag_value(NET_INCOME_LOSS_TAG, row), get_tag_value(ASSETS_TAG, row)
            ),
        ),
        (
            RATIO_RETURN_ON_EQUITY,  # ROE
            "8. ReturnOnEquity",
            lambda row: safe_divide(
                get_tag_value(NET_INCOME_LOSS_TAG, row), get_tag_value(EQUITY_AGG, row)
            ),
        ),
        (
            RATIO_CURRENT_RATIO,
            "9. CurrentRatio",
            lambda row: safe_divide(
                get_tag_value(CURRENT_ASSETS_TAG, row),
                get_tag_value(CURRENT_LIABILITIES_TAG, row),
            ),
        ),
        (
            RATIO_QUICK_RATIO,
            "10. QuickRatio",
            lambda row: safe_divide(
                get_tag_value(CURRENT_ASSETS_TAG, row)
                - get_tag_value(INVENTORY_NET_TAG, row),
                get_tag_value(CURRENT_LIABILITIES_TAG, row),
            ),
        ),
        (
            RATIO_CASH_RATIO,
            "11. CashRatio",
            lambda row: safe_divide(
                get_tag_value(CASH_TAG, row),
                get_tag_value(CURRENT_LIABILITIES_TAG, row),
            ),
        ),
        (
            RATIO_WC_TO_TOTAL_ASSETS,
            "12. WorkingCapitalToTotalAssets",
            lambda row: safe_divide(
                get_tag_value(CURRENT_ASSETS_TAG, row)
                - get_tag_value(CURRENT_LIABILITIES_TAG, row),
                get_tag_value(ASSETS_TAG, row),
            ),
        ),
        (
            RATIO_DEBT_TO_ASSETS,
            "13. DebtToAssetsRatio",
            lambda row: safe_divide(
                get_tag_value(TOTAL_DEBT_AGG, row), get_tag_value(ASSETS_TAG, row)
            ),
        ),
        (
            RATIO_DEBT_TO_EQUITY,
            "14. DebtToEquityRatio",
            lambda row: safe_divide(
                get_tag_value(TOTAL_DEBT_AGG, row),
                get_tag_value(EQUITY_AGG, row),
            ),
        ),
        (
            RATIO_INTEREST_COVERAGE,
            "15. InterestCoverageRatio",
            lambda row: safe_divide(
                get_tag_value(OPERATING_INCOME_LOSS_TAG, row),
                get_tag_value(INTEREST_EXPENSE_TAG, row),
            ),
        ),
        (
            RATIO_TOTAL_LIAB_TO_ASSETS,
            "16. TotalLiabilitiesToAssets",
            lambda row: safe_divide(
                get_tag_value(TOTAL_LIABILITIES_TAG, row),
                get_tag_value(ASSETS_TAG, row),
            ),
        ),
        (
            RATIO_ASSET_TURNOVER,
            "17. AssetTurnover",
            lambda row: safe_divide(
                get_tag_value(REVENUES_TAG, row), get_tag_avg_value(ASSETS_TAG, row)
            ),
        ),
        (
            RATIO_FIXED_ASSET_TURNOVER,
            "18. FixedAssetTurnover",
            lambda row: safe_divide(
                get_tag_value(REVENUES_TAG, row),
                get_tag_avg_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row),
            ),
        ),
        (
            RATIO_RECEIVABLES_TURNOVER,
            "19. ReceivablesTurnover",
            lambda row: safe_divide(
                get_tag_value(REVENUES_TAG, row),
                get_tag_value(ACCOUNT_RECEIVABLES_AGG, row),
            ),
        ),
        (
            RATIO_INVENTORY_TURNOVER,
            "20. InventoryTurnover",
            lambda row: safe_divide(
                get_tag_value(COST_OF_REVENUE_TAG, row),
                get_tag_avg_value(INVENTORY_NET_TAG, row),
            ),
        ),
        (
            RATIO_SALES_TURNOVER,
            "21. SalesTurnover",
            lambda row: safe_divide(
                get_tag_value(REVENUES_TAG, row),
                get_tag_avg_value(INVENTORY_NET_TAG, row),
            ),
        ),
        (
            RATIO_EQUITY_MULTIPLIER,
            "22. EquityMultiplier",
            lambda row: safe_divide(
                get_tag_value(ASSETS_TAG, row),
                get_tag_value(EQUITY_AGG, row),
            ),
        ),
        (
            RATIO_SGA_RATIO,
            "23. SG&ARatio",
            lambda row: safe_divide(
                get_tag_value(SELLING_GENERAL_ADMINISTRATIVE_EXPENSE_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_GOODWILL_TO_ASSETS,
            "24. GoodwilltoAssets",
            lambda row: safe_divide(
                get_tag_value(GOODWILL_TAG, row), get_tag_value(ASSETS_TAG, row)
            ),
        ),
        (
            RATIO_CASH_FLOW_TO_DEBT,
            "25. CashFlowToDebtRatio",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
                get_tag_value(TOTAL_DEBT_AGG, row),
            ),
        ),
        (
            RATIO_CF_FINANCING_ACTIVITIES,
            "26. CashFlowFinancing Activities",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_FINANCING_TAG, row),
                get_tag_value(NET_CASH_FLOW_AGG, row),
            ),
        ),
        (
            RATIO_CF_OPERATING_ACTIVITIES,
            "27. CashFlowOperatingActivities",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
                get_tag_value(NET_CASH_FLOW_AGG, row),
            ),
        ),
        (
            RATIO_EQUITY,
            "28. EquityRatio",
            lambda row: safe_divide(
                get_tag_value(EQUITY_AGG, row), get_tag_value(ASSETS_TAG, row)
            ),
        ),
        (
            RATIO_OP_CF_TO_CURRENT_LIAB,
            "29. OperatingCashFlowToCurrentLiabilities",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
                get_tag_value(CURRENT_LIABILITIES_TAG, row),
            ),
        ),
        (
            RATIO_CASH_FLOW_TO_REVENUE,
            "30. CashFlowToRevenue",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_CASH_FLOW_COVERAGE,
            "31. CashFlowCoverageRatio",
            lambda row: safe_divide(
                get_tag_value(NET_CASH_FROM_OPERATIONS_TAG, row),
                get_tag_value(TOTAL_DEBT_AGG, row),
            ),
        ),
        (
            RATIO_NET_WORKING_CAPITAL,
            "32. NetWorkingCapital",
            lambda row: get_tag_value(CURRENT_ASSETS_TAG, row)
            - get_tag_value(CURRENT_LIABILITIES_TAG, row),
        ),
        (
            RATIO_LT_DEBT_TO_EQUITY,
            "33. LongTermDebtToEquity",
            lambda row: safe_divide(
                get_tag_value(LONG_TERM_DEBT_AGG, row),
                get_tag_value(EQUITY_AGG, row),
            ),
        ),
        (
            RATIO_DEG_OF_FIN_LEVERAGE,
            "34. DegreeOfFinancialLeverage",
            lambda row: safe_divide(
                get_tag_value(REVENUES_TAG, row),
                get_tag_value(NET_INCOME_LOSS_TAG, row),
            ),
        ),
        (
            RATIO_INVESTED_CAPITAL,
            "35. InvestedCapitalRatio",
            lambda row: safe_divide(
                get_tag_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row)
                + get_tag_value(INVENTORY_NET_TAG, row),
                get_tag_value(ASSETS_TAG, row),
            ),
        ),
        (
            RATIO_CASH_TO_TOTAL_ASSET,
            "36. CashToTotalAsset",
            lambda row: safe_divide(
                get_tag_value(CASH_TAG, row), get_tag_value(ASSETS_TAG, row)
            ),
        ),
        (
            RATIO_DEBT_SERVICE_COVERAGE,
            "37. DebtServiceCoverage",
            lambda row: safe_divide(
                get_tag_value(OPERATING_INCOME_LOSS_TAG, row),
                get_tag_value(TOTAL_DEBT_AGG, row),
            ),
        ),
        (
            RATIO_FIN_LEVERAGE_INDEX,
            "38. FinancialLeverageIndex",
            lambda row: safe_divide(
                get_tag_value(OPERATING_INCOME_LOSS_TAG, row),
                get_tag_value(ASSETS_TAG, row),
            ),
        ),
        (
            RATIO_TIMES_INTEREST_EARNED,
            "39. TimesInterestEarnedRatio",
            lambda row: safe_divide(
                get_tag_value(NET_INCOME_LOSS_TAG, row)
                + get_tag_value(INTEREST_EXPENSE_TAG, row),
                get_tag_value(INTEREST_EXPENSE_TAG, row),
            ),
        ),
        (
            RATIO_CURRENT_ASSET_TO_REVENUES,
            "40. CurrentAssetToRevenues",
            lambda row: safe_divide(
                get_tag_value(CURRENT_ASSETS_TAG, row), get_tag_value(REVENUES_TAG, row)
            ),
        ),
        (
            RATIO_CURRENT_LIAB_TO_REVENUES,
            "41. CurrentLiabilitiesToRevenues",
            lambda row: safe_divide(
                get_tag_value(CURRENT_LIABILITIES_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_SHORT_TERM_DEBT_TO_REVENUE,
            "42. ShortTermDebtToRevenue",
            lambda row: safe_divide(
                get_tag_value(SHORT_TERM_DEBT_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_INTANGIBLE_ASSET_TO_REVENUE,
            "43. IntangibleAssetToRevenue",
            lambda row: safe_divide(
                get_tag_value(INTANGIBLE_ASSETS_NET_TAG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
        ),
        (
            RATIO_LONGTERM_LEVERAGE,
            "44. LongtermLeverage",
            lambda row: safe_divide(
                get_tag_value(LONG_TERM_DEBT_AGG, row),
                get_tag_value(ASSETS_TAG, row),
            ),
        ),
        (
            RATIO_CFF,
            "45. CFF",
            lambda row: safe_divide(
                safe_divide(
                    get_tag_value(NET_CASH_FROM_FINANCING_TAG, row),
                    get_tag_value(NET_CASH_FLOW_AGG, row),
                ),
                get_tag_avg_value(ASSETS_TAG, row),
            ),
        ),
    ]

    ratios_name_list = []
    for ratio_name, log_message, calculation in ratios_list:
        logging.info(f"  {log_message}")
        ratios_name_list.append(ratio_name)
        data[ratio_name] = data.progress_apply(calculation, axis=1)

    data[COL_N_RATIOS] = data[RATIO_FEATURES].ne(0).sum(axis=1)
    return data


def compute_beineish_features(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing Beneish model features")

    def calculate_beneish_features(row):
        dsr = safe_divide(
            safe_divide(
                get_tag_value(ACCOUNT_RECEIVABLES_AGG, row),
                get_tag_value(REVENUES_TAG, row),
            ),
            safe_divide(
                get_tag_value(ACCOUNT_RECEIVABLES_AGG, row, prev=True),
                get_tag_value(REVENUES_TAG, row, prev=True),
            ),
        )

        prev_gross_margin = safe_divide(
            safe_sum(
                get_tag_value(REVENUES_TAG, row, prev=True),
                -get_tag_value(COST_OF_REVENUE_TAG, row, prev=True),
            ),
            get_tag_value(REVENUES_TAG, row, prev=True),
        )

        curr_gross_margin = safe_divide(
            safe_sum(
                get_tag_value(REVENUES_TAG, row),
                -get_tag_value(COST_OF_REVENUE_TAG, row),
            ),
            get_tag_value(REVENUES_TAG, row),
        )

        gmi = safe_divide(prev_gross_margin, curr_gross_margin)

        prev_non_current_assets = safe_sum(
            get_tag_value(ASSETS_TAG, row, prev=True),
            -get_tag_value(CURRENT_ASSETS_TAG, row, prev=True),
            -get_tag_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row, prev=True),
        )
        prev_asset_quality = safe_divide(
            prev_non_current_assets, get_tag_value(ASSETS_TAG, row, prev=True)
        )

        curr_non_current_assets = safe_sum(
            get_tag_value(ASSETS_TAG, row),
            -get_tag_value(CURRENT_ASSETS_TAG, row),
            -get_tag_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row),
        )
        curr_asset_quality = safe_divide(
            curr_non_current_assets, get_tag_value(ASSETS_TAG, row)
        )

        aqi = safe_divide(curr_asset_quality, prev_asset_quality)

        sgi = safe_divide(
            get_tag_value(REVENUES_TAG, row),
            get_tag_value(REVENUES_TAG, row, prev=True),
        )

        prev_depr_rate = safe_divide(
            get_tag_value(DEPRECIATION_AND_AMORTIZATION_TAG, row, prev=True),
            safe_sum(
                get_tag_value(DEPRECIATION_AND_AMORTIZATION_TAG, row, prev=True),
                get_tag_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row, prev=True),
            ),
        )

        curr_depr_rate = safe_divide(
            get_tag_value(DEPRECIATION_AND_AMORTIZATION_TAG, row),
            safe_sum(
                get_tag_value(DEPRECIATION_AND_AMORTIZATION_TAG, row),
                get_tag_value(PROPERTY_PLANT_EQUIPMENT_NET_TAG, row),
            ),
        )

        depi = safe_divide(prev_depr_rate, curr_depr_rate)

        prev_sga_ratio = safe_divide(
            get_tag_value(SELLING_GENERAL_ADMINISTRATIVE_EXPENSE_TAG, row, prev=True),
            get_tag_value(REVENUES_TAG, row, prev=True),
        )

        curr_sga_ratio = safe_divide(
            get_tag_value(SELLING_GENERAL_ADMINISTRATIVE_EXPENSE_TAG, row),
            get_tag_value(REVENUES_TAG, row),
        )

        sgai = safe_divide(curr_sga_ratio, prev_sga_ratio)

        accruals = safe_divide(
            get_tag_value(ACCRUALS_AGG, row), get_tag_value(ASSETS_TAG, row)
        )

        prev_leverage = safe_divide(
            safe_sum(
                get_tag_value(LONG_TERM_DEBT_AGG, row, prev=True),
                get_tag_value(SHORT_TERM_DEBT_TAG, row, prev=True),
            ),
            get_tag_value(ASSETS_TAG, row, prev=True),
        )

        curr_leverage = safe_divide(
            safe_sum(
                get_tag_value(LONG_TERM_DEBT_AGG, row),
                get_tag_value(SHORT_TERM_DEBT_TAG, row),
            ),
            get_tag_value(ASSETS_TAG, row),
        )

        levi = safe_divide(curr_leverage, prev_leverage)

        prob_m = safe_sum(
            4.84,
            0.920 * dsr,
            0.528 * gmi,
            0.404 * aqi,
            0.892 * sgi,
            0.155 * depi,
            -0.172 * sgai,
            4.679 * accruals,
            -0.327 * levi,
        )

        return pd.Series(
            {
                BENEISH_PROBM: prob_m,
                BENEISH_ACCRUALS: accruals,
                BENEISH_DSR: dsr,
                BENEISH_GMI: gmi,
                BENEISH_AQI: aqi,
                BENEISH_SGI: sgi,
                BENEISH_DEPI: depi,
                BENEISH_SGAI: sgai,
                BENEISH_LEVI: levi,
            }
        )

    tqdm.tqdm.pandas()
    data = data.fillna(0)
    data[
        [
            BENEISH_PROBM,
            BENEISH_ACCRUALS,
            BENEISH_DSR,
            BENEISH_GMI,
            BENEISH_AQI,
            BENEISH_SGI,
            BENEISH_DEPI,
            BENEISH_SGAI,
            BENEISH_LEVI,
        ]
    ] = data.progress_apply(calculate_beneish_features, axis=1)

    data[COL_N_BENISH_FEATURES] = data[BENEISH_FEATURES].ne(0).sum(axis=1)

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
            FINANCIALS_DIR_EXTENDED
            / f"sec_financials_quarterly_{year_quarter_name}.csv"
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
        COL_N_TAGS,
        COL_N_TAGS_AUGMENTED,
        COL_N_TAGS_TOTAL,
        COL_N_IMPORTANT_TAGS,
        COL_N_RATIOS,
        COL_N_DIFF_FEATURES,
        COL_N_AGGREGATES,
        COL_N_BENISH_FEATURES,
        COL_N_FEATURES,
    ]

    dataset_csvfile = FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly.csv"
    main_df = pd.read_csv(dataset_csvfile, usecols=main_columns)

    # Get the nubmer of tags for the tags index
    tags_index_csvfile = FINANCIALS_DIR_EXTENDED / "sec_tags_index.csv"
    tags_index = pd.read_csv(tags_index_csvfile)
    num_tags = tags_index["tag"].nunique()

    def get_col_stat(col_name):
        return {
            "avg": main_df[col_name].mean().item(),
            "min": main_df[col_name].min().item(),
            "max": main_df[col_name].max().item(),
            "std": main_df[col_name].std().item(),
            "median": main_df[col_name].median().item(),
            "count": {
                COL_N_AGGREGATES: len(AGGREGATE_FEATURES),
                COL_N_BENISH_FEATURES: len(BENEISH_FEATURES),
                COL_N_DIFF_FEATURES: len(DIFF_FEATURES),
                COL_N_RATIOS: len(RATIO_FEATURES),
                COL_N_IMPORTANT_TAGS: len(IMPORTANT_TAGS),
                COL_N_TAGS: num_tags,
                COL_N_TAGS_AUGMENTED: num_tags,
                COL_N_TAGS_TOTAL: num_tags,
                COL_N_FEATURES: len(
                    AGGREGATE_FEATURES
                    + BENEISH_FEATURES
                    + DIFF_FEATURES
                    + RATIO_FEATURES
                    + IMPORTANT_TAGS
                ),
            }[col_name],
        }

    stats_dict = {
        "counts": {
            "n_reports": main_df[[COL_CIK, COL_YEAR, COL_QUARTER]]
            .drop_duplicates()
            .shape[0],
            "n_companies": main_df["cik"].nunique(),
        },
        "summary_sic_agg": main_df["sicagg"].value_counts().to_dict(),
        "summary_sic": main_df["sic"].value_counts().to_dict(),
        "n_tags_stats": get_col_stat(COL_N_TAGS),
        "n_tags_augmented_stats": get_col_stat(COL_N_TAGS_AUGMENTED),
        "n_tags_total_stats": get_col_stat(COL_N_TAGS_TOTAL),
        "n_ratios_stats": get_col_stat(COL_N_RATIOS),
        "n_diff_features_stats": get_col_stat(COL_N_DIFF_FEATURES),
        "n_aggregates_stats": get_col_stat(COL_N_AGGREGATES),
        "n_benish_features_stats": get_col_stat(COL_N_BENISH_FEATURES),
        "n_important_tags_stats": get_col_stat(COL_N_IMPORTANT_TAGS),
        "n_extended_features_stats": get_col_stat(COL_N_FEATURES),
    }
    save_dataset_config(str(FINANCIALS_DIR_EXTENDED), **stats_dict)

    main_df.to_csv(
        FINANCIALS_DIR_EXTENDED / "sec_financials_quarterly_no_financials.csv"
    )

    logging.info("Saving index of CIK, FISCAL_QUART TO PERIOD")
    # New feature, Save an index of (CIK, FISCAL_QUARTER): PERIOD  to know the periods of all fiscal quartesm
    main_df[COL_FISCAL_YEAR_QUARTER] = main_df["year"].astype(str) + main_df[
        "quarter"
    ].astype(str)
    df_periods = main_df[
        [COL_CIK, COL_FISCAL_YEAR_QUARTER, COL_PERIOD]
    ].drop_duplicates()

    index_periods = {}
    for _, row in df_periods.iterrows():
        index_periods.setdefault(row[COL_CIK], {})[row[COL_FISCAL_YEAR_QUARTER]] = row[
            COL_PERIOD
        ]

    with open(
        FINANCIALS_DIR_EXTENDED / "fiscal_quarter_to_periods_index.json", "w"
    ) as f:
        f.write(json.dumps(index_periods, indent=4))

    # Now find the fiscal year end dates for each CIK
    fiscal_year_ends = get_fiscal_year_ends(index_periods)
    with open(FINANCIALS_DIR_EXTENDED / "fiscal_year_end_dates.json", "w") as f:
        f.write(json.dumps(fiscal_year_ends, indent=4))


def get_fiscal_year_ends(data: dict) -> dict:
    """
    Infers the typical fiscal year-end month and day (MMDD format) for each CIK.

    It does this by:
    1. Taking the first available fiscal quarter end date for each CIK.
    2. Extracting the quarter number from the quarter string (e.g., 'q3' -> 3).
    3. Calculating the number of months needed to reach the end of the 4th quarter
       (assuming quarters are 3 months long and fiscal year ends after Q4).
    4. Adding these months to the extracted date to determine the fiscal year-end date.
    5. Formatting the fiscal year-end date as MMDD.

    Args:
        data (dict): A dictionary where keys are CIKs (str) and values are
                     dictionaries. Each inner dictionary contains fiscal period
                     strings (e.g., '2010q3') as keys and their integer end dates
                     (e.g., 20100531) as values.

    Returns:
        dict: A dictionary mapping CIKs (str) to their inferred fiscal year-end
              dates in MMDD string format (e.g., '0831'). Returns None for a CIK
              if data parsing fails.
    """
    fiscal_year_ends = {}

    for cik, period_dates_map in data.items():
        if not period_dates_map:
            # Handle cases where the inner dictionary might be empty for a CIK
            fiscal_year_ends[cik] = None
            continue

        # Get the first item from the inner dictionary.
        # dict.items() returns an iterable of (key, value) pairs.
        # next(iter(...)) gets the first element from this iterable.
        first_period_str, first_date_int = next(iter(period_dates_map.items()))

        # Extract the quarter number from the period string (e.g., '2010q3' -> 3)
        # This assumes the format is always 'YYYYqX'
        try:
            q_index = first_period_str.lower().find("q")
            if q_index != -1 and q_index + 1 < len(first_period_str):
                quarter_number = int(first_period_str[q_index + 1])
            else:
                print(
                    f"Warning: Could not parse quarter from '{first_period_str}' for CIK {cik}. Skipping."
                )
                fiscal_year_ends[cik] = None
                continue
        except (ValueError, IndexError):
            print(
                f"Warning: Invalid quarter number format in '{first_period_str}' for CIK {cik}. Skipping."
            )
            fiscal_year_ends[cik] = None
            continue

        # Convert the integer date to a datetime object
        date_str = str(first_date_int)
        try:
            dt_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(
                f"Warning: Could not parse date '{date_str}' for CIK {cik}. Skipping."
            )
            fiscal_year_ends[cik] = None
            continue

        # Calculate months to add to reach the end of Q4
        # (4 - quarter_number) gives the number of quarters remaining until Q4.
        # Multiply by 3 for months (assuming 3 months per quarter).
        months_to_add = (4 - quarter_number) * 3

        # Add the months using relativedelta to handle month and year transitions correctly.
        # This is crucial for dates like 'Nov 30' + 3 months -> 'Feb 28/29' of next year.
        fy_end_dt = dt_obj + relativedelta(months=months_to_add)

        # Format the month and day as MMDD string
        fy_end_mmdd = fy_end_dt.strftime("%m%d")

        fiscal_year_ends[cik] = fy_end_mmdd

    return fiscal_year_ends


if __name__ == "__main__":

    configure_logger(
        Path(f"sec_financial_preprocessing_{RUN_TIMESTAMP}.log"), logging.INFO
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
