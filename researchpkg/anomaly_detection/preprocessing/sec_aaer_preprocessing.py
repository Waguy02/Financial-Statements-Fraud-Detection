import calendar  # For is_leap_year
import json
import logging
from datetime import date, datetime

import pandas as pd
from dateutil.relativedelta import (
    relativedelta,  # You might need to install this: pip install python-dateutil
)
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    AAER_DATASET_PATH,
    AAER_DETAILS_PATH,
    CIK_TO_FY_END_INDEX,
    LIST_MISTATEMENT_TYPE,
    LIST_MISTATEMENT_TYPE_RENAMED,
    LIST_MISTATEMENT_TYPE_RENAMING_MAPPING,
    PREPROCESSED_PATH,
)

columns = [
    "aaer_id",
    "aaer_no",
    "aeer_date",
    "company",
    "role",
    "cik",
    "year",
    "fraud_scheme_description",
    "fiscal_quarter",
    "url",
    "summary",
] + [LIST_MISTATEMENT_TYPE_RENAMING_MAPPING[m] for m in LIST_MISTATEMENT_TYPE]


columns_display = [
    "aaer_id",
    "aeer_date",
    "company",
    "cik",
    "year",
    "role",
    "url",
    "summary",
]


AAER_START_YEAR = "2009"


def _is_leap_year(year: int) -> bool:
    """Helper to check if a year is a leap year."""
    return calendar.isleap(year)


def __civil_year_to_fiscal_year(
    civil_quarter_str: str, fiscal_year_end_month: str
) -> str:
    """
    Map a civil quarter string (YYYYQn) to its corresponding fiscal year and fiscal quarter (YYYYqn).

    This function assumes valid input formats for civil_quarter_str and fiscal_year_end_month.
    It will raise standard Python errors (e.g., ValueError, IndexError) if inputs are malformed.

    Args:
        civil_quarter_str (str): A string representing a civil quarter in
                                 'YYYYQn' format (e.g., '2023Q1', '2024q3').
                                 The 'n' represents the quarter number (1-4).
        fiscal_year_end_month (str): A string representing the typical fiscal year
                                     end month and day in 'MMDD' format (e.g., '0531').

    Returns:
        str: The inferred fiscal year and quarter in 'YYYYqn' string format (e.g., '2023q3').
    """
    # --- Parse civil_quarter_str (YYYYQn) to get the civil_date (end date of that quarter) ---

    civil_year = int(civil_quarter_str[:4])

    if civil_quarter_str[4].lower() != "q":
        raise ValueError(
            "civil_quarter_str must contain 'Q' or 'q' at the 5th position."
        )
    civil_quarter_num = int(civil_quarter_str[5:])

    # Ensure quarter number is valid
    if not (1 <= civil_quarter_num <= 4):
        raise ValueError("Quarter number in civil_quarter_str must be between 1 and 4.")

    # Determine the fixed end day and month for a standard civil quarter
    # These dates are always valid.
    if civil_quarter_num == 1:  # Q1 ends March 31
        civil_date = date(civil_year, 3, 31)
    elif civil_quarter_num == 2:  # Q2 ends June 30
        civil_date = date(civil_year, 6, 30)
    elif civil_quarter_num == 3:  # Q3 ends September 30
        civil_date = date(civil_year, 9, 30)
    else:  # civil_quarter_num == 4: Q4 ends December 31
        civil_date = date(civil_year, 12, 31)

    # --- Parse fiscal_year_end_month (MMDD) ---
    # Assume fiscal_year_end_month is 'MMDD' (4 characters)
    fy_end_month = int(fiscal_year_end_month[:2])
    fy_end_day = int(fiscal_year_end_month[2:])

    # --- Step 1: Determine the primary fiscal year based on the civil_date ---

    # Construct the fiscal year end date for the *current civil year* of the civil_date.
    # Explicitly handle Feb 29th for non-leap years *before* constructing the date.
    if fy_end_month == 2 and fy_end_day == 29 and not _is_leap_year(civil_date.year):
        fy_end_date_current_civil_year = date(civil_date.year, 2, 28)
    else:
        # This will raise ValueError if month/day combination is invalid (e.g., 0230, 0431)
        fy_end_date_current_civil_year = date(civil_date.year, fy_end_month, fy_end_day)

    # Determine the fiscal year based on the comparison with the current civil year's fiscal end.
    if civil_date <= fy_end_date_current_civil_year:
        fiscal_year = civil_date.year
    else:
        fiscal_year = civil_date.year + 1

    # --- Step 2: Calculate the exact fiscal year end date for the determined fiscal_year ---
    if fy_end_month == 2 and fy_end_day == 29 and not _is_leap_year(fiscal_year):
        fiscal_year_end_date = date(fiscal_year, 2, 28)
    else:
        # This will raise ValueError if month/day combination is invalid
        fiscal_year_end_date = date(fiscal_year, fy_end_month, fy_end_day)

    # --- Step 3: Calculate the start and end dates for all four quarters of the determined fiscal_year ---
    # These calculations rely on relativedelta, which handles month transitions correctly.
    q4_end = fiscal_year_end_date
    q4_start = (q4_end - relativedelta(months=3)) + relativedelta(days=1)

    q3_end = q4_start - relativedelta(days=1)
    q3_start = (q3_end - relativedelta(months=3)) + relativedelta(days=1)

    q2_end = q3_start - relativedelta(days=1)
    q2_start = (q2_end - relativedelta(months=3)) + relativedelta(days=1)

    q1_end = q2_start - relativedelta(days=1)
    q1_start = (q1_end - relativedelta(months=3)) + relativedelta(days=1)

    # --- Step 4: Determine which fiscal quarter the civil_date falls into ---

    def is_in_range(date_to_check, start_date, end_date, epsilon=2):
        """
        Check if date_to_check is within the range of start_date and end_date.
        """

        return (
            start_date - relativedelta(days=epsilon)
            <= date_to_check
            <= end_date + relativedelta(days=epsilon)
        )

    fiscal_quarter_num = None
    if is_in_range(civil_date, q1_start, q1_end):
        fiscal_quarter_num = 1
    elif is_in_range(civil_date, q2_start, q2_end):
        fiscal_quarter_num = 2
    elif is_in_range(civil_date, q3_start, q3_end):
        fiscal_quarter_num = 3
    elif is_in_range(civil_date, q4_start, q4_end):
        fiscal_quarter_num = 4

    # --- Step 5: Format and return the result in YYYYqn format ---
    return f"{fiscal_year}q{fiscal_quarter_num}"


def preprocess_downloaded_aaer_data():
    all_files = list(AAER_DETAILS_PATH.rglob("*.json"))
    df_aaer = pd.DataFrame({c: [] for c in columns})

    with open(CIK_TO_FY_END_INDEX, "r") as infile:
        cik_to_fy_end_index = json.load(infile)

    for f in tqdm(all_files, desc="Preprocessing aaer files"):
        with open(f, "r") as infile:
            instance = json.load(infile)
            aaerNo = instance["aaerNo"]
            aaerDate = instance["dateTime"][:10]

            details = instance["details"]
            if len(details) == 0:
                # Skip if details are empty
                continue

            summary = instance.get("summary", "")

            primary_url = [
                e["url"] for e in instance["urls"] if e["type"] == "primary"
            ][0]
            # for quarter_data in
            for quarter_report in details:
                misstatements = quarter_report["misstatements"]

                if len(misstatements) == 0:
                    # Skip if no misstatements
                    continue
                mistating_company = quarter_report["misstating_company"]
                cik = mistating_company.get("cik", "")
                if cik == "":
                    # Skip if CIK is empty
                    continue
                quarter = quarter_report["quarter"]

                if quarter.lower() == "n/a":
                    # Skip if quarter is "N/A"
                    continue

                is_fiscal_quarter = quarter_report["is_fiscal_quarter"]
                role = mistating_company["role"]
                name = mistating_company["name"]
                fraud_scheme_description = quarter_report["fraud_scheme_description"]
                misstatements_binary_indicator = [
                    1 if m in misstatements else 0 for m in LIST_MISTATEMENT_TYPE
                ]

                if is_fiscal_quarter:
                    fiscal_quarter = quarter
                else:
                    if cik not in cik_to_fy_end_index:
                        logging.warning(
                            f"CIK {cik} not found in cik_to_fy_end_index. Skipping."
                        )
                        continue

                    fiscal_quarter = __civil_year_to_fiscal_year(
                        quarter, cik_to_fy_end_index[cik]
                    )

                year = int(fiscal_quarter[:4])

                aaer_id = f"{aaerNo}_{cik}_{fiscal_quarter}"

                # Check if the cik and fiscal_quarter already exists in the DataFrame
                # If so merge the misstatements and concat the fraud_scheme_description
                if (
                    df_aaer[
                        (df_aaer["cik"] == cik)
                        & (df_aaer["fiscal_quarter"] == fiscal_quarter)
                    ].shape[0]
                    > 0
                ):
                    # Merge misstatements
                    existing_row = df_aaer[
                        (df_aaer["cik"] == cik)
                        & (df_aaer["fiscal_quarter"] == fiscal_quarter)
                    ].iloc[0]

                    # Update the misstatements binary indicator
                    for i, m in enumerate(LIST_MISTATEMENT_TYPE):
                        if misstatements_binary_indicator[i] == 1:
                            existing_row[m] = 1
                    # Update the fraud_scheme_description
                    existing_row[
                        "fraud_scheme_description"
                    ] += f" \n\n {fraud_scheme_description}"
                else:
                    # Create a new row if it does not exist

                    # Append new row using loc
                    new_rows = [
                        aaer_id,
                        aaerNo,
                        aaerDate,
                        name,
                        role,
                        cik,
                        year,
                        fraud_scheme_description,
                        fiscal_quarter,
                        primary_url,
                        summary,
                    ] + misstatements_binary_indicator
                    df_aaer.loc[len(df_aaer)] = new_rows

    print(f"Number of AAERs :", len(df_aaer))

    output_dir = PREPROCESSED_PATH / f"AAER_PREPROCESSED"
    output_dir.mkdir(exist_ok=True, parents=True)

    df_aaer.to_csv(output_dir / "df_aaer_with_misstatements.csv", index=False)

    # Aaer since 2009
    df_aaer_since_2009 = df_aaer[df_aaer["year"] >= int(AAER_START_YEAR)]
    df_aaer_since_2009.to_csv(
        output_dir / "df_aaer_with_misstatements_since_2009.csv", index=False
    )

    stats_dir = output_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    # Compute Statistics on the number of misstatements
    misstatements_count = df_aaer[LIST_MISTATEMENT_TYPE_RENAMED].sum(axis=0)
    misstatements_count = misstatements_count.sort_values(ascending=False)
    misstatements_count.to_csv(stats_dir / "misstatements_count.csv")

    # Plot the distribution of misstatements
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    sns.barplot(x=misstatements_count.index, y=misstatements_count.values)
    plt.xticks(rotation=90)
    plt.title("Distribution of Misstatements in AAERs")

    plt.xlabel("Misstatement Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(stats_dir / "misstatements_distribution.png")
    plt.close()

    # Distribution of misstatements as percentage
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=misstatements_count.index, y=misstatements_count.values / len(df_aaer) * 100
    )
    plt.xticks(rotation=90)
    plt.title("Percentage of Misstatements in AAERs")
    plt.xlabel("Misstatement Type")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(stats_dir / "misstatements_percentage_distribution.png")

    # Number of Misstatements per AAER
    df_aaer["num_misstatements"] = df_aaer[LIST_MISTATEMENT_TYPE_RENAMED].sum(axis=1)
    df_aaer["num_misstatements"].to_csv(stats_dir / "num_misstatements_per_aaer.csv")

    # Plot the distribution of number of misstatements per AAER
    plt.figure(figsize=(12, 6))
    sns.histplot(df_aaer["num_misstatements"], bins=30, kde=True)
    plt.title("Distribution of Number of Misstatements per AAER")
    plt.xlabel("Number of Misstatements")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(stats_dir / "num_misstatements_per_aaer_distribution.png")

    # Cases involving multiple companies  (ie same aaer_no same quarter but multiple ciks)
    df_aaer_multiple_companies = df_aaer[
        df_aaer.duplicated(subset=["aaer_no", "fiscal_quarter"], keep=False)
    ]
    df_aaer_multiple_companies.to_csv(
        stats_dir / "df_aaer_multiple_companies.csv",
        index=False,
    )
    print(f"Number of AAERs with multiple companies: {len(df_aaer_multiple_companies)}")

    # Stats since 2009q1 (Exactley as the AAER dataset)
    misstatements_count_since_2009 = df_aaer_since_2009[
        LIST_MISTATEMENT_TYPE_RENAMED
    ].sum(axis=0)
    misstatements_count_since_2009 = misstatements_count_since_2009.sort_values(
        ascending=False
    )
    misstatements_count_since_2009.to_csv(
        stats_dir / "misstatements_count_since_2009.csv"
    )

    # Plot the distribution of misstatements since 2009
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=misstatements_count_since_2009.index, y=misstatements_count_since_2009.values
    )
    plt.xticks(rotation=90)
    plt.title("Distribution of Misstatements in AAERs since 2009")
    plt.xlabel("Misstatement Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(stats_dir / "misstatements_distribution_since_2009.png")
    plt.close()
    # Distribution of misstatements as percentage since 2009
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=misstatements_count_since_2009.index,
        y=misstatements_count_since_2009.values / len(df_aaer_since_2009) * 100,
    )
    plt.xticks(rotation=90)
    plt.title("Percentage of Misstatements in AAERs since 2009")
    plt.xlabel("Misstatement Type")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(stats_dir / "misstatements_percentage_distribution_since_2009.png")
    plt.close()

    # Number of Misstatements per AAER since 2009
    df_aaer_since_2009["num_misstatements"] = df_aaer_since_2009[
        LIST_MISTATEMENT_TYPE_RENAMED
    ].sum(axis=1)
    df_aaer_since_2009["num_misstatements"].to_csv(
        stats_dir / "num_misstatements_per_aaer_since_2009.csv"
    )
    # Plot the distribution of number of misstatements per AAER since 2009
    plt.figure(figsize=(12, 6))
    sns.histplot(df_aaer_since_2009["num_misstatements"], bins=30, kde=True)
    plt.title("Distribution of Number of Misstatements per AAER since 2009")
    plt.xlabel("Number of Misstatements")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(stats_dir / "num_misstatements_per_aaer_distribution_since_2009.png")
    plt.close()
    # Cases involving multiple companies since 2009 (ie same aaer_no same quarter but multiple ciks)
    df_aaer_multiple_companies_since_2009 = df_aaer_since_2009[
        df_aaer_since_2009.duplicated(subset=["aaer_no", "fiscal_quarter"], keep=False)
    ]
    df_aaer_multiple_companies_since_2009.to_csv(
        stats_dir / "df_aaer_multiple_companies_since_2009.csv",
        index=False,
    )

    print(f"Number of AAERs since 2009: {len(df_aaer_since_2009)}")
    print(
        f"Number of AAERs with multiple companies since 2009: {len(df_aaer_multiple_companies_since_2009)}"
    )

    print(
        f"Preprocessing done. Files saved in {output_dir} and statistics in {stats_dir}."
    )


if __name__ == "__main__":
    preprocess_downloaded_aaer_data()
