import datetime
import json
import logging
import multiprocessing
import re
import sys
import time
import zipfile
from typing import Tuple

from joblib import Parallel, delayed

from researchpkg.anomaly_detection.config import (
    PREPROCESSED_PATH,  # Added, was missing in the original provided 'current code'
)
from researchpkg.anomaly_detection.config import (
    FINANCIALS_DIR_EXTENDED,
    FORM_10K_MDA_ITEM,
    FORM_10Q_MDA_ITEM,
    MDA_DATASET_PATH,
    MDA_EXTRACT_FORMAT,
    SEC_API_AAER_ENDPOINT,
    SEC_API_QUERY_ENDPOINT,
    SEC_API_SAMPLE_SIZE,
    SEC_API_TOKEN,
    SEC_DOWNLOADS_DIR,
    SEC_FINANCIALS_RAW_DATASET_PATH,
)
from researchpkg.utils import configure_logger

time_txt = time.strftime("%Y%m%d-%H%M%S")


from pathlib import Path

import click
import pandas as pd
import ratelimit
import requests
from sec_api import ExtractorApi

# Selenium and BeautifulSoup imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

from researchpkg.anomaly_detection.preprocessing.sec_aaer_content_extractor import (
    extract_aaer_quarters,
)


@click.group()
def cli():
    pass


@click.command()
@click.option("-s", "--start_qtr", default="2009q1")
@click.option("-e", "--end_qtr", default="2024q4")
def download_sec_raw_data(start_qtr: str, end_qtr: str):
    """Downloads and extracts SEC raw data using Selenium and BeautifulSoup."""

    def _qtr_range(startqtr, endqtr):
        """Returns a list of zip file names for calendar qtrs between startqtr & endqtr"""
        yr, endyr = int(startqtr[:4]), int(endqtr[:4])
        return [f"{yr}q{qtr}" for yr in range(yr, endyr + 1) for qtr in range(1, 5)]

    def _download_archive_selenium(qtr, path: Path):
        """Download a single zip archive from SEC.gov using Selenium."""
        _baseurl = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{qtr}.zip"
        url = f"{_baseurl}/{filename}"
        filepath = path / filename  # Define here

        # Selenium setup
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
        chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        chrome_options.add_argument(
            "--disable-dev-shm-usage"
        )  # overcome limited resource problems
        chrome_options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": str(path),
                "download.prompt_for_download": False,
                "plugins.always_open_pdf_externally": True,  # Auto-download PDFs
            },
        )
        driver = webdriver.Chrome(
            options=chrome_options
        )  # Ensure chromedriver is in PATH

        try:
            driver.get(url)
            # Wait for the download to start (adjust timeout as needed)
            # Check if the file has been downloaded
            start_time = time.time()
            while (
                not filepath.exists() and time.time() - start_time < 30
            ):  # Wait for max 30 seconds
                time.sleep(0.1)

            if filepath.exists():
                return "complete"
            else:
                return "Download failed (timeout or file not found)"

        except Exception as e:
            return str(e)
        finally:
            driver.quit()  # Close the browser

    def _extract_archive(path: Path, destination: Path, singledir: bool, rmzip: bool):
        """Extract single archive from zip file."""
        _text_files = [
            "sub.txt",
            "num.txt",
            "pre.txt",
            "tag.txt",
        ]  # Moved inside to be local
        _readme = "readme.htm"  # Moved inside to be local
        zip_re = re.compile(r"\d{4}q[1-4]\.zip")  # Moved inside to be local

        zip_filename = path.name
        qtr = zip_filename.replace(".zip", "")

        # Update quarter_num in filename (it is shifted by -1)
        year = int(qtr[:4])
        quarter_num = int(qtr[5])  # e.g., '1' for 'Q1'

        if quarter_num == 1:
            quarter_num = 4
            year -= 1
        else:
            quarter_num -= 1

        qtr = f"{year}q{quarter_num}"

        destination = destination / qtr if not singledir else destination
        destination.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path) as zip_file:
            if singledir:
                zip_file.extractall(str(destination), _text_files)
                for filename in _text_files:
                    basename = filename.replace(".txt", "")
                    newname = f"{basename}_{qtr}.txt"
                    (destination / filename).rename(destination / newname)

                if not (destination / ("_" + _readme)).exists():
                    zip_file.extract(_readme, str(destination))
                    (destination / _readme).rename(destination / ("_" + _readme))
            else:
                zip_file.extractall(str(destination))

        if rmzip:
            path.unlink()

    def latest_qtr():
        """Get the latest available quarter from SEC.gov.  Uses requests, not Selenium."""
        _baseurl = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"  # Moved inside to be local
        today = datetime.date.today()
        yr, qtr = today.year, (today.month - 1) // 3
        yr, qtr = (yr - 1, 4) if qtr == 0 else (yr, qtr)
        qtr = f"{yr}q{qtr}"

        try:
            response = requests.get(f"{_baseurl}/{qtr}.zip")
            response.raise_for_status()
            return qtr
        except requests.exceptions.RequestException:
            yr, qtr = (yr - 1, 4) if qtr == 1 else (yr, qtr - 1)
            return f"{yr}q{qtr}"

    def download(qtr=None, startqtr="2009q1", endqtr=None, path: Path = None):
        """Download zip archive(s) from SEC.gov using Selenium."""
        _download_path = str(Path.cwd() / "sec_zip")  # Moved inside to be local
        qtr = latest_qtr() if qtr == "latest" else qtr
        path = Path(_download_path) if path is None else path
        path.mkdir(parents=True, exist_ok=True)

        sys.stdout.write(
            f"\nDownloading from https://www.sec.gov/files/dera/data/financial-statement-data-sets/\n"
        )
        sys.stdout.flush()

        if qtr:
            sys.stdout.write(f"{qtr}.zip... ")
            sys.stdout.flush()
            status = _download_archive_selenium(qtr, path)
            print(status, flush=True)
        else:
            print("...")
            endqtr = latest_qtr() if endqtr is None else endqtr
            for qtr in _qtr_range(startqtr, endqtr):
                sys.stdout.write(f"  {qtr}.zip... ")
                sys.stdout.flush()
                status = _download_archive_selenium(qtr, path)
                print(status, flush=True)

    def extract(
        path: Path = None, destination: Path = None, singledir=False, rmzip=False
    ):
        """Extract sec zip archive(s) located at/within path"""
        _download_path = str(Path.cwd() / "sec_zip")  # Moved inside to be local
        path = Path(_download_path) if path is None else path
        destination = Path.cwd() if destination is None else destination

        if path.is_file():
            zip_re = re.compile(r"\d{4}q[1-4]\.zip")  # Moved inside to be local
            if zip_re.match(path.name):
                sys.stdout.write(f"\nExtracting {path}... ")
                sys.stdout.flush()
                _extract_archive(path, destination, singledir, rmzip)
                print("complete")
        elif path.is_dir():
            for zip_file in sorted(path.iterdir()):
                zip_re = re.compile(r"\d{4}q[1-4]\.zip")  # Moved inside to be local
                if zip_re.match(zip_file.name):
                    sys.stdout.write(f"  {zip_file.name}... ")
                    sys.stdout.flush()
                    _extract_archive(zip_file, destination, singledir, rmzip)
                    print("complete", flush=True)
            if rmzip:
                try:
                    path.rmdir()
                except OSError:  # Directory not empty
                    pass

    # ---------------- Main execution block  --------------------
    time_txt = time.strftime("%Y%m%d-%H%M%S")
    configure_logger(
        Path(f"sec_api_downloader::1-download-raw-data{time_txt}.log"),
        logging.INFO,
    )
    logging.info(f"Starting SEC raw data download and extraction")

    download(
        startqtr=start_qtr,
        endqtr=end_qtr,
        path=Path(SEC_FINANCIALS_RAW_DATASET_PATH),
    )
    logging.info("\nExtracting SEC raw data...")
    extract(
        path=Path(SEC_FINANCIALS_RAW_DATASET_PATH),
        destination=Path(SEC_FINANCIALS_RAW_DATASET_PATH),
        singledir=False,
        rmzip=True,
    )

    logging.info(f"Finished SEC raw data download and extraction")


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=10000, period=1)
def download_batch_aaer(offset, download_dir, output_dir, skip_if_exists):
    """
    Download AAER data from SEC-API in batches.
    """
    download_filename = download_dir / f"{offset}-{offset + SEC_API_SAMPLE_SIZE}.json"
    output_filename = output_dir / f"{offset}-{offset + SEC_API_SAMPLE_SIZE}.json"

    if skip_if_exists and download_filename.exists():
        logging.info(
            f"Batch AAER data for offset {offset} already exists. Skipping download."
            " and loading existing data."
        )
        batch_data = json.load(open(download_filename, "r", encoding="utf-8"))
        for aaer_data in batch_data["data"]:
            if "quarter" in aaer_data:
                del aaer_data["quarter"]  # Remove 'quarter' key if exists
            if "is_fiscal_year" in aaer_data:
                del aaer_data["is_fiscal_year"]
        data = batch_data

    else:
        response = requests.post(
            f"{SEC_API_AAER_ENDPOINT}?token={SEC_API_TOKEN}",
            json={
                "query": "dateTime:[1997-01-01 TO 2026-01-01]",
                "from": offset,
                "size": SEC_API_SAMPLE_SIZE,
                "sort": [{"dateTime": {"order": "desc"}}],
            },
        )

        if response.status_code == 429:
            raise Exception("API response: {}".format(response.status_code))
        data = response.json()

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    if len(data["data"]) == 0:
        return f"No more data available at offset {offset}"

    # Check if all aaer already have quarters extracted
    all_aaer_exist = True
    for aaer_data in data["data"]:
        aaer_no = aaer_data.get("aaerNo")
        file = output_dir / f"{aaer_no}.json"
        if not file.exists():
            all_aaer_exist = False
            break
    if all_aaer_exist:
        logging.info(
            f"All AAER data for offset {offset} already processed and exists in {output_dir}. Skipping extraction."
        )
        return f"All AAER data for offset {offset} already processed and exists in {output_dir}. Skipping extraction."

    # Extracrt fiscal years. (This part was truncated in the 'update' code, keeping the original correct one)
    aaer_with_details = extract_aaer_quarters(data["data"])

    for aaer_data in aaer_with_details:
        output_file = f"{aaer_data['aaerNo']}.json"
        output_path = output_dir / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aaer_data, f, indent=4)

    return f"Downloaded {offset}-{offset + SEC_API_SAMPLE_SIZE}"


def process_sec_financial_entry_chunk(entries, output_dir, skip_if_exists=True):
    """Processes a single entry to download and store MDA data."""

    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=5, period=1)
    def get_mda_section_yearly(
        cik, company, filing_10k, num_remaining_retries=5
    ) -> Tuple[str, bool]:
        """
        Retrieves the Management Discussion and Analysis (MDA) section from a 10-K filing for a given CIK and fiscal year.
        Uses SEC API's query API and XBRL to JSON converter API.

        Args:
            cik (str): Central Index Key of the company.
            filing_10k: The 10k filing is the yearly financial record (e.g., '2020Q1' will extract 2020 10-K)
            num_remaining_retries: Number of retries in case of failure not caught by ratelimit (429 due to multithreading and 404)

        Returns:
        Tuple of 2 str!
            1.The content of the Management Discussion and Analysis section, or None if not found.
            2.Bool. True if the NT-10-K section is found or False if not.
        """

        extractor_api = ExtractorApi(SEC_API_TOKEN)

        # Derive the filing year from the filing_10k string (e.g., "2020Q1" -> "2020")
        year = filing_10k[:4]
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        query = {
            "query": {
                "query_string": {
                    "query": f'cik:{cik} AND formType:"10-K" AND periodOfReport:[{start_date} TO {end_date}]',
                    "default_field": "content",
                }
            },
            "from": 0,
            "size": 1,
            "sort": [{"periodOfReport": {"order": "desc"}}],
        }

        try:
            response = requests.post(
                f"{SEC_API_QUERY_ENDPOINT}?token={SEC_API_TOKEN}", json=query
            )

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            query_data = response.json()

            raw_filings = query_data["filings"] or []

            # Filter to avoid NT 10-K filings (non-timely : Explanation of late filing)
            _10k_filings = [f for f in raw_filings if f["formType"] == "10-K"]
            nt_10k_filings = [f for f in raw_filings if f["formType"] == "NT 10-K"]
            section, has_nt_section = None, False
            # Extract the accession number from the query data
            if not _10k_filings:
                logging.critical(
                    f"get_mda_section_yearly:No filings found for {company}, cik={cik} in year: {year}"
                )
            else:
                query_data = _10k_filings[
                    0
                ]  # Taking the latest filing anyway (because of order.desc)
                html_url = query_data["linkToHtml"]

                section = extractor_api.get_section(
                    filing_url=html_url,
                    section=FORM_10K_MDA_ITEM,  # Management Discussion and Analysis"
                    return_type=MDA_EXTRACT_FORMAT,
                )

            # Now we extract the NT-10-K section if it exists
            has_nt_section = len(nt_10k_filings) > 0

            return section, has_nt_section

        except requests.exceptions.RequestException as e:
            if num_remaining_retries > 0:
                time.sleep(2)
                logging.warning(
                    f"Error fetching data for {company}, cik={cik} in year: {year}  - {e}."
                    f"Retrying... (Remaining retries: {num_remaining_retries})"
                )
                return get_mda_section_yearly(
                    cik, company, filing_10k, num_remaining_retries - 1
                )

            logging.error(
                f"Error fetching data for  {company}, cik={cik} in year: {year}  - {e}."
                f"(Remaining retries: {num_remaining_retries})"
            )
            return None, False
        except Exception as e:
            if num_remaining_retries > 0:
                time.sleep(2)
                logging.warning(
                    f"Unexpected Error processing data for {company}, cik={cik} in year: {year}  - {e}."
                    f"Retrying... (Remaining retries: {num_remaining_retries})"
                )
                return get_mda_section_yearly(
                    cik, company, filing_10k, num_remaining_retries - 1
                )
            logging.error(
                f"Unexpected Error processing data for {company}, cik={cik} in year: {year}  - {e}."
                f"(Remaining retries: {num_remaining_retries})"
            )
            return None, False

    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=10, period=1)
    def get_mda_section_quarterly(
        cik, company, filing_quarter, fiscal_period, num_remaining_retries=5
    ) -> Tuple[str, bool]:
        """
        Retrieves the Management Discussion and Analysis (MDA) section from a 10-Q filing for a given CIK and fiscal quarter.
        Uses SEC API's query API and XBRL to JSON converter API.

        Args:
            cik (str): Central Index Key of the company.
            filing_quarter (str): The fiscal quarter of the filing in the format "YYYYQn".
            fiscal_period (str): The fiscal period end date in YYYYMMDD format.

        Returns:
            Tuple of String and a boolean: The content of the Management Discussion and Analysis section, or None if not found and a Bool indicates if the NT filing has been found for the given year/quarter.
        """
        # Format of fiscal_period is YYYYMMDD
        fiscal_period = str(fiscal_period)
        year = fiscal_period[:4]
        month = fiscal_period[4:6]
        day = fiscal_period[6:8]

        end_date = f"{year}-{month}-{day}"

        start_month = int(month) - 1
        if (
            start_month == 0
        ):  # If it's Jan (01), start_month becomes 0, so go to Dec of previous year
            start_month = 12
            start_year = int(year) - 1
        else:
            start_year = int(year)

        if start_month < 10:
            start_month = f"0{start_month}"
        start_date = f"{start_year}-{start_month}-01"

        query = {
            "query": {
                "query_string": {
                    "query": f'cik:{cik} AND formType:"10-Q" AND periodOfReport:[{start_date} TO {end_date}]',
                    "default_field": "content",
                }
            },
            "from": 0,
            "size": 1,
            "sort": [{"periodOfReport": {"order": "desc"}}],
        }

        try:
            response = requests.post(
                f"{SEC_API_QUERY_ENDPOINT}?token={SEC_API_TOKEN}", json=query
            )

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            query_data = response.json()

            filings = query_data["filings"] or []
            # We filter the 10-Q filings because the response can contain NT 10-Q filings (non-timely : Explanation of late filing)
            _10_q_filings = [f for f in filings if f["formType"] == "10-Q"]
            nt_10q_filings = [f for f in filings if f["formType"] == "NT 10-Q"]

            # Extract the accession number from the query data
            section, has_nt_section = None, False

            if not _10_q_filings:
                logging.critical(
                    f"get_mda_section_quarterly:No filings found for {company}, "
                    f"cik={cik} in quarter : {filing_quarter}, "
                    f"period : {fiscal_period}"
                )
            else:

                query_data = _10_q_filings[
                    0
                ]  # Taking the latest filing anyway (because of order.desc)

                html_url = query_data["linkToHtml"]
                extractor_api = ExtractorApi(SEC_API_TOKEN)
                section = extractor_api.get_section(
                    filing_url=html_url,
                    section=FORM_10Q_MDA_ITEM,  # Management Discussion and Analysis"
                    return_type=MDA_EXTRACT_FORMAT,
                )
                if len(section) == 0:
                    logging.warning(
                        f"get_mda_section_quarterly:Empty section for {company}, "
                        f"cik={cik} in quarter : {filing_quarter}, "
                        f"period : {fiscal_period}"
                    )
                    section = None

            has_nt_section = len(nt_10q_filings) > 0

            return section, has_nt_section

        except requests.exceptions.RequestException as e:

            if num_remaining_retries > 0:
                time.sleep(2)
                logging.warning(
                    f"get_mda_section_quarterly:Error fetching data for {company}, cik={cik}, quarter:{filing_quarter}  - {e}. "
                    f"Retrying... (Remaining retries: {num_remaining_retries})"
                )
                return get_mda_section_quarterly(
                    cik,
                    company,
                    filing_quarter,
                    fiscal_period,
                    num_remaining_retries - 1,
                )

            logging.critical(
                f"get_mda_section_quarterly:Error fetching data for {company}, {cik}={cik}, quarter:{filing_quarter}  - {e}. "
                f"(Remaining retries: {num_remaining_retries})"
            )
            return None, False
        except Exception as e:

            if num_remaining_retries > 0:
                time.sleep(2)
                logging.warning(
                    f"get_mda_section_quarterly:Unexpected Error processing data for {company}, cik={cik}, quarter:{filing_quarter}  - {e}. "
                    f"Retrying... (Remaining retries: {num_remaining_retries})"
                )
                return get_mda_section_quarterly(
                    cik,
                    company,
                    filing_quarter,
                    fiscal_period,
                    num_remaining_retries - 1,
                )

            logging.critical(
                f"get_mda_section_quarterly:Unexpected Error processing data for {company}, cik={cik}, quarter:{filing_quarter}  - {e}. "
                f"(Remaining retries: {num_remaining_retries})"
            )
            return None, False

    output_dir = Path(output_dir)
    for entry in tqdm(entries, "Processing SEC financial entries"):
        mda_quarterly_dir = output_dir / "quarterly"
        mda_quarterly_dir.mkdir(parents=True, exist_ok=True)
        mda_yearly_dir = (
            output_dir / "yearly"
        )  # Yearly dir is still created but not used for downloads.
        mda_yearly_dir.mkdir(parents=True, exist_ok=True)

        company = entry["company"]
        cik = entry["cik"]
        quarter = entry["fiscal_quarter"]  # Now a single YYYYQn string
        fiscal_period = entry["period"]  # Now a YYYYMMDD string

        # Initialize for MDA data, yearly is no longer downloaded in this function
        mda_data_quarterly = {}
        mda_yearly_content, has_nt_mda_yearly = (
            None,
            False,
        )  # Yearly MDA not downloaded here

        year = quarter[:4]

        # Check if the quarterly entry already exists
        mda_quarterly_file = mda_quarterly_dir / f"{cik}_{quarter}.txt"
        if mda_quarterly_file.exists() and skip_if_exists:
            logging.info(f"Entry {cik}, {quarter} already exists. Skipping...")
            has_nt_mda_quarterly = False
            mda_quarterly_content = "OK"  # Dont use empty to pass the check
        else:
            # Download quarterly MDA
            mda_quarterly_content, has_nt_mda_quarterly = get_mda_section_quarterly(
                cik, company, quarter, fiscal_period  # Pass fiscal_period
            )

            mda_data_quarterly[quarter] = (
                mda_quarterly_content if mda_quarterly_content else []
            )

            # mda_yearly_file remains None as it's not downloaded here.
            if mda_quarterly_content:
                with open(mda_quarterly_file, "w", encoding="utf-8") as f:
                    f.write(mda_quarterly_content)

        # Append the row to the CSV file
        with open(output_dir / "mda_ref_quarterly.csv", "a") as f:
            data = [
                entry["company"],
                entry["cik"],
                year,
                quarter,
                None,  # mda_year_id is None since yearly MDA is not downloaded here
                mda_quarterly_file.stem if mda_quarterly_content else None,
                False,  # has_nt_mda_year is False since yearly MDA is not downloaded here
                has_nt_mda_quarterly,
            ]
            f.write("|".join(map(str, data)) + "\n")

        logging.info(
            f"Successfully processed {company}:{cik}, quarter: {quarter}, fiscal_period: {fiscal_period}"
        )


def download_batch_mda_from_sec_financials(
    data_index: list[dict],
    output_dir: Path,
    num_workers: int = 18,
    skip_if_exists: bool = True,
):
    """
    Download MDA data from SEC-API in parallel
    """

    columns = [
        "company",
        "cik",
        "year",
        "quarter",
        "mda_year_id",
        "mda_quarter_id",
        "has_nt_mda_year",  # Non time mda data yearly boolean
        "has_nt_mda_quarter",  # Non time mda data quarterly boolean
    ]

    if not (output_dir / "mda_ref_quarterly.csv").exists():
        with open(output_dir / "mda_ref_quarterly.csv", "w") as f:
            # Initialize the CSV file with the columns
            f.write("|".join(columns) + "\n")

    else:
        # Filter data_index to remove already processed entries
        # Note: This only checks cik and quarter, not fiscal_period
        processed_df = pd.read_csv(
            output_dir / "mda_ref_quarterly.csv",
            sep="|",
            dtype={"cik": str},
            usecols=["cik", "quarter"],
        ).drop_duplicates()

        processed_index = set(
            [
                (r["cik"], r["quarter"])
                for r in tqdm(
                    processed_df.to_dict(orient="records"), "reading processed index"
                )
            ]
        )

        old_size = len(data_index)
        data_index = [
            entry
            for entry in tqdm(
                data_index, "Filtering data index to remove already processed"
            )
            if (entry["cik"], entry["fiscal_quarter"]) not in processed_index
        ]

        print(f"Data index size before : {old_size}, after : {len(data_index)}")

    chunk_size = len(data_index) // num_workers
    chunks = [
        data_index[i : i + chunk_size] for i in range(0, len(data_index), chunk_size)
    ]

    # use job lib
    Parallel(n_jobs=num_workers, prefer="threads")(
        delayed(process_sec_financial_entry_chunk)(chunk, output_dir)
        for chunk in chunks
    )

    # Create excel from the csv file
    df = pd.read_csv(output_dir / "mda_ref_quarterly.csv", sep="|")
    df.sort_values(by=["year", "quarter", "cik"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_dir / "mda_ref_quarterly.csv", index=False, sep="|")
    df.to_excel(output_dir / "mda_ref_quarterly.xlsx", index=False)

    # Create a mdaref yearly (this will now primarily contain None for mda_year_id and False for has_nt_mda_year)
    df_yearly = df[
        ["company", "cik", "year", "mda_year_id", "has_nt_mda_year"]
    ].drop_duplicates()
    df_yearly.to_csv(output_dir / "mda_ref_yearly.csv", index=False, sep="|")
    df_yearly.to_excel(output_dir / "mda_ref_yearly.xlsx", index=False)


@click.command()
@click.option(
    "-o", "--override", is_flag=True, default=False, help="override existing files."
)
@click.option(
    "--num-jobs", default=1, type=int, help="Number of parallel download jobs."
)
def download_sec_aaer(override, num_jobs):
    """
    Download AAER data from SEC-API in batches of SEC_API_SAMPLE_SIZE.
    Continues until no more data is available.
    """
    configure_logger(
        Path(f"sec_api_downloader::1-download-sec-aaer{time_txt}.log"), logging.INFO
    )
    logging.info(f"Downloading SEC AAER datas using {num_jobs} concurrent jobs.log")

    download_dir = SEC_DOWNLOADS_DIR / "AAER"
    download_dir.mkdir(parents=True, exist_ok=True)

    output_dir = PREPROCESSED_PATH / "AAER_PREPROCESSED" / "AAER_DETAILS"
    output_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    futures = []
    stop_flag = False
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        while not stop_flag:
            logging.info(
                f"Extracting AAER data from offset {offset} to {offset + SEC_API_SAMPLE_SIZE}"
            )
            future = executor.submit(
                download_batch_aaer, offset, download_dir, output_dir, not override
            )
            futures.append(future)
            offset += SEC_API_SAMPLE_SIZE

            if len(futures) >= num_jobs:
                for future in as_completed(futures):
                    result = future.result()
                    click.echo(result)
                    if "No more data available" in result:
                        stop_flag = True
                        break
                futures = []

            # Wait 0.1 second to cooldown
            time.sleep(0.1)

    # Process any remaining futures
    for future in as_completed(futures):
        click.echo(future.result())

    click.echo(f"Successfully processed up to offset {offset}")


@click.command()
def download_sec_mda_from_financial_ref():
    """
    Downloads MDA data from SEC-API based on a financial reference CSV.
    Handles civil calendar matches by copying existing files.
    """
    configure_logger(
        Path(f"sec_api_downloader::2-download-sec-mda_{time_txt}.log"), logging.INFO
    )
    logging.info("Starting SEC MDA download from financial reference.")

    def load_and_prepare_sec_financials(sec_financials_dir: Path):
        """
        Load sec_financial index and prepare data for processing.
        """
        quarterly_file = sec_financials_dir / "sec_financials_quarterly.csv"

        assert (
            quarterly_file.exists()
        ), f"{quarterly_file} not found. First run. `python sec_financial_preprocessing_quarterly.py`"

        df = pd.read_csv(
            quarterly_file,
            usecols=["cik", "year", "quarter", "company", "period"],
            dtype={
                "cik": str,
                "year": str,
                "quarter": str,
                "company": str,
                "period": str,
            },
        )
        df_data = df[["company", "cik", "year", "quarter", "period"]]
        # sort by year
        df_data.sort_values(by=["year"], inplace=True, ascending=False)

        # Ensure fiscal_quarter is in YYYYQn format (e.g., "2020Q1")
        df_data["fiscal_quarter"] = df_data.apply(
            lambda x: f"{x['year']}{x['quarter']}", axis=1
        )
        df_data = df_data.drop_duplicates().dropna(subset=["fiscal_quarter"])
        return df_data.to_dict(orient="records")

    data_index = load_and_prepare_sec_financials(FINANCIALS_DIR_EXTENDED)

    logging.info(f"Data index size: {len(data_index)}")

    target_dir = MDA_DATASET_PATH / "quarterly"  # Use MDA_DATASET_PATH as the base
    target_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (
        MDA_DATASET_PATH  # This is passed to download_batch_mda_from_sec_financials
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    download_batch_mda_from_sec_financials(data_index, output_dir)
    logging.info("Finished SEC MDA download from financial reference.")


cli.add_command(download_sec_raw_data, name="1-download-sec-raw")
cli.add_command(download_sec_aaer, name="2-download-sec-aaer")
cli.add_command(download_sec_mda_from_financial_ref, name="3-download-sec-mda")

if __name__ == "__main__":
    cli()
    # download_sec_mda_from_financial_ref()
