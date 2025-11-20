# import datetime
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Union

# import tqdm
# from joblib import Parallel, delayed

# from researchpkg.anomaly_detection.config import MAX_CORE_USAGE, MDA_DATASET_PATH


# @dataclass
# class MDA_descriptor:
#     company_name: str
#     cik: str
#     sic: str
#     date: datetime
#     file_path: Path
#     sections: Union[List[str], None] = None

#     @property
#     def year(self) -> str:
#         return str(self.date.year)

#     def extract_sections(self):

#         # 1. Reparse the file content
#         descriptor = list(
#             filter(
#                 lambda x: x.cik == self.cik and x.date == self.date,
#                 parse_single_mda_file(self.file_path, extract_sections=True),
#             )
#         )
#         assert (
#             len(descriptor) == 1
#         ), f"Error. Descriptor for {self.cik} and {self.date} not found"
#         self.sections = descriptor[0].sections


# def parse_single_mda_file(
#     filepath: Path, min_year: int, max_year: int, extract_sections: bool = False
# ) -> List[MDA_descriptor]:
#     """
#     Parse an input file containing multiple MDA sections
#     :param filepath: The path to the file to parse
#     :param min_year: The first year to consider in the dataset
#     :param extract_sections: If True, extract the sections of the report.
#     Prefer false because sections are huge amount of text.
#     and return a list of MDA_descriptor objects.
#     """

#     # 1. Read the file content:
#     with open(filepath, "r") as file:
#         try:
#             file_content = file.read()
#         except UnicodeDecodeError:
#             print(f"Error reading file {filepath}")
#             return []

#     reports = [
#         f"<HEADER>{report}"
#         for report in file_content.split("<HEADER>")
#         if report.strip()
#     ]

#     # 2. Parse the content:
#     mda_descriptors = []
#     for report in reports:
#         header, *sections = report.split("<SECTION>")
#         if extract_sections:
#             sections = [
#                 section.replace("</SECTION>", "").strip()
#                 for section in sections
#                 if section.strip()
#             ]
#         else:
#             sections = None

#         # 2.1 Parse the header:
#         header_lines = header.strip().replace("<HEADER>", "").strip().split("\n")

#         company_name = header_lines[0].split(":")[1].strip()
#         cik = header_lines[1].split(":")[1].strip()
#         sic = header_lines[2].split(":")[1].strip()

#         # Date a the row 4 : REPORT PERIOD END DATE: 20160531
#         date = datetime.datetime.strptime(
#             header_lines[4].split(":")[1].strip(), "%Y%m%d"
#         )

#         if date.year < min_year or date.year > max_year:
#             continue

#         # Parse the sections of the report:
#         mda_descriptors.append(
#             MDA_descriptor(company_name, cik, sic, date, filepath, sections)
#         )

#     return mda_descriptors


# def parser_multiple_mda_files(
#     files: List[Path], min_year: int, max_year: int
# ) -> List[MDA_descriptor]:
#     all_descriptors = []
#     for file in files:
#         all_descriptors.extend(parse_single_mda_file(file, min_year, max_year))
#     return all_descriptors


# def extract_all_mda_descriptors(
#     root_dir: str, min_year: int, max_year: int, max_core: int = MAX_CORE_USAGE
# ) -> List[MDA_descriptor]:
#     """
#     Extract all MDA descriptors from all files in the root directory
#     :param root_dir: The root directory containing the MDA files
#     :param min_year: The first year to consider in the dataset
#     :param max_year: The last year to consider in the dataset
#     :return: A list of MDA_descriptor objects
#     """

#     all_files = list(Path(root_dir).rglob("*.txt"))
#     all_files = list(filter(lambda x: "__MACOSX" not in str(x), all_files))

#     njobs = min(max_core, len(all_files))
#     print(
#         f"Extracting MDA descriptors from {len(all_files)} files using {njobs} concurrent jobs"
#     )
#     outputs = Parallel(n_jobs=njobs)(
#         delayed(parse_single_mda_file)(file, min_year, max_year)
#         for file in tqdm.tqdm(all_files, "Extracting MDA descriptors")
#     )

#     return [descriptor for output in outputs for descriptor in output]


# if __name__ == "__main__":
#     mda_descriptors = extract_all_mda_descriptors(MDA_DATASET_PATH, 2009, 2014)
#     print(f"Extracted {len(mda_descriptors)} MDA descriptors")
