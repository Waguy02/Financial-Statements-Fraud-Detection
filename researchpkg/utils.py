import logging
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd  # Added pandas import
import torch  # Added PyTorch import
from tqdm import tqdm  # Added tqdm import


def to_snake_case_remove_accents(input_string):
    """
    Converts a string to snake_case and removes accents.

    Args:
        input_string (str): The input string to convert.

    Returns:
        str: The processed string in snake_case without accents.
    """
    # Normalize the string to decompose accents
    normalized_string = unicodedata.normalize("NFKD", input_string)
    # Remove accents by filtering out non-ASCII characters
    without_accents = "".join(
        char for char in normalized_string if not unicodedata.combining(char)
    )
    # Replace non-alphanumeric characters with underscores
    no_special_chars = re.sub(r"[^a-zA-Z0-9]", "_", without_accents)
    # Convert to lowercase
    snake_case = re.sub(r"__+", "_", no_special_chars).strip("_").lower()
    return snake_case


def format_feature_name(feature_name):
    """
    Formats the feature name by removing special characters, accents,
    replacing spaces with underscores, and converting to lowercase.

    Args:
        feature_name (str): The original feature name.

    Returns:
        str: The formatted feature name.
    """

    feature_name = feature_name.replace(".", "_")
    # Normalize the text to decompose accents and special characters
    name = unicodedata.normalize("NFD", feature_name)

    # Remove accents by filtering out characters with a combining mark
    name = "".join([c for c in name if not unicodedata.combining(c)])

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove any special characters (except underscores) using regex
    name = re.sub(r"[^A-Za-z0-9_]", "", name)

    # Convert to lowercase
    name = name.lower()

    return name


def configure_logger(logFile, logLevel=logging.INFO):
    """
    Configures a logger to write to a file and the console.
    Removes any existing handlers from the root logger before configuration.
    """
    logFile.parent.mkdir(parents=True, exist_ok=True)  # Ensure the log directory exists

    # Remove existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:  # Iterate over a copy of the handlers list
        root_logger.removeHandler(handler)
        handler.close()  # Ensure handler resources are released

    # Clear any existing log levels on the root logger
    root_logger.setLevel(logging.NOTSET)

    # Configure the logger
    logging.basicConfig(
        level=logLevel,
        format="%(asctime)s  [%(levelname)s] - %(funcName)s () ==>  %(message)s",
        handlers=[
            logging.FileHandler(logFile),
            logging.StreamHandler(),  # Output to console
        ],
    )
    # logging.info("Logger configured.")


def reset_logger():
    """
    Resets the root logger to its default state. This is helpful for removing
    all custom handlers and configurations.
    """
    root_logger = logging.getLogger()

    # Remove all handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Reset level
    root_logger.setLevel(
        logging.WARNING
    )  # or logging.NOTSET if you really want to clear everything


def numpy_to_scalar(data: Union[dict, np.ndarray, str, list]):
    """
    Recursively convert numpy arrays in a dictionary to scalars.
    """
    if isinstance(data, dict):
        return {key: numpy_to_scalar(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, list):
        return [numpy_to_scalar(item) for item in data]
    elif isinstance(data, str):
        return data
    else:
        # print("Unknown type:", data,":", type(data),)
        return data


def torch_to_scalar(data: Union[dict, torch.Tensor, np.ndarray, str, list]):
    """
    Recursively convert PyTorch tensors in a dictionary or list to Python scalars.

    Args:
        data (Union[dict, torch.Tensor, np.ndarray, str, list]): The data to convert.
            Can be a dictionary, PyTorch tensor, numpy array, string, list, or other types.

    Returns:
        The input data with any PyTorch tensors converted to Python scalars.
    """
    if isinstance(data, dict):
        return {key: torch_to_scalar(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, list):
        return [torch_to_scalar(item) for item in data]
    elif isinstance(data, str):
        return data
    else:
        return data


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


import time

# Timer context manager
from contextlib import contextmanager


@contextmanager
def TimeCounter(name: str, logLevel=logging.INFO):
    start = time.time()
    yield
    end = time.time()
    logging.log(logLevel, f"{name} took {end - start} seconds")


class LLMCache(dict):
    """Sparse row store of sentence embeddings keyed by text hash."""

    def __init__(self, shape: Tuple[int, ...]):
        self.hash_to_idx: Dict[str, int] = {}
        self.n_dimensions = len(shape) + 1
        self.shape = shape
        self.storage = torch.sparse_coo_tensor(
            indices=torch.empty((self.n_dimensions, 0), dtype=torch.int64),
            values=torch.tensor([]),
            size=(0, *shape),
        )

    def __setitem__(self, key, value):
        if key in self.hash_to_idx:
            raise NotImplementedError(
                "Update of existing sparse tensor rows not supported"
            )
        dense_vec = value
        non_zero_indices = torch.nonzero(dense_vec, as_tuple=False).t()
        non_zero_values = dense_vec[dense_vec != 0]
        sparse_vec = torch.sparse_coo_tensor(
            indices=non_zero_indices,
            values=non_zero_values,
            size=dense_vec.shape,
        )
        self.storage = self._append_sparse_vector(self.storage, sparse_vec)
        idx = self.storage.shape[0] - 1
        self.hash_to_idx[key] = idx

    def __getitem__(self, key):
        idx = self.hash_to_idx.get(key, None)
        if idx is None:
            raise KeyError(f"Key {key} not found in cache")
        return self._get_sparse_tensor_at_index(self.storage, idx)

    def _append_sparse_vector(self, storage, vec):
        if storage.shape[0] == 0:
            return vec.unsqueeze(0)
        return torch.cat([storage, vec.unsqueeze(0)], dim=0)

    def _get_sparse_tensor_at_index(self, storage, index):
        subtensor = storage.select(dim=0, index=index).coalesce()
        dense_shape = subtensor.size()
        dense_tensor = torch.zeros(dense_shape, dtype=subtensor.values().dtype)
        indices = subtensor.indices()
        values = subtensor.values()
        coord_tuple = tuple(indices[i] for i in range(indices.size(0)))
        dense_tensor[coord_tuple] = values
        return dense_tensor

    def __contains__(self, key) -> bool:
        return key in self.hash_to_idx

    def __len__(self) -> int:
        return len(self.hash_to_idx)

    def clear(self):
        self.hash_to_idx.clear()        
      
        
        self.storage = torch.sparse_coo_tensor(
        indices=torch.empty((self.n_dimensions, 0), dtype=torch.int64),
        values=torch.tensor([]),
        size=(0, *self.shape),
    )
