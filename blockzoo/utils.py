"""Utility functions for BlockZoo framework.

This module provides helper functions for dynamic imports, CSV result handling,
and formatting utilities used throughout the BlockZoo framework.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def append_results(path: str, data: Dict[str, Any]) -> None:
    """
    Append a row of results to a CSV file, creating directories and headers as needed.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    data : dict
        Dictionary containing the data to append as a row.

    Notes
    -----
    If the file doesn't exist, it will be created with appropriate headers.
    Missing directories in the path will be created automatically.
    """
    file_path = Path(path)

    # create directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # check if file exists to determine if we need headers
    file_exists = file_path.exists()

    # convert data to ensure all values are serializable
    serialized_data = {}
    for key, value in data.items():
        if value is None:
            serialized_data[key] = ""
        elif isinstance(value, (int, float, str, bool)):
            serialized_data[key] = value
        else:
            serialized_data[key] = str(value)

    # write to CSV
    with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=serialized_data.keys())

        # write headers if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(serialized_data)


def format_bytes(num_bytes: int) -> str:
    """
    Format a byte count into a human-readable string.

    Parameters
    ----------
    num_bytes : int
        Number of bytes to format.

    Returns
    -------
    str
        Human-readable byte count (e.g., '1.5 GB', '256 MB').

    Examples
    --------
    >>> format_bytes(1024)
    '1.0 KB'
    >>> format_bytes(1073741824)
    '1.0 GB'
    """
    if num_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0

    return f"{num_bytes:.1f} PB"


def load_results(path: str) -> Optional[pd.DataFrame]:
    """
    Load results from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing the results, or None if file doesn't exist.
    """
    file_path = Path(path)

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"[BlockZoo] Warning: Could not load results from {path}: {e}")
        return None


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Path to the directory.

    Returns
    -------
    pathlib.Path
        Path object for the directory.

    Examples
    --------
    >>> results_dir = ensure_directory('results')
    >>> print(results_dir.exists())
    True
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
