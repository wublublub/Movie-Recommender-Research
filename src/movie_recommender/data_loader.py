"""Load the raw MovieLens Latest Small CSV files."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-latest-small"
)


def _load_csv(filename: str, data_dir: Path | str) -> pd.DataFrame:
    """Load one MovieLens CSV file.

    Args:
        filename: Name of the CSV file to load.
        data_dir: Directory containing the extracted MovieLens files.

    Returns:
        A DataFrame containing the unmodified CSV contents.

    Raises:
        FileNotFoundError: If the data directory or requested CSV does not exist.
        NotADirectoryError: If ``data_dir`` is not a directory.
    """
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"MovieLens data directory does not exist: {directory.resolve()}"
        )
    if not directory.is_dir():
        raise NotADirectoryError(
            f"MovieLens data path is not a directory: {directory.resolve()}"
        )

    file_path = directory / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"MovieLens CSV file does not exist: {file_path}")

    return pd.read_csv(file_path)


def load_ratings(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens user ratings.

    Args:
        data_dir: Directory containing the extracted MovieLens files.

    Returns:
        The contents of ``ratings.csv`` as a DataFrame.
    """
    return _load_csv("ratings.csv", data_dir)


def load_movies(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens movie metadata.

    Args:
        data_dir: Directory containing the extracted MovieLens files.

    Returns:
        The contents of ``movies.csv`` as a DataFrame.
    """
    return _load_csv("movies.csv", data_dir)


def load_tags(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens user-generated tags.

    Args:
        data_dir: Directory containing the extracted MovieLens files.

    Returns:
        The contents of ``tags.csv`` as a DataFrame.
    """
    return _load_csv("tags.csv", data_dir)


def load_links(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens external identifier links.

    Args:
        data_dir: Directory containing the extracted MovieLens files.

    Returns:
        The contents of ``links.csv`` as a DataFrame.
    """
    return _load_csv("links.csv", data_dir)
