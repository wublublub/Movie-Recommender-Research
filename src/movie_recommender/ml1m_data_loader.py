"""Load the raw MovieLens 1M DAT files."""

from pathlib import Path

import pandas as pd


DEFAULT_ML1M_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-1m"

RATINGS_COLUMNS = ("UserID", "MovieID", "Rating", "Timestamp")
MOVIES_COLUMNS = ("MovieID", "Title", "Genres")
USERS_COLUMNS = ("UserID", "Gender", "Age", "Occupation", "ZipCode")


def _load_dat(
    filename: str,
    columns: tuple[str, ...],
    data_dir: Path | str,
) -> pd.DataFrame:
    """Load one MovieLens 1M file with the multi-character delimiter.

    Args:
        filename: Name of the DAT file to load.
        columns: Expected column names in file order.
        data_dir: Directory containing the MovieLens 1M files.

    Returns:
        The parsed file contents.

    Raises:
        FileNotFoundError: If the data directory or requested file is missing.
        NotADirectoryError: If the supplied data path is not a directory.
        ValueError: If the parsed data is empty or malformed.
    """
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"MovieLens 1M data directory does not exist: {directory.resolve()}"
        )
    if not directory.is_dir():
        raise NotADirectoryError(
            f"MovieLens 1M data path is not a directory: {directory.resolve()}"
        )

    file_path = directory / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"MovieLens 1M file does not exist: {file_path}")

    try:
        frame = pd.read_csv(
            file_path,
            sep=r"::",
            engine="python",
            names=list(columns),
            header=None,
            encoding="latin-1",
        )
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not parse MovieLens 1M file {file_path}: {error}") from error

    if frame.empty:
        raise ValueError(f"MovieLens 1M file is empty: {file_path}")
    if tuple(frame.columns) != columns:
        raise ValueError(
            f"Malformed columns in {file_path}; expected {columns}, "
            f"received {tuple(frame.columns)}"
        )
    return frame


def load_ml1m_ratings(
    data_dir: Path | str = DEFAULT_ML1M_DATA_DIR,
) -> pd.DataFrame:
    """Load MovieLens 1M rating interactions.

    Args:
        data_dir: Directory containing the MovieLens 1M files.

    Returns:
        The contents of ``ratings.dat``.
    """
    return _load_dat("ratings.dat", RATINGS_COLUMNS, data_dir)


def load_ml1m_movies(
    data_dir: Path | str = DEFAULT_ML1M_DATA_DIR,
) -> pd.DataFrame:
    """Load MovieLens 1M movie metadata.

    Args:
        data_dir: Directory containing the MovieLens 1M files.

    Returns:
        The contents of ``movies.dat``.
    """
    return _load_dat("movies.dat", MOVIES_COLUMNS, data_dir)


def load_ml1m_users(
    data_dir: Path | str = DEFAULT_ML1M_DATA_DIR,
) -> pd.DataFrame:
    """Load MovieLens 1M user metadata.

    Args:
        data_dir: Directory containing the MovieLens 1M files.

    Returns:
        The contents of ``users.dat``.
    """
    return _load_dat("users.dat", USERS_COLUMNS, data_dir)
