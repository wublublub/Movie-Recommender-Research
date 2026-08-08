from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from movie_recommender.data_loader import (
    DEFAULT_DATA_DIR,
    load_links,
    load_movies,
    load_ratings,
    load_tags,
)


Loader = Callable[[Path | str], pd.DataFrame]

DATASETS: tuple[tuple[str, Loader, set[str]], ...] = (
    ("ratings.csv", load_ratings, {"userId", "movieId", "rating", "timestamp"}),
    ("movies.csv", load_movies, {"movieId", "title", "genres"}),
    ("tags.csv", load_tags, {"userId", "movieId", "tag", "timestamp"}),
    ("links.csv", load_links, {"movieId", "imdbId", "tmdbId"}),
)


@pytest.mark.parametrize(("filename", "loader", "expected_columns"), DATASETS)
def test_dataset_exists(
    filename: str, loader: Loader, expected_columns: set[str]
) -> None:
    assert (DEFAULT_DATA_DIR / filename).is_file()


@pytest.mark.parametrize(("filename", "loader", "expected_columns"), DATASETS)
def test_loader_returns_nonempty_dataframe_with_expected_columns(
    filename: str, loader: Loader, expected_columns: set[str]
) -> None:
    dataframe = loader(DEFAULT_DATA_DIR)

    assert isinstance(dataframe, pd.DataFrame)
    assert expected_columns.issubset(dataframe.columns)
    assert len(dataframe) > 0
