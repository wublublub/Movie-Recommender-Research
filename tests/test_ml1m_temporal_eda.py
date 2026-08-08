from pathlib import Path

import pandas as pd
import pytest

from movie_recommender.ml1m_data_loader import (
    load_ml1m_movies,
    load_ml1m_ratings,
    load_ml1m_users,
)
from movie_recommender.ml1m_temporal_eda import (
    build_genre_evolution_features,
    build_user_temporal_features,
    validate_dataset,
)


def test_ml1m_loaders_parse_multicharacter_delimiter(tmp_path: Path) -> None:
    (tmp_path / "ratings.dat").write_text("1::10::5::0\n", encoding="latin-1")
    (tmp_path / "movies.dat").write_text(
        "10::Example (2000)::Drama|Comedy\n", encoding="latin-1"
    )
    (tmp_path / "users.dat").write_text(
        "1::F::25::1::12345\n", encoding="latin-1"
    )

    assert list(load_ml1m_ratings(tmp_path).columns) == [
        "UserID",
        "MovieID",
        "Rating",
        "Timestamp",
    ]
    assert load_ml1m_movies(tmp_path).iloc[0]["Title"] == "Example (2000)"
    assert load_ml1m_users(tmp_path).iloc[0]["ZipCode"] == 12345


def test_temporal_features_handle_ties_and_strict_session_threshold() -> None:
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 1, 1],
            "MovieID": [10, 11, 12, 13],
            "Rating": [5, 4, 3, 2],
            "Timestamp": [0, 0, 1800, 3601],
        }
    )
    ratings["RatingTime"] = pd.to_datetime(ratings["Timestamp"], unit="s", utc=True)

    features = build_user_temporal_features(ratings).iloc[0]

    assert features["minimum_gap_minutes"] == 0
    assert features["session_count_30m"] == 2
    assert features["maximum_session_size_30m"] == 3
    assert features["maximum_ratings_in_one_hour"] == 3


def test_odd_history_assigns_middle_rating_to_early_half() -> None:
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 1],
            "MovieID": [10, 11, 12],
            "Rating": [5, 5, 5],
            "Timestamp": [1, 2, 3],
        }
    )
    ratings["RatingTime"] = pd.to_datetime(ratings["Timestamp"], unit="s", utc=True)
    movies = pd.DataFrame(
        {
            "MovieID": [10, 11, 12],
            "Title": ["A", "B", "C"],
            "Genres": ["Drama", "Drama", "Comedy"],
        }
    )

    features = build_genre_evolution_features(ratings, movies).iloc[0]

    assert features["genre_js_divergence"] == pytest.approx(1.0)
    assert features["early_genre_entropy"] == pytest.approx(0.0)
    assert features["late_genre_entropy"] == pytest.approx(0.0)


def test_validation_rejects_invalid_timestamp() -> None:
    ratings = pd.DataFrame(
        {"UserID": [1], "MovieID": [10], "Rating": [5], "Timestamp": ["bad"]}
    )
    movies = pd.DataFrame({"MovieID": [10], "Title": ["A"], "Genres": ["Drama"]})
    users = pd.DataFrame(
        {"UserID": [1], "Gender": ["F"], "Age": [25], "Occupation": [1], "ZipCode": ["12345"]}
    )

    with pytest.raises(ValueError, match="invalid timestamps"):
        validate_dataset(ratings, movies, users)
