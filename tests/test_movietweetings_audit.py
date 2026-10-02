from pathlib import Path

import pandas as pd
import pytest

from movie_recommender.movietweetings_audit import (
    audit,
    load_movies,
    load_ratings,
    split_audit,
)


def ratings(count=9):
    return pd.DataFrame({
        "UserID": [1] * count,
        "MovieID": range(1, count + 1),
        "Rating": [8] * count,
        "Timestamp": range(1_500_000_000, 1_500_000_000 + count),
    })


def test_load_ratings_preserves_zero_and_large_imdb_id(tmp_path: Path):
    path = tmp_path / "ratings.dat"
    path.write_text("1::0000008::0::1500000000\n2::12345678::10::1500000001\n")
    frame = load_ratings(path)
    assert frame["Rating"].tolist() == [0, 10]
    assert frame["MovieID"].tolist() == [8, 12345678]
    assert frame["RawMovieID"].tolist() == ["0000008", "12345678"]


@pytest.mark.parametrize("line", [
    "1::2::11::1500000000", "1::2::8.5::1500000000",
    "1::2::8", "1::2::8::1500000000::extra", "1::2::::1500000000",
])
def test_load_rejects_bad_ratings(tmp_path: Path, line):
    path = tmp_path / "ratings.dat"
    path.write_text(line + "\n")
    with pytest.raises(ValueError):
        load_ratings(path)


def test_movies_accept_missing_genres_and_title_separator(tmp_path: Path):
    path = tmp_path / "movies.dat"
    path.write_text("1::A::B (2000)::Drama\n2::Title (2001)::\n")
    frame = load_movies(path)
    assert frame["Title"].tolist() == ["A::B (2000)", "Title (2001)"]
    assert frame["Genres"].tolist() == ["Drama", ""]


def test_movie_aliases_are_preserved_and_reported(tmp_path: Path):
    path = tmp_path / "movies.dat"
    path.write_text("001::Title (2001)::\n1::Title (2001)::Drama\n")
    movies = load_movies(path)
    assert movies["RawMovieID"].tolist() == ["001", "1"]
    summary, _ = audit(ratings(1), movies)
    assert summary["catalog_metadata_rows"] == 2
    assert summary["catalog_unique_normalized_movies"] == 1
    assert summary["catalog_ids_with_multiple_metadata_rows"] == 1


def test_existing_nine_positive_event_minimum():
    result = split_audit(ratings(), 8, 9)
    assert result["eligible_users"] == 1
    assert [result[k] for k in ("train_events", "validation_events", "test_events")] == [5, 2, 2]
    assert result["users_tied_at_train_validation_boundary"] == 0
    assert result["test_events_item_absent_from_positive_pretest"] == 2
    assert split_audit(ratings(8), 8, 9)["eligible_users"] == 0
    assert split_audit(ratings(), 9, 9)["eligible_users"] == 0


def test_audit_reports_tied_boundaries_without_changing_rows():
    frame = ratings()
    frame["Timestamp"] = 1_500_000_000
    original = frame.copy(deep=True)
    movies = pd.DataFrame({"MovieID": range(1, 10), "Title": ["A"] * 9, "Genres": [""] * 9})
    summary, thresholds = audit(frame, movies)
    assert summary["events_in_same_user_timestamp_ties"] == 9
    primary = thresholds.loc[thresholds["threshold"].eq(8)].iloc[0]
    assert primary["users_tied_at_train_validation_boundary"] == 1
    assert primary["users_tied_at_validation_test_boundary"] == 1
    pd.testing.assert_frame_equal(frame, original)


def test_repeated_user_movie_requires_explicit_policy():
    frame = ratings()
    frame.loc[1, "MovieID"] = 1
    movies = pd.DataFrame({"MovieID": range(1, 10)})
    with pytest.raises(ValueError, match="repeated user-movie"):
        audit(frame, movies)
