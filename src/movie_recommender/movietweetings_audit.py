"""Offline dataset audit only: no training or changes to the MovieLens pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig,
    revised_chronological_split,
)


COMMIT = "44c525d0c766944910686c60697203cda39305d6"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "movietweetings" / COMMIT
OUTPUT_DIR = ROOT / "outputs" / "movietweetings_audit" / COMMIT
BASE_URL = f"https://raw.githubusercontent.com/sidooms/MovieTweetings/{COMMIT}"


def load_ratings(path: Path) -> pd.DataFrame:
    """Validate all raw fields without removing or recoding any records."""
    frame = pd.read_csv(path, sep="::", engine="python", header=None, dtype=str)
    if frame.shape[1] != 4 or frame.empty or frame.isna().any().any():
        raise ValueError("Expected nonempty ratings with exactly four nonmissing fields")
    frame.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    raw_movie_ids = frame["MovieID"].copy()
    for column in frame:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain finite integers")
        frame[column] = values.astype("int64")
    if (frame[["UserID", "MovieID", "Timestamp"]] <= 0).any().any():
        raise ValueError("IDs and timestamps must be positive")
    if not frame["Rating"].between(0, 10).all():
        raise ValueError("MovieTweetings ratings must be between 0 and 10")
    pd.to_datetime(frame["Timestamp"], unit="s", utc=True, errors="raise")
    frame["RawMovieID"] = raw_movie_ids
    return frame


def load_movies(path: Path) -> pd.DataFrame:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            movie_id, rest = line.split("::", 1)
            title, genres = rest.rsplit("::", 1)
            rows.append((int(movie_id), title, genres, movie_id))
        except ValueError as error:
            raise ValueError(f"Invalid movie metadata at line {line_number}") from error
    frame = pd.DataFrame(rows, columns=["MovieID", "Title", "Genres", "RawMovieID"])
    if frame.empty:
        raise ValueError("Movie metadata must be nonempty")
    if (frame["MovieID"] <= 0).any() or frame["Title"].eq("").any():
        raise ValueError("Movie IDs and titles must be valid")
    return frame


def split_audit(ratings: pd.DataFrame, threshold: int, catalog_size: int) -> dict:
    """Run the existing split, so the audit cannot drift from experiment rules."""
    config = RevisedConfig(relevance_threshold=float(threshold))
    split = revised_chronological_split(ratings, config)
    train, validation, test = (
        split.train_positive, split.validation_positive, split.test_positive
    )
    pretest = pd.concat([train, validation], ignore_index=True)
    pretest_counts = pretest.groupby("UserID").size()
    result = {
        "threshold": threshold,
        "positive_events_all_users": int(ratings["Rating"].ge(threshold).sum()),
        "eligible_users": len(split.eligible_users),
        "excluded_users": len(split.excluded_users),
        "excluded_user_percent": 100 * len(split.excluded_users) / ratings["UserID"].nunique(),
        "train_events": len(train),
        "validation_events": len(validation),
        "test_events": len(test),
        "eligible_positive_items": int(pd.concat([pretest, test])["MovieID"].nunique()),
        "median_pretest_length": float(pretest_counts.median()) if len(pretest) else None,
        "users_pretest_longer_than_64": int(pretest_counts.gt(64).sum()),
        "test_events_item_absent_from_positive_pretest": int(
            (~test["MovieID"].isin(pretest["MovieID"])).sum()
        ),
        "one_float32_user_catalog_score_matrix_gib": (
            len(split.eligible_users) * catalog_size * 4 / 1024**3
        ),
    }
    for name, left, right in (
        ("train_validation", train, validation),
        ("validation_test", validation, test),
    ):
        boundaries = pd.concat(
            [left.groupby("UserID")["Timestamp"].max().rename("left"),
             right.groupby("UserID")["Timestamp"].min().rename("right")], axis=1
        ).dropna()
        if (boundaries["left"] > boundaries["right"]).any():
            raise AssertionError("Chronological split reversed a timestamp boundary")
        result[f"users_tied_at_{name}_boundary"] = int(
            boundaries["left"].eq(boundaries["right"]).sum()
        )
    return result


def audit(ratings: pd.DataFrame, movies: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    timestamps = pd.to_datetime(ratings["Timestamp"], unit="s", utc=True)
    ties = ratings.duplicated(["UserID", "Timestamp"], keep=False)
    histories = ratings.groupby("UserID").size()
    missing = ~ratings["MovieID"].isin(movies["MovieID"])
    duplicate_pairs = int(ratings.duplicated(["UserID", "MovieID"]).sum())
    if missing.any():
        raise ValueError(f"{int(missing.sum())} ratings have no movie metadata")
    # Repeated items can leak between set-based targets and history; do not silently deduplicate.
    if duplicate_pairs:
        raise ValueError(f"{duplicate_pairs} repeated user-movie rows need a policy before splitting")
    summary = {
        "ratings": len(ratings),
        "users": int(ratings["UserID"].nunique()),
        "rated_movies": int(ratings["MovieID"].nunique()),
        "catalog_metadata_rows": len(movies),
        "catalog_unique_normalized_movies": int(movies["MovieID"].nunique()),
        "catalog_duplicate_id_extra_rows": int(movies["MovieID"].duplicated().sum()),
        "catalog_ids_with_multiple_metadata_rows": int(
            movies.loc[movies["MovieID"].duplicated(keep=False), "MovieID"].nunique()
        ),
        "missing_metadata_rows": int(missing.sum()),
        "exact_duplicate_rows": int(ratings.duplicated(
            ["UserID", "MovieID", "Rating", "Timestamp"]
        ).sum()),
        "repeated_user_movie_rows": duplicate_pairs,
        "zero_ratings": int(ratings["Rating"].eq(0).sum()),
        "earliest_utc": timestamps.min().isoformat(),
        "latest_utc": timestamps.max().isoformat(),
        "future_timestamp_rows": int((timestamps > pd.Timestamp.now(tz="UTC")).sum()),
        "events_in_same_user_timestamp_ties": int(ties.sum()),
        "users_with_timestamp_ties": int(ratings.loc[ties, "UserID"].nunique()),
        "events_not_at_utc_midnight": int(ratings["Timestamp"].mod(86400).ne(0).sum()),
        "raw_history_length_quantiles": {
            str(q): float(histories.quantile(q)) for q in (0, .25, .5, .75, .9, .95, 1)
        },
        "movies_without_genres": int(movies["Genres"].eq("").sum()),
        "catalog_type_verified": False,
        "note": "No authoritative title-type field; do not claim every item is a feature film.",
    }
    thresholds = pd.DataFrame([
        split_audit(ratings, t, int(movies["MovieID"].nunique())) for t in (7, 8, 9)
    ])
    return summary, thresholds


def file_provenance(data_dir: Path) -> dict:
    files = {}
    for name in ("ratings.dat", "movies.dat", "README.md", "LICENSE"):
        data = (data_dir / name).read_bytes()
        remote = f"latest/{name}" if name.endswith(".dat") else name
        files[name] = {
            "source_url": f"{BASE_URL}/{remote}",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": hashlib.sha1(
                f"blob {len(data)}\0".encode() + data
            ).hexdigest(),
        }
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    provenance = file_provenance(DATA_DIR)
    # This blob ID was supplied by the upstream commit API for the pinned ratings.
    if provenance["ratings.dat"]["git_blob_sha1"] != "d1e392168f487faca0fb781d647b97f57a778494":
        raise ValueError("Ratings do not match the pinned upstream Git blob")
    ratings = load_ratings(DATA_DIR / "ratings.dat")
    movies = load_movies(DATA_DIR / "movies.dat")
    summary, thresholds = audit(ratings, movies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "MovieTweetings",
        "commit": COMMIT,
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": provenance,
        "summary": summary,
        "threshold_sensitivity": thresholds.to_dict(orient="records"),
        "protocol": {
            "provisional_primary_threshold": 8,
            "thresholds_7_and_9": "audit-only sensitivity; not selected from model scores",
            "split": "existing revised_chronological_split; min 5 train + 2 validation + 2 test",
            "tie_break": "timestamp, movie ID, source row",
            "training_performed": False,
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    thresholds.to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)
    movies.loc[movies["MovieID"].duplicated(keep=False)].to_csv(
        args.output_dir / "movie_id_collisions.csv", index=False
    )
    ratings["Rating"].value_counts().sort_index().rename_axis("rating").to_csv(
        args.output_dir / "rating_distribution.csv", header=["count"]
    )
    selected = thresholds.loc[thresholds["threshold"].eq(8)].iloc[0]
    lines = [
        "# MovieTweetings feasibility audit", "",
        f"Pinned upstream commit: `{COMMIT}`. No models were trained.", "",
        "## Decision", "",
        "The eligible sample supports a pilot, not yet an unqualified formal replication.",
        "Confirm the provisional threshold and title scope; resolve metadata aliases and",
        "use memory-conscious ranking before training. Retain the original MovieLens results.", "",
        "## Raw data", "",
        *[f"- {key}: {value}" for key, value in summary.items()], "",
        "## Existing split at the provisional rating >= 8 threshold", "",
        *[f"- {key}: {value}" for key, value in selected.items()], "",
        "## Interpretation and decisions before training", "",
        "- Threshold 8 is provisional, not evidence that 8/10 and 4/5 mean the same thing.",
        "- Thresholds 7 and 9 are feasibility sensitivity checks only; see the CSV.",
        "- Timestamps are recorded rating events, not verified viewing times.",
        "- Second-valued timestamps do not prove accurate real-world event order.",
        "- Equal-time split boundaries are deterministic, not strictly earlier/later events.",
        "- Raw IDs are preserved; numeric normalization collapses leading-zero aliases.",
        "- Duplicate metadata is listed in movie_id_collisions.csv, not silently merged.",
        "  The audit uses unique normalized IDs for catalog size; no genre/title winner is chosen.",
        "- The source is a movie-rating dataset but has no authoritative title-type field.",
        "  A strict feature-film-only scope needs separate metadata verification.",
        "- Excluding short histories changes the population; report the exclusion rate.",
        "- Items absent from positive pre-test training need a declared cold-item policy.",
        "- Memory estimate covers only ONE float32 score matrix, not total peak RAM.",
        "- Preserve the same users, targets, catalog and hyperparameters across order conditions.",
        "- Shuffling before truncation changes visible items as well as order for long histories.",
        "- Next: adapt a separate runner, smoke-test all conditions, then repeat model/shuffle seeds.",
        "- Do not interpret this audit as evidence for or against an order effect.", "",
        "## Reproduce", "",
        "```sh", "PYTHONPATH=src .venv/bin/python -m movie_recommender.movietweetings_audit",
        "```", "",
        "Source URLs, sizes and hashes are recorded in audit.json. Raw files stay Git-ignored.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(thresholds.to_string(index=False))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
