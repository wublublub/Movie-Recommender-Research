"""Temporal feasibility analysis for MovieLens 1M rating submissions."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Final

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "movie_recommender_matplotlib"),
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from movie_recommender.ml1m_data_loader import (
    DEFAULT_ML1M_DATA_DIR,
    load_ml1m_movies,
    load_ml1m_ratings,
    load_ml1m_users,
)


DEFAULT_OUTPUT_DIR: Final = (
    Path(__file__).resolve().parents[2] / "outputs" / "ml1m_temporal_feasibility"
)
SESSION_THRESHOLDS_MINUTES: Final = {"30m": 30.0, "2h": 120.0, "24h": 1440.0}
QUANTILE_LEVELS: Final = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
SELECTED_CORRELATION_FEATURES: Final = (
    "rating_count",
    "history_span_days",
    "median_gap_minutes",
    "largest_day_ratio",
    "largest_session_ratio_30m",
    "largest_session_ratio_2h",
    "largest_session_ratio_24h",
    "genre_js_divergence",
)


def _require_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def validate_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate source frames and convert timestamps in a working copy.

    Returns:
        A validation table and timestamp-enriched ratings frame.
    """
    _require_columns(ratings, {"UserID", "MovieID", "Rating", "Timestamp"}, "ratings")
    _require_columns(movies, {"MovieID", "Title", "Genres"}, "movies")
    _require_columns(
        users,
        {"UserID", "Gender", "Age", "Occupation", "ZipCode"},
        "users",
    )

    timed_ratings = ratings.copy()
    timed_ratings["RatingTime"] = pd.to_datetime(
        timed_ratings["Timestamp"], unit="s", utc=True, errors="coerce"
    )
    invalid_timestamps = int(timed_ratings["RatingTime"].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"ratings contains {invalid_timestamps} invalid timestamps")

    rating_counts = timed_ratings.groupby("UserID", sort=False).size()
    users_without_ratings = set(users["UserID"]).difference(rating_counts.index)
    if users_without_ratings:
        raise ValueError(
            f"users.dat contains {len(users_without_ratings)} users without ratings"
        )

    unknown_movies = set(timed_ratings["MovieID"]).difference(movies["MovieID"])
    if unknown_movies:
        raise ValueError(
            f"ratings.dat references {len(unknown_movies)} unknown movie IDs"
        )

    rows: list[dict[str, object]] = []
    for dataset_name, frame in (
        ("ratings", ratings),
        ("movies", movies),
        ("users", users),
    ):
        rows.extend(
            {
                "metric": f"{dataset_name}_missing_{column}",
                "value": int(count),
            }
            for column, count in frame.isna().sum().items()
        )

    rows.extend(
        [
            {"metric": "ratings_row_count", "value": len(ratings)},
            {"metric": "movies_row_count", "value": len(movies)},
            {"metric": "users_row_count", "value": len(users)},
            {"metric": "unique_rated_users", "value": ratings["UserID"].nunique()},
            {"metric": "unique_rated_movies", "value": ratings["MovieID"].nunique()},
            {"metric": "unique_catalog_movies", "value": movies["MovieID"].nunique()},
            {"metric": "duplicated_rating_rows", "value": int(ratings.duplicated().sum())},
            {"metric": "minimum_rating", "value": ratings["Rating"].min()},
            {"metric": "maximum_rating", "value": ratings["Rating"].max()},
            {"metric": "invalid_timestamps", "value": invalid_timestamps},
            {"metric": "minimum_timestamp_utc", "value": timed_ratings["RatingTime"].min().isoformat()},
            {"metric": "maximum_timestamp_utc", "value": timed_ratings["RatingTime"].max().isoformat()},
            {"metric": "users_with_fewer_than_20_ratings", "value": int(rating_counts.lt(20).sum())},
        ]
    )
    return pd.DataFrame(rows), timed_ratings


def build_user_temporal_features(timed_ratings: pd.DataFrame) -> pd.DataFrame:
    """Create user-level timing and session features.

    Equal timestamps retain their original file order. A new session starts only
    when the gap is strictly greater than the selected inactivity threshold.
    """
    work = timed_ratings.reset_index(names="_source_order").sort_values(
        ["UserID", "Timestamp", "_source_order"], kind="mergesort"
    )
    grouped = work.groupby("UserID", sort=True)
    work["gap_minutes"] = grouped["RatingTime"].diff().dt.total_seconds().div(60)
    work["active_day"] = work["RatingTime"].dt.floor("D")
    work["hour_bucket"] = work["RatingTime"].dt.floor("h")

    base = grouped.agg(
        rating_count=("MovieID", "size"),
        first_rating_time=("RatingTime", "min"),
        last_rating_time=("RatingTime", "max"),
    )
    base["history_span_hours"] = (
        base["last_rating_time"] - base["first_rating_time"]
    ).dt.total_seconds().div(3600)
    base["history_span_days"] = base["history_span_hours"].div(24)

    gap_features = work.groupby("UserID")["gap_minutes"].agg(
        mean_gap_minutes="mean",
        median_gap_minutes="median",
        gap_std_minutes="std",
        minimum_gap_minutes="min",
        maximum_gap_minutes="max",
    )
    gap_features["maximum_gap_days"] = gap_features.pop("maximum_gap_minutes").div(1440)

    active_day_counts = work.groupby("UserID")["active_day"].nunique()
    max_day_counts = work.groupby(["UserID", "active_day"]).size().groupby("UserID").max()
    max_hour_counts = work.groupby(["UserID", "hour_bucket"]).size().groupby("UserID").max()

    features = base.join(gap_features)
    features["unique_active_days"] = active_day_counts
    features["ratings_per_active_day"] = features["rating_count"].div(active_day_counts)
    features["maximum_ratings_in_one_hour"] = max_hour_counts
    features["maximum_ratings_in_one_day"] = max_day_counts
    features["largest_hour_ratio"] = max_hour_counts.div(features["rating_count"])
    features["largest_day_ratio"] = max_day_counts.div(features["rating_count"])

    for label, threshold_minutes in SESSION_THRESHOLDS_MINUTES.items():
        new_session = work["gap_minutes"].isna() | work["gap_minutes"].gt(
            threshold_minutes
        )
        session_number = new_session.groupby(work["UserID"]).cumsum()
        session_sizes = work.groupby(["UserID", session_number]).size()
        session_stats = session_sizes.groupby(level=0).agg(
            session_count="size",
            mean_session_size="mean",
            median_session_size="median",
            maximum_session_size="max",
        )
        session_stats["largest_session_ratio"] = session_stats[
            "maximum_session_size"
        ].div(features["rating_count"])
        session_stats = session_stats.add_suffix(f"_{label}")
        features = features.join(session_stats)

    features.index.name = "user_id"
    features = features.reset_index()
    expected_users = set(timed_ratings["UserID"])
    if set(features["user_id"]) != expected_users:
        raise RuntimeError("Temporal feature construction did not preserve every user")
    return features


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    """Calculate row-wise Shannon entropy in bits."""
    safe = np.where(probabilities > 0, probabilities, 1.0)
    return -(np.where(probabilities > 0, probabilities * np.log2(safe), 0.0)).sum(axis=1)


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Calculate numerically safe row-wise Jensen-Shannon divergence in bits."""
    midpoint = 0.5 * (first + second)
    first_ratio = np.ones_like(first)
    second_ratio = np.ones_like(second)
    np.divide(first, midpoint, out=first_ratio, where=first > 0)
    np.divide(second, midpoint, out=second_ratio, where=second > 0)
    first_log_ratio = np.zeros_like(first)
    second_log_ratio = np.zeros_like(second)
    np.log2(first_ratio, out=first_log_ratio, where=first > 0)
    np.log2(second_ratio, out=second_log_ratio, where=second > 0)
    first_terms = first * first_log_ratio
    second_terms = second * second_log_ratio
    return 0.5 * (first_terms.sum(axis=1) + second_terms.sum(axis=1))


def build_genre_evolution_features(
    timed_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Compare early and late rated-movie genre composition for every user.

    Histories are stably ordered by timestamp and source-file order. For an odd
    history length, the early half receives the middle interaction. Each movie
    contributes total weight one, divided equally among its listed genres.
    """
    ordered = timed_ratings.reset_index(names="_source_order").sort_values(
        ["UserID", "Timestamp", "_source_order"], kind="mergesort"
    )
    ordered["position"] = ordered.groupby("UserID").cumcount()
    ordered["history_length"] = ordered.groupby("UserID")["MovieID"].transform("size")
    ordered["early_size"] = (ordered["history_length"] + 1) // 2
    ordered["history_half"] = np.where(
        ordered["position"] < ordered["early_size"], "early", "late"
    )

    movie_genres = movies[["MovieID", "Genres"]].copy()
    movie_genres["genre"] = movie_genres["Genres"].str.split("|")
    movie_genres["genre_count"] = movie_genres["genre"].str.len()
    movie_genres = movie_genres.explode("genre")
    movie_genres["genre_weight"] = 1.0 / movie_genres["genre_count"]

    weighted = ordered[["UserID", "MovieID", "history_half"]].merge(
        movie_genres[["MovieID", "genre", "genre_weight"]],
        on="MovieID",
        how="left",
        validate="many_to_many",
    )
    if weighted["genre"].isna().any():
        raise ValueError("At least one rated movie has no genre metadata")

    half_weights = weighted.groupby(
        ["UserID", "history_half", "genre"], observed=True
    )["genre_weight"].sum()
    half_distributions = half_weights.div(
        half_weights.groupby(level=[0, 1]).transform("sum")
    ).unstack(["history_half", "genre"], fill_value=0.0)

    genres = sorted(movie_genres["genre"].unique())
    users = sorted(timed_ratings["UserID"].unique())
    expected_columns = pd.MultiIndex.from_product(
        [["early", "late"], genres], names=["history_half", "genre"]
    )
    half_distributions = half_distributions.reindex(
        index=users, columns=expected_columns, fill_value=0.0
    )
    early = half_distributions["early"].to_numpy(dtype=float)
    late = half_distributions["late"].to_numpy(dtype=float)

    overall_weights = weighted.groupby(["UserID", "genre"], observed=True)[
        "genre_weight"
    ].sum()
    overall_distributions = overall_weights.div(
        overall_weights.groupby(level=0).transform("sum")
    ).unstack("genre", fill_value=0.0).reindex(index=users, columns=genres, fill_value=0.0)
    overall = overall_distributions.to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "user_id": users,
            "genre_js_divergence": _jensen_shannon(early, late),
            "early_genre_entropy": _entropy(early),
            "late_genre_entropy": _entropy(late),
            "overall_genre_entropy": _entropy(overall),
        }
    )


def build_temporal_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize user counts and percentages for requested conditions."""
    conditions: list[tuple[str, str, pd.Series]] = [
        ("history_duration", "all_ratings_within_1_hour", features["history_span_hours"].le(1)),
        ("history_duration", "all_ratings_within_1_day", features["history_span_days"].le(1)),
        ("history_duration", "history_longer_than_7_days", features["history_span_days"].gt(7)),
        ("history_duration", "history_longer_than_30_days", features["history_span_days"].gt(30)),
        ("history_duration", "history_longer_than_90_days", features["history_span_days"].gt(90)),
        ("history_duration", "history_longer_than_365_days", features["history_span_days"].gt(365)),
    ]
    for threshold in (0.5, 0.8, 0.9):
        conditions.append(
            (
                "daily_concentration",
                f"largest_day_ratio_gte_{threshold:.2f}",
                features["largest_day_ratio"].ge(threshold),
            )
        )
    for label in SESSION_THRESHOLDS_MINUTES:
        for threshold in (0.5, 0.8, 0.9):
            conditions.append(
                (
                    f"session_concentration_{label}",
                    f"largest_session_ratio_{label}_gte_{threshold:.2f}",
                    features[f"largest_session_ratio_{label}"].ge(threshold),
                )
            )

    return pd.DataFrame(
        [
            {
                "category": category,
                "condition": condition,
                "user_count": int(mask.sum()),
                "user_percentage": float(mask.mean() * 100),
            }
            for category, condition, mask in conditions
        ]
    )


def build_quantiles(features: pd.DataFrame) -> pd.DataFrame:
    """Create long-form quantiles for requested temporal features."""
    quantile_features = [
        "rating_count",
        "history_span_days",
        "median_gap_minutes",
        "largest_day_ratio",
    ]
    for label in SESSION_THRESHOLDS_MINUTES:
        quantile_features.extend(
            [f"largest_session_ratio_{label}", f"session_count_{label}"]
        )
    quantiles = features[quantile_features].quantile(QUANTILE_LEVELS)
    quantiles.index.name = "quantile"
    return quantiles.reset_index().melt(
        id_vars="quantile", var_name="feature", value_name="value"
    )[["feature", "quantile", "value"]]


def build_spearman_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman correlations for selected user-level features."""
    return features[list(SELECTED_CORRELATION_FEATURES)].corr(method="spearman")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a Matplotlib figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Create all requested temporal feasibility visualizations."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(features["rating_count"], bins=np.logspace(np.log10(20), np.log10(features["rating_count"].max()), 40))
    ax.set_xscale("log")
    ax.set_title("Distribution of User Rating Counts")
    ax.set_xlabel("Ratings per user (log scale)")
    ax.set_ylabel("Number of users")
    _save_figure(fig, plots_dir / "01_user_rating_counts.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(features["history_span_days"]), bins=40)
    ax.set_title("Distribution of Rating-History Span")
    ax.set_xlabel("log1p(history span in days)")
    ax.set_ylabel("Number of users")
    _save_figure(fig, plots_dir / "02_history_span_days.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(features["largest_day_ratio"], bins=40, range=(0, 1))
    ax.set_title("Distribution of Largest-Day Ratio")
    ax.set_xlabel("Largest-day ratio")
    ax.set_ylabel("Number of users")
    _save_figure(fig, plots_dir / "03_largest_day_ratio.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in SESSION_THRESHOLDS_MINUTES:
        ax.hist(
            features[f"largest_session_ratio_{label}"],
            bins=35,
            range=(0, 1),
            alpha=0.45,
            label=label,
        )
    ax.set_title("Largest-Session Ratio by Inactivity Threshold")
    ax.set_xlabel("Largest-session ratio")
    ax.set_ylabel("Number of users")
    ax.legend(title="Threshold")
    _save_figure(fig, plots_dir / "04_largest_session_ratios.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in SESSION_THRESHOLDS_MINUTES:
        ax.hist(
            np.log1p(features[f"session_count_{label}"]),
            bins=35,
            alpha=0.45,
            label=label,
        )
    ax.set_title("Session Counts by Inactivity Threshold")
    ax.set_xlabel("log1p(session count)")
    ax.set_ylabel("Number of users")
    ax.legend(title="Threshold")
    _save_figure(fig, plots_dir / "05_session_counts.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(features["genre_js_divergence"], bins=40, range=(0, 1))
    ax.set_title("Distribution of Early–Late Genre Divergence")
    ax.set_xlabel("Jensen–Shannon divergence (bits)")
    ax.set_ylabel("Number of users")
    _save_figure(fig, plots_dir / "06_genre_js_divergence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        np.log1p(features["rating_count"]),
        np.log1p(features["history_span_days"]),
        alpha=0.25,
        s=12,
    )
    ax.set_title("Rating Count vs Rating-History Span")
    ax.set_xlabel("log1p(rating count)")
    ax.set_ylabel("log1p(history span in days)")
    _save_figure(fig, plots_dir / "07_rating_count_vs_history_span.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        features["largest_day_ratio"],
        features["genre_js_divergence"],
        alpha=0.25,
        s=12,
    )
    ax.set_title("Daily Concentration vs Genre Divergence")
    ax.set_xlabel("Largest-day ratio")
    ax.set_ylabel("Jensen–Shannon divergence (bits)")
    _save_figure(fig, plots_dir / "08_day_ratio_vs_genre_divergence.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlations.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Spearman Correlations of User-Level Features")
    ax.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlations.index)), correlations.index)
    for row in range(len(correlations.index)):
        for column in range(len(correlations.columns)):
            ax.text(column, row, f"{correlations.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    _save_figure(fig, plots_dir / "09_spearman_correlation_heatmap.png")


def _summary_value(summary: pd.DataFrame, condition: str) -> tuple[int, float]:
    row = summary.loc[summary["condition"].eq(condition)].iloc[0]
    return int(row["user_count"]), float(row["user_percentage"])


def _quantile_value(quantiles: pd.DataFrame, feature: str, quantile: float) -> float:
    row = quantiles.loc[
        quantiles["feature"].eq(feature) & quantiles["quantile"].eq(quantile),
        "value",
    ]
    return float(row.iloc[0])


def create_feasibility_report(
    features: pd.DataFrame,
    summary: pd.DataFrame,
    quantiles: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:
    """Create a concise evidence-based feasibility report."""
    within_day_count, within_day_pct = _summary_value(
        summary, "all_ratings_within_1_day"
    )
    over_year_count, over_year_pct = _summary_value(
        summary, "history_longer_than_365_days"
    )
    day_80_count, day_80_pct = _summary_value(
        summary, "largest_day_ratio_gte_0.80"
    )
    span_q10 = _quantile_value(quantiles, "history_span_days", 0.1)
    span_median = _quantile_value(quantiles, "history_span_days", 0.5)
    span_q90 = _quantile_value(quantiles, "history_span_days", 0.9)
    js_q10 = float(features["genre_js_divergence"].quantile(0.1))
    js_median = float(features["genre_js_divergence"].median())
    js_q90 = float(features["genre_js_divergence"].quantile(0.9))

    session_lines = []
    for label in SESSION_THRESHOLDS_MINUTES:
        concentrated_count, concentrated_percentage = _summary_value(
            summary, f"largest_session_ratio_{label}_gte_0.80"
        )
        session_count_q90 = _quantile_value(
            quantiles, f"session_count_{label}", 0.9
        )
        session_lines.append(
            f"- **{label}:** median session count "
            f"{features[f'session_count_{label}'].median():.1f}, 90th-percentile "
            f"session count {session_count_q90:.1f}, and {concentrated_count:,} users "
            f"({concentrated_percentage:.2f}%) with at least 80% of ratings in one session."
        )

    upper_triangle = correlations.where(
        np.triu(np.ones(correlations.shape), k=1).astype(bool)
    ).stack()
    strongest_pair = upper_triangle.abs().idxmax()
    strongest_value = float(correlations.loc[strongest_pair[0], strongest_pair[1]])
    near_redundant = upper_triangle[upper_triangle.abs().ge(0.9)]
    redundancy_text = (
        f"{len(near_redundant)} feature pair(s) have |rho| >= 0.90"
        if len(near_redundant)
        else "no feature pair has |rho| >= 0.90"
    )

    return f"""# MovieLens 1M Temporal Feasibility Report

## Scope and interpretation

This analysis concerns **rating-submission sequences**. A timestamp records when a rating was submitted, not when a movie was watched. Temporal concentration is therefore a proxy for whether submission order may carry useful sequence information, not a ground-truth reliability label.

## A. Variation in rating-history duration

MovieLens 1M contains substantial duration variation. The 10th, median, and 90th percentiles of history span are {span_q10:.2f}, {span_median:.2f}, and {span_q90:.2f} days. {within_day_count:,} users ({within_day_pct:.2f}%) submitted all ratings within one day, while {over_year_count:,} ({over_year_pct:.2f}%) have histories longer than one year.

## B. Concentrated and distributed comparison groups

The data supports both concentrated and temporally distributed groups. {day_80_count:,} users ({day_80_pct:.2f}%) submitted at least 80% of ratings on a single calendar day, alongside {over_year_count:,} users with histories longer than one year. Group definitions should be preregistered and sensitivity-tested rather than treated as labels of true sequence reliability.

## C. Session-threshold sensitivity

Session structure changes with the inactivity threshold:

{chr(10).join(session_lines)}

The threshold is consequential, so results should be compared across all three definitions rather than relying on one arbitrary cutoff.

## D. Observed genre-composition change

Early-versus-late genre Jensen–Shannon divergence varies across users: the 10th, median, and 90th percentiles are {js_q10:.3f}, {js_median:.3f}, and {js_q90:.3f} bits. This indicates heterogeneous change in the genres of rated movies, but it is only a proxy for preference evolution and cannot confirm real-world interest drift.

## E. Feature distinctness

The strongest absolute selected-feature Spearman correlation is between `{strongest_pair[0]}` and `{strongest_pair[1]}` (rho = {strongest_value:.3f}); {redundancy_text}. Threshold-specific session measures may be partly redundant, while duration, concentration, activity, and genre-change measures retain different interpretations. No causal claims are implied.

## F. Course-project suitability

**Conclusion: Conditionally suitable.** MovieLens 1M has enough users and measurable variation in history duration, rating concentration, session structure, and observed genre composition to support a course-project comparison of sequential and non-sequential recommendation strategies. Suitability is conditional because timestamps are rating-submission times, actual watching times are unavailable, concentrated ratings may represent profile-building or historical backfilling, genre change is only a proxy for preference evolution, and no ground-truth timestamp reliability label exists. Any algorithm-selection experiment must frame these features as observable proxies and use robust sensitivity analyses.
"""


def run_analysis(data_dir: Path, output_dir: Path) -> None:
    """Run the complete MovieLens 1M temporal feasibility analysis."""
    print("[1/7] Loading MovieLens 1M files")
    ratings = load_ml1m_ratings(data_dir)
    movies = load_ml1m_movies(data_dir)
    users = load_ml1m_users(data_dir)

    print("[2/7] Validating source data and timestamps")
    validation, timed_ratings = validate_dataset(ratings, movies, users)

    print("[3/7] Building temporal and session features")
    temporal_features = build_user_temporal_features(timed_ratings)

    print("[4/7] Building observed genre-evolution features")
    genre_features = build_genre_evolution_features(timed_ratings, movies)
    features = temporal_features.merge(
        genre_features, on="user_id", how="left", validate="one_to_one"
    )
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise RuntimeError(f"Generated user features contain missing values: {missing_columns}")

    print("[5/7] Calculating summaries and correlations")
    summary = build_temporal_summary(features)
    quantiles = build_quantiles(features)
    correlations = build_spearman_correlations(features)

    print("[6/7] Writing tables, figures, and feasibility report")
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "user_temporal_features.csv", index=False)
    summary.to_csv(output_dir / "temporal_summary.csv", index=False)
    quantiles.to_csv(output_dir / "temporal_quantiles.csv", index=False)
    correlations.to_csv(output_dir / "spearman_correlations.csv", index=True)
    validation.to_csv(output_dir / "dataset_validation.csv", index=False)
    create_plots(features, correlations, output_dir / "plots")
    report = create_feasibility_report(features, summary, quantiles, correlations)
    (output_dir / "feasibility_report.md").write_text(report, encoding="utf-8")

    print("[7/7] Verifying user preservation")
    if len(features) != ratings["UserID"].nunique():
        raise RuntimeError("Output feature table does not contain exactly one row per user")
    print(f"Complete: {len(features):,} users written to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ML1M_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    arguments = parse_args()
    run_analysis(arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
