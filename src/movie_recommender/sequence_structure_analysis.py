"""Explain user-level order sensitivity using pre-test sequence structure.

This module consumes frozen experiment artifacts. It never trains a recommender or
changes an experiment output. Sequence features are computed from original-order
training plus validation events, matching the history available before test ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ("fpmc", "lstm", "sasrec")
FEATURE_COLUMNS = (
    "sequence_length",
    "unique_item_count",
    "unique_item_ratio",
    "repeat_event_rate",
    "immediate_repeat_rate",
    "visits_per_unique_item",
    "maximum_item_frequency_share",
    "normalized_item_entropy",
    "repeated_bigram_rate",
    "local_transition_concentration",
    "first_order_conditional_entropy",
    "local_information_gain",
    "first_order_supported_transition_rate",
    "repeated_trigram_rate",
    "second_order_conditional_entropy",
    "long_range_gain",
    "normalized_long_range_gain",
    "second_order_supported_transition_rate",
    "median_gap_seconds",
    "mean_gap_seconds",
    "gap_standard_deviation_seconds",
    "gap_coefficient_of_variation",
    "burstiness",
    "timestamp_tie_rate",
    "short_gap_rate_1_minute",
    "short_gap_rate_1_hour",
    "short_gap_rate_1_day",
)
EXPLORATORY_FEATURES = {
    "first_order_conditional_entropy",
    "local_information_gain",
    "repeated_trigram_rate",
    "second_order_conditional_entropy",
    "long_range_gain",
    "normalized_long_range_gain",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the frozen-artifact explanatory analysis."""

    output_dir: str = "outputs/sequence_structure_analysis_20260929"
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 50760
    minimum_entropy_transitions: int = 20
    minimum_supported_transition_rate: float = 0.25


def _entropy(values: Sequence[int]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    if counts.size <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def _conditional_entropy(contexts: Iterable[object], targets: Iterable[int]) -> float:
    counts: dict[object, Counter[int]] = defaultdict(Counter)
    total = 0
    for context, target in zip(contexts, targets):
        counts[context][int(target)] += 1
        total += 1
    if total == 0:
        return math.nan
    result = 0.0
    for target_counts in counts.values():
        context_total = sum(target_counts.values())
        probabilities = np.asarray(list(target_counts.values()), dtype=float) / context_total
        result += (context_total / total) * float(
            -(probabilities * np.log(probabilities)).sum()
        )
    return result


def _repeated_ngram_rate(items: Sequence[int], width: int) -> float:
    total = len(items) - width + 1
    if total <= 0:
        return math.nan
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for index in range(total):
        gram = tuple(int(value) for value in items[index : index + width])
        if gram in seen:
            repeated += 1
        seen.add(gram)
    return repeated / total


def extract_user_sequence_features(
    items: Sequence[int], timestamps: Sequence[float]
) -> dict[str, float | int | bool]:
    """Compute interpretable features for one original-order pre-test history.

    Args:
        items: Ordered item identifiers.
        timestamps: Ordered timestamps in seconds, aligned with ``items``.

    Returns:
        A dictionary of scale, recurrence, entropy, and temporal features.

    Raises:
        ValueError: If inputs differ in length or contain fewer than two events.
    """
    if len(items) != len(timestamps):
        raise ValueError("items and timestamps must have identical lengths")
    if len(items) < 2:
        raise ValueError("at least two pre-test events are required")

    item_array = np.asarray(items, dtype=np.int64)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    sequence_length = len(item_array)
    item_counts = Counter(int(item) for item in item_array)
    unique_count = len(item_counts)

    seen: set[int] = set()
    repeat_events = 0
    for item in item_array:
        item_value = int(item)
        if item_value in seen:
            repeat_events += 1
        seen.add(item_value)

    previous_items = item_array[:-1]
    next_items = item_array[1:]
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for previous, following in zip(previous_items, next_items):
        transition_counts[int(previous)][int(following)] += 1
    transition_total = len(next_items)
    local_concentration = sum(
        max(counts.values()) for counts in transition_counts.values()
    ) / transition_total
    context_totals = {
        context: sum(counts.values()) for context, counts in transition_counts.items()
    }
    first_support = sum(
        context_totals[int(previous)] >= 2 for previous in previous_items
    ) / transition_total

    next_entropy = _entropy(next_items.tolist())
    first_entropy = _conditional_entropy(previous_items.tolist(), next_items.tolist())
    local_gain = (
        1.0 - first_entropy / next_entropy if next_entropy > 0 else math.nan
    )

    trigram_total = sequence_length - 2
    if trigram_total > 0:
        second_contexts = [
            (int(item_array[index - 2]), int(item_array[index - 1]))
            for index in range(2, sequence_length)
        ]
        second_targets = item_array[2:].tolist()
        first_suffix_contexts = item_array[1:-1].tolist()
        first_suffix_entropy = _conditional_entropy(
            first_suffix_contexts, second_targets
        )
        second_entropy = _conditional_entropy(second_contexts, second_targets)
        second_counts = Counter(second_contexts)
        second_support = sum(
            second_counts[context] >= 2 for context in second_contexts
        ) / trigram_total
        long_range_gain = first_suffix_entropy - second_entropy
        suffix_entropy = _entropy(second_targets)
        normalized_long_range_gain = (
            long_range_gain / suffix_entropy if suffix_entropy > 0 else math.nan
        )
    else:
        second_entropy = math.nan
        second_support = math.nan
        long_range_gain = math.nan
        normalized_long_range_gain = math.nan

    gaps = np.diff(timestamp_array)
    if np.any(gaps < 0):
        raise ValueError("timestamps must be in nondecreasing original order")
    mean_gap = float(np.mean(gaps))
    gap_std = float(np.std(gaps, ddof=0))
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.size:
        positive_mean = float(np.mean(positive_gaps))
        positive_std = float(np.std(positive_gaps, ddof=0))
        burstiness = (
            (positive_std - positive_mean) / (positive_std + positive_mean)
            if positive_std + positive_mean > 0
            else math.nan
        )
    else:
        burstiness = math.nan

    normalized_entropy = (
        _entropy(item_array.tolist()) / math.log(unique_count)
        if unique_count > 1
        else 0.0
    )
    reliable_first = (
        transition_total >= 20 and first_support >= 0.25
    )
    reliable_second = (
        trigram_total >= 20 and second_support >= 0.25
    )
    return {
        "sequence_length": sequence_length,
        "unique_item_count": unique_count,
        "unique_item_ratio": unique_count / sequence_length,
        "repeat_event_rate": repeat_events / sequence_length,
        "immediate_repeat_rate": float(np.mean(next_items == previous_items)),
        "visits_per_unique_item": sequence_length / unique_count,
        "maximum_item_frequency_share": max(item_counts.values()) / sequence_length,
        "normalized_item_entropy": normalized_entropy,
        "repeated_bigram_rate": _repeated_ngram_rate(item_array.tolist(), 2),
        "local_transition_concentration": local_concentration,
        "first_order_conditional_entropy": first_entropy,
        "local_information_gain": local_gain,
        "first_order_supported_transition_rate": first_support,
        "first_order_entropy_reliable": reliable_first,
        "repeated_trigram_rate": _repeated_ngram_rate(item_array.tolist(), 3),
        "second_order_conditional_entropy": second_entropy,
        "long_range_gain": long_range_gain,
        "normalized_long_range_gain": normalized_long_range_gain,
        "second_order_supported_transition_rate": second_support,
        "second_order_entropy_reliable": reliable_second,
        "median_gap_seconds": float(np.median(gaps)),
        "mean_gap_seconds": mean_gap,
        "gap_standard_deviation_seconds": gap_std,
        "gap_coefficient_of_variation": gap_std / mean_gap if mean_gap > 0 else math.nan,
        "burstiness": burstiness,
        "timestamp_tie_rate": float(np.mean(gaps == 0)),
        "short_gap_rate_1_minute": float(np.mean(gaps <= 60)),
        "short_gap_rate_1_hour": float(np.mean(gaps <= 3600)),
        "short_gap_rate_1_day": float(np.mean(gaps <= 86400)),
    }


def extract_feature_table(split_table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Extract per-user features from original train and validation events."""
    required = {"UserID", "MovieID", "Timestamp", "split"}
    missing = required.difference(split_table.columns)
    if missing:
        raise ValueError(f"split table is missing columns: {sorted(missing)}")
    pretest = split_table.loc[split_table["split"].isin(["train", "validation"])].copy()
    order_columns = ["UserID"]
    if "_user_order_rank" in pretest.columns:
        order_columns.append("_user_order_rank")
    else:
        order_columns.extend(
            column
            for column in ("Timestamp", "MovieID", "_source_order")
            if column in pretest.columns
        )
    pretest = pretest.sort_values(order_columns, kind="stable")
    records: list[dict[str, object]] = []
    for user_id, user_frame in pretest.groupby("UserID", sort=True):
        record: dict[str, object] = {"dataset": dataset, "user_id": int(user_id)}
        record.update(
            extract_user_sequence_features(
                user_frame["MovieID"].astype(int).tolist(),
                user_frame["Timestamp"].astype(float).tolist(),
            )
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_user_order_effects(
    metrics: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair Original and shuffled NDCG within initialization seeds.

    Shuffled scores are first averaged over shuffle seeds. Paired effects are then
    averaged over initialization seeds. Unmatched initialization seeds raise an
    error rather than being compared implicitly.
    """
    required = {
        "user_id",
        "model_name",
        "ndcg_10",
        "order_condition",
        "model_initialization_seed",
        "sequence_shuffle_seed",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics are missing columns: {sorted(missing)}")
    selected = metrics.loc[metrics["model_name"].isin(MODELS)].copy()
    keys = ["user_id", "model_name", "model_initialization_seed"]
    original = selected.loc[selected["order_condition"].eq("original_order")]
    shuffled = selected.loc[
        selected["order_condition"].eq("fully_shuffled_pretest")
    ]
    if original.duplicated(keys).any():
        raise ValueError("Original metrics contain duplicate user/model/init rows")
    shuffled_means = (
        shuffled.groupby(keys, as_index=False)
        .agg(
            shuffled_ndcg_10=("ndcg_10", "mean"),
            shuffle_seed_count=("sequence_shuffle_seed", "nunique"),
        )
    )
    paired = original[keys + ["ndcg_10"]].merge(
        shuffled_means, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"Original/Shuffled initialization pairing failed: {counts}")
    paired = paired.drop(columns="_merge").rename(
        columns={"ndcg_10": "original_ndcg_10"}
    )
    paired["delta_ndcg_10"] = (
        paired["original_ndcg_10"] - paired["shuffled_ndcg_10"]
    )
    paired["relative_order_drop"] = np.where(
        paired["original_ndcg_10"].abs() > 1e-12,
        paired["delta_ndcg_10"] / paired["original_ndcg_10"],
        np.nan,
    )
    paired.insert(0, "dataset", dataset)
    aggregate = (
        paired.groupby(["dataset", "user_id", "model_name"], as_index=False)
        .agg(
            original_ndcg_10=("original_ndcg_10", "mean"),
            shuffled_ndcg_10=("shuffled_ndcg_10", "mean"),
            delta_ndcg_10=("delta_ndcg_10", "mean"),
            delta_standard_deviation_across_initializations=(
                "delta_ndcg_10",
                "std",
            ),
            initialization_seed_count=("model_initialization_seed", "nunique"),
            shuffle_seed_count=("shuffle_seed_count", "min"),
        )
    )
    aggregate["relative_order_drop"] = np.where(
        aggregate["original_ndcg_10"].abs() > 1e-12,
        aggregate["delta_ndcg_10"] / aggregate["original_ndcg_10"],
        np.nan,
    )
    return aggregate, paired


def assign_quantile_groups(values: pd.Series) -> pd.Series:
    """Assign Low/Medium/High groups while preserving genuine value ties."""
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    valid = values.dropna()
    if valid.nunique() < 3:
        return result
    percentile_ranks = valid.rank(method="average", pct=True)
    groups = pd.cut(
        percentile_ranks,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    result.loc[valid.index] = groups.astype("string")
    return result


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan
    result = spearmanr(x, y, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def _rank_residuals(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    ranked = rankdata(values)
    design = np.column_stack([np.ones(len(control)), rankdata(control)])
    coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - design @ coefficients


def partial_spearman(
    x: Sequence[float], y: Sequence[float], control: Sequence[float]
) -> float:
    """Compute partial Spearman correlation controlling one variable."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    control_array = np.asarray(control, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(control_array)
    if valid.sum() < 4:
        return math.nan
    x_residual = _rank_residuals(x_array[valid], control_array[valid])
    y_residual = _rank_residuals(y_array[valid], control_array[valid])
    if np.std(x_residual) == 0 or np.std(y_residual) == 0:
        return math.nan
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def bootstrap_correlation_interval(
    x: Sequence[float],
    y: Sequence[float],
    *,
    samples: int,
    seed: int,
    control: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Return a reproducible user-bootstrap percentile interval."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    arrays = [x_array, y_array]
    if control is not None:
        control_array = np.asarray(control, dtype=float)
        arrays.append(control_array)
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    x_array, y_array = x_array[valid], y_array[valid]
    control_array = control_array[valid] if control is not None else None
    if len(x_array) < 4 or np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(x_array), len(x_array))
        if control_array is None:
            estimate, _ = _rank_correlation(x_array[indices], y_array[indices])
        else:
            estimate = partial_spearman(
                x_array[indices], y_array[indices], control_array[indices]
            )
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return math.nan, math.nan
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def bootstrap_mean_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    """Return a reproducible percentile interval for a paired mean effect."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sample_indices = rng.integers(0, len(array), len(array))
        estimates[index] = array[sample_indices].mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Adjust finite p-values with the Benjamini-Hochberg procedure."""
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.dropna().sort_values()
    count = len(finite)
    if count == 0:
        return adjusted
    raw_adjusted = finite.to_numpy() * count / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted.loc[finite.index] = np.minimum(monotone, 1.0)
    return adjusted


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create the requested feature quality summary by dataset."""
    records: list[dict[str, object]] = []
    for dataset, frame in features.groupby("dataset", sort=True):
        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(frame[feature], errors="coerce")
            finite = values.dropna()
            correlations = frame[[feature, "sequence_length"]].dropna()
            length_rho = (
                _rank_correlation(
                    correlations[feature].to_numpy(),
                    correlations["sequence_length"].to_numpy(),
                )[0]
                if feature != "sequence_length"
                else 1.0
            )
            records.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "count": int(finite.size),
                    "missing_count": int(values.isna().sum()),
                    "unique_value_count": int(finite.nunique()),
                    "mean": finite.mean(),
                    "median": finite.median(),
                    "standard_deviation": finite.std(),
                    "p05": finite.quantile(0.05),
                    "p25": finite.quantile(0.25),
                    "p75": finite.quantile(0.75),
                    "p95": finite.quantile(0.95),
                    "minimum": finite.min(),
                    "maximum": finite.max(),
                    "spearman_with_sequence_length": length_rho,
                    "near_constant": finite.nunique() <= 2,
                    "exploratory_due_to_sparse_estimation": feature
                    in EXPLORATORY_FEATURES,
                }
            )
    return pd.DataFrame.from_records(records)


def feature_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """Return long-form within-dataset Spearman feature correlations."""
    records: list[dict[str, object]] = []
    for dataset, frame in features.groupby("dataset", sort=True):
        correlation = frame[list(FEATURE_COLUMNS)].corr(method="spearman")
        for left_index, left in enumerate(FEATURE_COLUMNS):
            for right in FEATURE_COLUMNS[left_index + 1 :]:
                records.append(
                    {
                        "dataset": dataset,
                        "feature_a": left,
                        "feature_b": right,
                        "spearman_rho": correlation.loc[left, right],
                    }
                )
    return pd.DataFrame.from_records(records)


def association_analysis(
    features: pd.DataFrame,
    effects: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Relate each feature to per-user absolute Delta NDCG@10."""
    merged = effects.merge(features, on=["dataset", "user_id"], validate="many_to_one")
    records: list[dict[str, object]] = []
    for (dataset, model), frame in merged.groupby(["dataset", "model_name"], sort=True):
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            clean = pd.DataFrame(
                {
                    "feature_value": frame[feature],
                    "delta_ndcg_10": frame["delta_ndcg_10"],
                    "sequence_length": frame["sequence_length"],
                }
            ).dropna()
            rho, p_value = _rank_correlation(
                clean["feature_value"].to_numpy(),
                clean["delta_ndcg_10"].to_numpy(),
            )
            seed = config.bootstrap_seed + feature_index + len(records) * 31
            lower, upper = bootstrap_correlation_interval(
                clean["feature_value"],
                clean["delta_ndcg_10"],
                samples=config.bootstrap_samples,
                seed=seed,
            )
            partial = partial_spearman(
                clean["feature_value"],
                clean["delta_ndcg_10"],
                clean["sequence_length"],
            )
            partial_lower, partial_upper = bootstrap_correlation_interval(
                clean["feature_value"],
                clean["delta_ndcg_10"],
                control=clean["sequence_length"],
                samples=config.bootstrap_samples,
                seed=seed + 1,
            )
            records.append(
                {
                    "dataset": dataset,
                    "model_name": model,
                    "feature": feature,
                    "sample_size": len(clean),
                    "spearman_rho": rho,
                    "bootstrap_ci_95_lower": lower,
                    "bootstrap_ci_95_upper": upper,
                    "raw_p_value": p_value,
                    "partial_spearman_controlling_sequence_length": partial,
                    "partial_bootstrap_ci_95_lower": partial_lower,
                    "partial_bootstrap_ci_95_upper": partial_upper,
                    "exploratory_due_to_sparse_estimation": feature
                    in EXPLORATORY_FEATURES,
                }
            )
    result = pd.DataFrame.from_records(records)
    result["bh_fdr_adjusted_p_value"] = result.groupby(
        ["dataset", "model_name"], group_keys=False
    )["raw_p_value"].transform(benjamini_hochberg)
    return result


def seed_robustness_analysis(
    features: pd.DataFrame, paired_effects: pd.DataFrame
) -> pd.DataFrame:
    """Compute feature associations separately for every initialization seed."""
    merged = paired_effects.merge(
        features, on=["dataset", "user_id"], validate="many_to_one"
    )
    records: list[dict[str, object]] = []
    group_columns = ["dataset", "model_name", "model_initialization_seed"]
    for keys, frame in merged.groupby(group_columns, sort=True):
        dataset, model, initialization_seed = keys
        for feature in FEATURE_COLUMNS:
            clean = pd.DataFrame(
                {
                    "feature_value": frame[feature],
                    "delta_ndcg_10": frame["delta_ndcg_10"],
                    "sequence_length": frame["sequence_length"],
                }
            ).dropna()
            rho, _ = _rank_correlation(
                clean["feature_value"], clean["delta_ndcg_10"]
            )
            records.append(
                {
                    "dataset": dataset,
                    "model_name": model,
                    "model_initialization_seed": initialization_seed,
                    "feature": feature,
                    "sample_size": len(clean),
                    "spearman_rho": rho,
                    "partial_spearman_controlling_sequence_length": partial_spearman(
                        clean["feature_value"],
                        clean["delta_ndcg_10"],
                        clean["sequence_length"],
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def subgroup_analysis(
    features: pd.DataFrame,
    effects: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Estimate paired effects within Low/Medium/High feature groups."""
    merged = effects.merge(features, on=["dataset", "user_id"], validate="many_to_one")
    records: list[dict[str, object]] = []
    for dataset, dataset_frame in merged.groupby("dataset", sort=True):
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            user_groups = (
                dataset_frame[["user_id", feature]]
                .drop_duplicates("user_id")
                .set_index("user_id")
            )
            user_groups["feature_group"] = assign_quantile_groups(user_groups[feature])
            for model, model_frame in dataset_frame.groupby("model_name", sort=True):
                grouped = model_frame.join(user_groups["feature_group"], on="user_id")
                for group_index, group in enumerate(("Low", "Medium", "High")):
                    subset = grouped.loc[grouped["feature_group"].eq(group)]
                    if subset.empty:
                        continue
                    lower, upper = bootstrap_mean_interval(
                        subset["delta_ndcg_10"],
                        samples=config.bootstrap_samples,
                        seed=(
                            config.bootstrap_seed
                            + feature_index * 101
                            + group_index * 17
                            + len(records)
                        ),
                    )
                    records.append(
                        {
                            "dataset": dataset,
                            "model_name": model,
                            "feature": feature,
                            "feature_group": group,
                            "user_count": len(subset),
                            "feature_minimum": subset[feature].min(),
                            "feature_maximum": subset[feature].max(),
                            "original_mean_ndcg_10": subset["original_ndcg_10"].mean(),
                            "shuffled_mean_ndcg_10": subset["shuffled_ndcg_10"].mean(),
                            "delta_mean_ndcg_10": subset["delta_ndcg_10"].mean(),
                            "bootstrap_ci_95_lower": lower,
                            "bootstrap_ci_95_upper": upper,
                        }
                    )
    return pd.DataFrame.from_records(records)


def model_contrast_analysis(
    features: pd.DataFrame,
    effects: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Relate sequence features to within-user model Delta contrasts."""
    wide = effects.pivot_table(
        index=["dataset", "user_id"],
        columns="model_name",
        values="delta_ndcg_10",
    ).reset_index()
    merged = wide.merge(features, on=["dataset", "user_id"], validate="one_to_one")
    contrasts = (
        ("fpmc_minus_sasrec", "fpmc", "sasrec"),
        ("fpmc_minus_lstm", "fpmc", "lstm"),
        ("sasrec_minus_lstm", "sasrec", "lstm"),
    )
    records: list[dict[str, object]] = []
    for dataset, frame in merged.groupby("dataset", sort=True):
        for contrast_name, positive_model, negative_model in contrasts:
            if positive_model not in frame or negative_model not in frame:
                continue
            contrast = frame[positive_model] - frame[negative_model]
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                clean = pd.DataFrame(
                    {
                        "feature": frame[feature],
                        "contrast": contrast,
                        "sequence_length": frame["sequence_length"],
                    }
                ).dropna()
                rho, p_value = _rank_correlation(clean["feature"], clean["contrast"])
                lower, upper = bootstrap_correlation_interval(
                    clean["feature"],
                    clean["contrast"],
                    samples=config.bootstrap_samples,
                    seed=config.bootstrap_seed + feature_index + len(records) * 13,
                )
                records.append(
                    {
                        "dataset": dataset,
                        "contrast": contrast_name,
                        "feature": feature,
                        "sample_size": len(clean),
                        "spearman_rho": rho,
                        "bootstrap_ci_95_lower": lower,
                        "bootstrap_ci_95_upper": upper,
                        "raw_p_value": p_value,
                        "partial_spearman_controlling_sequence_length": partial_spearman(
                            clean["feature"],
                            clean["contrast"],
                            clean["sequence_length"],
                        ),
                    }
                )
    result = pd.DataFrame.from_records(records)
    result["bh_fdr_adjusted_p_value"] = result.groupby(
        ["dataset", "contrast"], group_keys=False
    )["raw_p_value"].transform(benjamini_hochberg)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_movietweetings_artifacts(
    metric_path: Path, split_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt frozen MovieTweetings metrics and original pre-test histories."""
    for path in (metric_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required MovieTweetings artifact is missing: {path}")
    metrics = pd.read_csv(metric_path)
    required_metrics = {
        "user_id", "model_name", "order_condition", "ndcg_10",
        "model_initialization_seed", "sequence_shuffle_seed",
    }
    missing = required_metrics.difference(metrics.columns)
    if missing:
        raise ValueError(f"MovieTweetings metrics are missing columns: {sorted(missing)}")
    metrics = metrics.loc[metrics["model_name"].isin(MODELS)].copy()
    required_conditions = {"original_order", "fully_shuffled_pretest"}
    if not required_conditions.issubset(set(metrics["order_condition"])):
        raise ValueError("MovieTweetings Original/Fully Shuffled conditions are missing")
    selected = metrics.loc[metrics["order_condition"].isin(required_conditions)]
    keys = ["user_id", "model_name", "model_initialization_seed"]
    original = selected.loc[selected["order_condition"].eq("original_order")]
    shuffled = selected.loc[selected["order_condition"].eq("fully_shuffled_pretest")]
    if original.duplicated(keys).any() or shuffled.duplicated(keys + ["sequence_shuffle_seed"]).any():
        raise ValueError("MovieTweetings contains duplicate Original/Shuffled seed cells")
    if original["sequence_shuffle_seed"].notna().any() or shuffled["sequence_shuffle_seed"].isna().any():
        raise ValueError("MovieTweetings shuffle-seed assignments are invalid")
    if set(original["model_initialization_seed"]) != set(range(760, 765)):
        raise ValueError("MovieTweetings initialization seeds do not match the frozen run")
    if set(shuffled["sequence_shuffle_seed"]) != {20760, 20761, 20762}:
        raise ValueError("MovieTweetings shuffle seeds do not match the frozen run")
    expected_cells = len(original) * 3
    if len(shuffled) != expected_cells or not shuffled.groupby(keys).size().eq(3).all():
        raise ValueError("MovieTweetings Original/Shuffled seed pairing is incomplete")
    split = pd.read_csv(split_path)
    required_split = {"UserID", "MovieID", "Timestamp", "_source_order", "split"}
    missing = required_split.difference(split.columns)
    if missing:
        raise ValueError(f"MovieTweetings split is missing columns: {sorted(missing)}")
    if not set(split["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("MovieTweetings split has unexpected labels")
    features = extract_feature_table(split, "MovieTweetings")
    if features.duplicated("user_id").any() or set(features["user_id"]) != set(original["user_id"]):
        raise ValueError("MovieTweetings feature users do not match metric users")
    if not features["unique_item_count"].eq(features["sequence_length"]).all():
        raise ValueError("MovieTweetings pre-test histories violate the no-repeat property")
    return metrics.assign(dataset="MovieTweetings"), features


def feature_applicability_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Flag unsupported or degenerate features without changing their values."""
    unsupported_without_repeats = {
        "repeat_event_rate", "immediate_repeat_rate", "visits_per_unique_item",
        "maximum_item_frequency_share", "normalized_item_entropy",
        "repeated_bigram_rate", "local_transition_concentration",
        "first_order_conditional_entropy", "local_information_gain",
        "first_order_supported_transition_rate", "repeated_trigram_rate",
        "second_order_conditional_entropy", "long_range_gain",
        "normalized_long_range_gain", "second_order_supported_transition_rate",
    }
    records = []
    for dataset, frame in features.groupby("dataset", sort=True):
        no_repeats = frame["unique_item_ratio"].eq(1).all()
        for feature in FEATURE_COLUMNS:
            values = pd.to_numeric(frame[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            reliability_column = (
                "first_order_entropy_reliable" if feature in {
                    "first_order_conditional_entropy", "local_information_gain"
                } else "second_order_entropy_reliable" if feature in {
                    "second_order_conditional_entropy", "long_range_gain",
                    "normalized_long_range_gain"
                } else None
            )
            reliable_count = (
                int(frame[reliability_column].sum()) if reliability_column else None
            )
            applicable = (
                finite.nunique() > 2
                and not (no_repeats and feature in unsupported_without_repeats)
                and (reliable_count is None or reliable_count > 0)
            )
            concerns = []
            if finite.nunique() <= 2:
                concerns.append("constant or nearly constant")
            if no_repeats and feature in unsupported_without_repeats:
                concerns.append("no repeated user-item events; mechanical or unsupported")
            if reliable_count == 0:
                concerns.append("no users meet transition reliability rule")
            if feature in EXPLORATORY_FEATURES:
                concerns.append("exploratory sparse-transition estimate")
            if feature in {"unique_item_count", "maximum_item_frequency_share"} and no_repeats:
                concerns.append("deterministic function of sequence length")
            records.append({
                "dataset": dataset, "feature": feature, "applicable": bool(applicable),
                "support_count": int(finite.size), "missing_count": int(values.isna().sum()),
                "variance": float(finite.var()) if len(finite) > 1 else np.nan,
                "unique_value_count": int(finite.nunique()),
                "reliable_user_count": reliable_count,
                "reliability_concerns": "; ".join(concerns),
            })
    return pd.DataFrame.from_records(records)


def cross_dataset_feature_comparison(
    associations: pd.DataFrame, applicability: pd.DataFrame,
    seed_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Present matched feature-model associations and applicability together."""
    stability = seed_stability.groupby(["dataset", "model_name", "feature"]).agg(
        finite_seed_count=("spearman_rho", "count"),
        positive_seed_count=("spearman_rho", lambda values: int((values > 0).sum())),
        negative_seed_count=("spearman_rho", lambda values: int((values < 0).sum())),
    ).reset_index()
    return associations.merge(
        applicability[["dataset", "feature", "applicable", "support_count", "reliability_concerns"]],
        on=["dataset", "feature"], validate="many_to_one",
    ).merge(stability, on=["dataset", "model_name", "feature"], validate="one_to_one")


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ml_metric_paths = [
        Path("outputs/ml1m_order_experiment_rating4_20260817/per_user_model_metrics.csv"),
        *[
            Path(
                f"outputs/ml1m_order_robustness_runs_20260817/"
                f"rating4_ruleA_init{seed}/per_user_model_metrics.csv"
            )
            for seed in range(761, 765)
        ],
        Path(
            "outputs/ml1m_lstm_order_experiment_rating4_init760_20260916/"
            "per_user_model_metrics.csv"
        ),
        *[
            Path(
                f"outputs/ml1m_lstm_robustness_runs_20260916/"
                f"lstm_rating4_ruleA_init{seed}/per_user_model_metrics.csv"
            )
            for seed in range(761, 765)
        ],
    ]
    foursquare_metric_path = Path(
        "outputs/foursquare_nyc_order_experiment_formal_20260929/"
        "per_user_model_metrics.csv"
    )
    movietweetings_dir = Path("outputs/movietweetings_experiment/formal_rating8_5init_20261002")
    movietweetings_metric_path = movietweetings_dir / "per_user_metrics.csv"
    movietweetings_split_path = movietweetings_dir / "split_assignments.csv"
    split_paths = {
        "MovieLens 1M": Path(
            "outputs/ml1m_order_robustness_runs_20260817/rating4_ruleA_init761/"
            "split_audit_table.csv"
        ),
        "Foursquare NYC": Path(
            "outputs/foursquare_nyc_order_experiment_formal_20260929/"
            "split_audit_table.csv"
        ),
    }
    required_paths = ml_metric_paths + [
        foursquare_metric_path, *split_paths.values(),
        movietweetings_metric_path, movietweetings_split_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen artifacts are missing: {missing}")

    ml_metrics = pd.concat(
        [pd.read_csv(path) for path in ml_metric_paths], ignore_index=True
    )
    ml_metrics = ml_metrics.loc[ml_metrics["model_name"].isin(MODELS)]
    foursquare_metrics = pd.read_csv(foursquare_metric_path)
    movietweetings_metrics, movietweetings_features = load_movietweetings_artifacts(
        movietweetings_metric_path, movietweetings_split_path
    )
    metrics = pd.concat(
        [
            ml_metrics.assign(dataset="MovieLens 1M"),
            foursquare_metrics.assign(dataset="Foursquare NYC"),
            movietweetings_metrics,
        ],
        ignore_index=True,
    )
    features = pd.concat(
        [
            *[
                extract_feature_table(pd.read_csv(path), dataset)
                for dataset, path in split_paths.items()
            ],
            movietweetings_features,
        ],
        ignore_index=True,
    )
    audit = pd.DataFrame(
        [
            {
                "dataset": (
                    "MovieLens 1M" if "ml1m" in str(path) else
                    "MovieTweetings" if "movietweetings" in str(path) else "Foursquare NYC"
                ),
                "artifact_type": "per_user_metrics" if "metrics" in path.name else "split_audit",
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in required_paths
        ]
    )
    return metrics, features, audit


def _write_figures(
    output_dir: Path,
    associations: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    display_features = [
        "sequence_length",
        "unique_item_ratio",
        "repeat_event_rate",
        "normalized_item_entropy",
        "repeated_bigram_rate",
        "local_transition_concentration",
        "local_information_gain",
        "repeated_trigram_rate",
        "normalized_long_range_gain",
        "burstiness",
        "short_gap_rate_1_hour",
    ]
    for dataset in associations["dataset"].unique():
        frame = associations.loc[
            associations["dataset"].eq(dataset)
            & associations["feature"].isin(display_features)
        ]
        matrix = frame.pivot(index="feature", columns="model_name", values="spearman_rho")
        matrix = matrix.reindex(display_features).reindex(columns=MODELS)
        fig, axis = plt.subplots(figsize=(7.2, 6.0))
        image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25)
        axis.set_xticks(range(len(matrix.columns)), [name.upper() for name in matrix.columns])
        axis.set_yticks(range(len(matrix.index)), [name.replace("_", " ") for name in matrix.index])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
        axis.set_title(f"{dataset}: sequence feature vs user-level ΔNDCG@10")
        fig.colorbar(image, ax=axis, label="Spearman ρ")
        fig.tight_layout()
        path = figure_dir / f"{dataset.lower().replace(' ', '_')}_feature_model_heatmap.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    strongest = (
        associations.assign(abs_rho=lambda frame: frame["spearman_rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .groupby(["dataset", "model_name"], as_index=False)
        .first()
    )
    selected = subgroups.merge(
        strongest[["dataset", "model_name", "feature"]],
        on=["dataset", "model_name", "feature"],
        how="inner",
    )
    for dataset, frame in selected.groupby("dataset"):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)
        for axis, model in zip(axes, MODELS):
            model_frame = frame.loc[frame["model_name"].eq(model)].copy()
            model_frame["feature_group"] = pd.Categorical(
                model_frame["feature_group"], ["Low", "Medium", "High"], ordered=True
            )
            model_frame = model_frame.sort_values("feature_group")
            axis.errorbar(
                model_frame["feature_group"],
                model_frame["delta_mean_ndcg_10"],
                yerr=np.vstack(
                    [
                        model_frame["delta_mean_ndcg_10"] - model_frame["bootstrap_ci_95_lower"],
                        model_frame["bootstrap_ci_95_upper"] - model_frame["delta_mean_ndcg_10"],
                    ]
                ),
                marker="o",
                capsize=4,
            )
            feature = model_frame["feature"].iloc[0] if not model_frame.empty else ""
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(model.upper())
            axis.set_xlabel(feature.replace("_", " "), fontsize=8)
            axis.set_ylabel("Original − shuffled NDCG@10")
        fig.suptitle(f"{dataset}: strongest univariate subgroup effects")
        fig.tight_layout()
        path = figure_dir / f"{dataset.lower().replace(' ', '_')}_strongest_subgroups.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    profile = associations.loc[associations["feature"].isin(display_features)].copy()
    profile["label"] = profile["dataset"] + " | " + profile["model_name"].str.upper()
    matrix = profile.pivot(
        index="feature",
        columns="label",
        values="partial_spearman_controlling_sequence_length",
    ).reindex(display_features)
    fig, axis = plt.subplots(figsize=(11.5, 6.0))
    image = axis.imshow(matrix.to_numpy(), cmap="PiYG", vmin=-0.2, vmax=0.2)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(matrix.index)), [name.replace("_", " ") for name in matrix.index])
    axis.set_title("Model sequence profiles after controlling sequence length")
    fig.colorbar(image, ax=axis, label="Partial Spearman ρ")
    fig.tight_layout()
    path = figure_dir / "cross_dataset_model_profile.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def _git_information() -> tuple[str, str]:
    safe_directory = str(Path.cwd().resolve()).replace("\\", "/")
    git_prefix = ["git", "-c", f"safe.directory={safe_directory}"]
    commit = subprocess.run(
        [*git_prefix, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        [*git_prefix, "status", "--short"], capture_output=True, text=True, check=True
    ).stdout
    return commit, status


def run_analysis(config: AnalysisConfig) -> Path:
    """Run the complete frozen-artifact explanatory analysis."""
    started = time.perf_counter()
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    commit, git_status = _git_information()
    metrics, features, audit = _load_inputs()

    effects_frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    for dataset, frame in metrics.groupby("dataset", sort=True):
        effects, paired = compute_user_order_effects(frame, dataset)
        effects_frames.append(effects)
        paired_frames.append(paired)
    effects = pd.concat(effects_frames, ignore_index=True)
    paired_effects = pd.concat(paired_frames, ignore_index=True)

    summary = summarize_features(features)
    correlations = feature_correlations(features)
    associations = association_analysis(features, effects, config)
    subgroups = subgroup_analysis(features, effects, config)
    contrasts = model_contrast_analysis(features, effects, config)
    seed_robustness = seed_robustness_analysis(features, paired_effects)
    applicability = feature_applicability_audit(features)
    comparison = cross_dataset_feature_comparison(
        associations, applicability, seed_robustness
    )

    tables = {
        "artifact_audit.csv": audit,
        "user_sequence_features.csv": features,
        "user_model_order_effects.csv": effects,
        "user_model_order_effects_by_initialization.csv": paired_effects,
        "feature_summary_by_dataset.csv": summary,
        "feature_correlations.csv": correlations,
        "feature_model_spearman.csv": associations,
        "subgroup_order_effects.csv": subgroups,
        "model_contrast_analysis.csv": contrasts,
        "feature_model_spearman_by_initialization.csv": seed_robustness,
        "seed_stability.csv": seed_robustness,
        "feature_applicability.csv": applicability,
        "cross_dataset_feature_comparison.csv": comparison,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    figure_paths = _write_figures(output_dir, associations, subgroups)

    analysis_config = {
        **asdict(config),
        "datasets_completed": ["MovieLens 1M", "MovieTweetings", "Foursquare NYC"],
        "datasets_pending": [],
        "feature_history": "original-order train + validation events only",
        "dependent_variable": (
            "within initialization seed: original NDCG@10 minus mean fully shuffled "
            "NDCG@10 across shuffle seeds; then mean across initialization seeds"
        ),
        "relative_user_effect_policy": "reported only when original NDCG@10 is nonzero",
        "timestamp_note": {
            "MovieLens 1M": "rating-submission time, not confirmed viewing time",
            "MovieTweetings": "rating-submission time, not confirmed viewing time",
            "Foursquare NYC": "recorded UTC check-in time",
        },
        "entropy_reliability_rule": (
            f"at least {config.minimum_entropy_transitions} transitions and supported "
            f"transition rate >= {config.minimum_supported_transition_rate}"
        ),
        "regression": "not run; transparent correlation, subgroup, contrast, and seed analyses preferred",
        "git_commit": commit,
        "git_status_before_analysis": git_status,
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2), encoding="utf-8"
    )
    manifest = {
        "runtime_seconds": time.perf_counter() - started,
        "created_files": sorted(
            [*tables.keys(), "analysis_config.json", "analysis_manifest.json"]
            + [str(path.relative_to(output_dir)) for path in figure_paths]
        ),
        "row_counts": {filename: len(table) for filename, table in tables.items()},
        "input_artifacts": audit.to_dict(orient="records"),
        "notes": [
            "No recommender models were trained or rerun.",
            "No frozen experiment outputs were modified.",
            "MovieTweetings uses frozen user metrics and original-order train plus validation history only.",
            "No-repeat features remain numerically unchanged but are flagged when unsupported.",
            "Entropy and higher-order features are exploratory when transition support is sparse.",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/sequence_structure_analysis_20260929"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=50760)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_analysis(
        AnalysisConfig(
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    )
    print(f"Sequence-structure analysis written to {output}")


if __name__ == "__main__":
    main()
