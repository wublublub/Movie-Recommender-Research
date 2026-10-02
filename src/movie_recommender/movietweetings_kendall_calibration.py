"""Calibrate local and broad MovieTweetings perturbations without model scores."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.movietweetings_structured_perturbation import (
    FORMAL, PERTURBATION_SEEDS, ROOT, _file_hash, _frozen_histories,
    _write_json, perturbation_strength,
)


OUTPUT_DIR = ROOT / "outputs" / "movietweetings_kendall_calibration_20261003"
BUDGETS = (0.05, 0.10, 0.15, 0.20)
TOLERANCE = 0.01
LOCAL_WINDOW = 4
LONG_HISTORY_LOCAL_WINDOW = 8
MINIMUM_COVERAGE = 0.80
MINIMUM_ADJACENCY_GAP = 0.20
MINIMUM_DIRECTION_FRACTION = 0.75


def normalized_kendall_distance(original: np.ndarray, perturbed: np.ndarray) -> float:
    """Return exact normalized inversion distance for unique-item sequences."""
    original = np.asarray(original)
    perturbed = np.asarray(perturbed)
    if len(original) != len(perturbed) or Counter(original) != Counter(perturbed):
        raise ValueError("Kendall distance requires identical item multisets")
    if len(set(original)) != len(original):
        raise ValueError("Kendall distance requires unique item IDs")
    length = len(original)
    if length < 2:
        raise ValueError("Kendall distance requires at least two items")
    ranks = {int(item): index for index, item in enumerate(original)}
    tree = [0] * (length + 1)

    def prefix_sum(index: int) -> int:
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    inversions = 0
    for seen, item in enumerate(perturbed):
        rank = ranks[int(item)] + 1
        inversions += seen - prefix_sum(rank)
        while rank <= length:
            tree[rank] += 1
            rank += rank & -rank
    return inversions / (length * (length - 1) // 2)


@lru_cache(maxsize=None)
def _window_options(length: int) -> dict[int, tuple[tuple[int, ...], ...]]:
    options: dict[int, list[tuple[int, ...]]] = {}
    for permutation in itertools.permutations(range(length)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(length) for right in range(left + 1, length)
        )
        options.setdefault(inversions, []).append(permutation)
    return {key: tuple(value) for key, value in options.items()}


def local_window_size(length: int) -> int:
    """Keep windows small relative to short histories and feasible for long ones."""
    return LOCAL_WINDOW if length <= 32 else LONG_HISTORY_LOCAL_WINDOW


def local_budget_indices(length: int, target: float, seed: int) -> np.ndarray:
    """Spend a feasible inversion budget within disjoint four-item windows."""
    if length < 2 or not 0 <= target <= 1:
        raise ValueError("Invalid local sequence length or Kendall target")
    rng = np.random.default_rng(seed)
    window_size = local_window_size(length)
    windows = [(start, min(start + window_size, length))
               for start in range(0, length, window_size)]
    capacities = np.array([(end - start) * (end - start - 1) // 2
                           for start, end in windows], dtype=np.int32)
    requested = round(target * (length * (length - 1) // 2))
    remaining = min(requested, int(capacities.sum()))
    assigned = np.zeros(len(windows), dtype=np.int32)
    order = rng.permutation(len(windows))
    while remaining:
        for index in order:
            if assigned[index] < capacities[index]:
                assigned[index] += 1
                remaining -= 1
                if remaining == 0:
                    break
    pieces = []
    for (start, end), inversions in zip(windows, assigned, strict=True):
        choices = _window_options(end - start)[int(inversions)]
        choice = choices[int(rng.integers(len(choices)))]
        pieces.extend(start + offset for offset in choice)
    return np.asarray(pieces, dtype=np.int32)


@lru_cache(maxsize=None)
def _broad_shapes(length: int, target: float) -> tuple[tuple[int, int], ...]:
    """Choose adjacent stable-block swaps near the requested inversion count."""
    total_pairs = length * (length - 1) // 2
    requested = target * total_pairs
    candidates: set[tuple[int, int]] = set()
    for left_length in range(2, length):
        nearest = requested / left_length
        for right_length in {
            1, length - left_length,
            math.floor(nearest), math.ceil(nearest), round(nearest),
        }:
            if 1 <= right_length <= length - left_length:
                candidates.add((left_length, right_length))
    if not candidates:
        raise ValueError("No broad block swap is possible")
    feasible = [shape for shape in candidates
                if abs(shape[0] * shape[1] / total_pairs - target) <= TOLERANCE]
    if feasible:
        best_error = min(abs(shape[0] * shape[1] / total_pairs - target)
                         for shape in feasible)
        accurate = [shape for shape in feasible
                    if abs(abs(shape[0] * shape[1] / total_pairs - target) - best_error) < 1e-12]
        best_balance = max(min(shape) for shape in accurate)
        selected = [shape for shape in accurate if min(shape) == best_balance]
    else:
        best_error = min(abs(shape[0] * shape[1] / total_pairs - target)
                         for shape in candidates)
        nearest = [shape for shape in candidates
                   if abs(abs(shape[0] * shape[1] / total_pairs - target) - best_error) < 1e-12]
        best_balance = max(min(shape) for shape in nearest)
        selected = [shape for shape in nearest if min(shape) == best_balance]
    return tuple(sorted(selected))


def broad_budget_indices(length: int, target: float, seed: int) -> np.ndarray:
    """Swap two adjacent intact blocks, preserving order inside both."""
    if length < 3 or not 0 <= target <= 1:
        raise ValueError("Invalid broad sequence length or Kendall target")
    rng = np.random.default_rng(seed)
    shapes = _broad_shapes(length, target)
    left_length, right_length = shapes[int(rng.integers(len(shapes)))]
    start = int(rng.integers(length - left_length - right_length + 1))
    left = np.arange(start, start + left_length, dtype=np.int32)
    right = np.arange(start + left_length, start + left_length + right_length, dtype=np.int32)
    return np.concatenate((
        np.arange(start, dtype=np.int32), right, left,
        np.arange(start + left_length + right_length, length, dtype=np.int32),
    ))


def _length_group(lengths: pd.Series) -> pd.Series:
    return pd.cut(lengths, [0, 8, 16, 32, 64, np.inf],
                  labels=["2-8", "9-16", "17-32", "33-64", "65+to"])


def structurally_eligible_users(
    paired: pd.DataFrame, budget: float, tolerance: float = TOLERANCE
) -> np.ndarray:
    """Select users solely by all-seed structural feasibility and matching."""
    rows = paired.loc[paired["target_kendall_distance"].eq(budget)]
    valid = rows.loc[
        rows["local_target_error"].le(tolerance)
        & rows["broad_target_error"].le(tolerance)
        & rows["pair_kendall_mismatch"].le(tolerance)
        & ~rows["local_effectively_unchanged"]
        & ~rows["broad_effectively_unchanged"]
    ]
    seeds_per_user = valid.groupby("user_id")["perturbation_seed"].nunique()
    return np.sort(seeds_per_user.index[seeds_per_user.eq(len(PERTURBATION_SEEDS))].to_numpy(dtype=np.int64))


def _distribution(values: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        **{f"{prefix}_p{int(percentile * 100)}": float(values.quantile(percentile))
           for percentile in (0.05, 0.25, 0.75, 0.95)},
    }


def run_calibration(output_dir: Path = OUTPUT_DIR) -> Path:
    """Audit candidate budgets and freeze a choice without reading model metrics."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    users, histories, split = _frozen_histories()
    output_dir.mkdir(parents=True)
    records: list[dict] = []
    for user_id, original in zip(users, histories, strict=True):
        length = len(original)
        total_pairs = length * (length - 1) // 2
        for budget_index, budget in enumerate(BUDGETS):
            for perturbation_seed in PERTURBATION_SEEDS:
                seed = int(user_id) * 1009 + perturbation_seed + budget_index * 1_000_003
                local_indices = local_budget_indices(length, budget, seed)
                broad_indices = broad_budget_indices(length, budget, seed + 100_000_019)
                local = original[local_indices]
                broad = original[broad_indices]
                if not np.array_equal(np.sort(original), np.sort(local)) or not np.array_equal(np.sort(original), np.sort(broad)):
                    raise AssertionError("Perturbation changed an item multiset")
                window_size = local_window_size(length)
                if any(index // window_size != position // window_size
                       for position, index in enumerate(local_indices)):
                    raise AssertionError("Local item crossed a window boundary")
                local_kendall = normalized_kendall_distance(original, local)
                broad_kendall = normalized_kendall_distance(original, broad)
                local_strength = perturbation_strength(original, local)
                broad_strength = perturbation_strength(original, broad)
                records.append({
                    "user_id": int(user_id), "sequence_length": length,
                    "target_kendall_distance": budget,
                    "perturbation_seed": perturbation_seed,
                    "local_window_size": window_size,
                    "local_kendall_distance": local_kendall,
                    "broad_kendall_distance": broad_kendall,
                    "local_inversions": round(local_kendall * total_pairs),
                    "broad_inversions": round(broad_kendall * total_pairs),
                    "local_target_error": abs(local_kendall - budget),
                    "broad_target_error": abs(broad_kendall - budget),
                    "pair_kendall_mismatch": abs(local_kendall - broad_kendall),
                    **{f"local_{key}": value for key, value in local_strength.items()},
                    **{f"broad_{key}": value for key, value in broad_strength.items()},
                })
    paired = pd.DataFrame.from_records(records)
    paired["sequence_length_group"] = _length_group(paired["sequence_length"])
    paired.to_csv(output_dir / "user_seed_budget_audit.csv", index=False)
    summaries = []
    selected_users_by_budget = {}
    for budget, frame in paired.groupby("target_kendall_distance", sort=True):
        eligible_users = structurally_eligible_users(paired, float(budget))
        selected_users_by_budget[budget] = eligible_users
        eligible = frame.loc[frame["user_id"].isin(eligible_users)]
        local_valid = frame["local_target_error"].le(TOLERANCE) & ~frame["local_effectively_unchanged"]
        broad_valid = frame["broad_target_error"].le(TOLERANCE) & ~frame["broad_effectively_unchanged"]
        both_valid = local_valid & broad_valid & frame["pair_kendall_mismatch"].le(TOLERANCE)
        record = {
            "target_kendall_distance": budget,
            "all_users": len(users),
            "local_valid_users_all_seeds": int(frame.loc[local_valid].groupby("user_id")["perturbation_seed"].nunique().eq(3).sum()),
            "broad_valid_users_all_seeds": int(frame.loc[broad_valid].groupby("user_id")["perturbation_seed"].nunique().eq(3).sum()),
            "both_valid_users_all_seeds": len(eligible_users),
            "coverage": len(eligible_users) / len(users),
            "both_valid_user_seed_cells": int(both_valid.sum()),
            "local_unchanged_user_seed_cells": int(frame["local_effectively_unchanged"].sum()),
            "broad_unchanged_user_seed_cells": int(frame["broad_effectively_unchanged"].sum()),
            "selected_local_lower_adjacency_fraction": float((
                eligible["local_preserved_adjacency_rate"]
                < eligible["broad_preserved_adjacency_rate"]
            ).mean()) if len(eligible) else None,
            "selected_adjacency_gap_median": float((
                eligible["broad_preserved_adjacency_rate"]
                - eligible["local_preserved_adjacency_rate"]
            ).median()) if len(eligible) else None,
        }
        for family in ("local", "broad"):
            for measure in ("kendall_distance", "target_error", "preserved_adjacency_rate",
                            "normalized_displacement", "rank_correlation", "fixed_position_rate"):
                record.update(_distribution(frame[f"{family}_{measure}"], f"{family}_{measure}"))
        record.update(_distribution(frame["pair_kendall_mismatch"], "pair_kendall_mismatch"))
        summaries.append(record)
    summary = pd.DataFrame.from_records(summaries)
    summary.to_csv(output_dir / "candidate_budget_summary.csv", index=False)
    by_length = paired.groupby(["target_kendall_distance", "sequence_length_group"], observed=True).agg(
        users=("user_id", "nunique"),
        local_kendall_median=("local_kendall_distance", "median"),
        broad_kendall_median=("broad_kendall_distance", "median"),
        pair_mismatch_median=("pair_kendall_mismatch", "median"),
        local_adjacency_mean=("local_preserved_adjacency_rate", "mean"),
        broad_adjacency_mean=("broad_preserved_adjacency_rate", "mean"),
    ).reset_index()
    by_length.to_csv(output_dir / "calibration_by_sequence_length.csv", index=False)
    unique_lengths = paired[["user_id", "sequence_length", "sequence_length_group"]].drop_duplicates("user_id")
    coverage_records = []
    for budget, included_users in selected_users_by_budget.items():
        frame = unique_lengths.assign(included=unique_lengths["user_id"].isin(included_users))
        for length_group, group in frame.groupby("sequence_length_group", observed=True):
            coverage_records.append({
                "target_kendall_distance": budget,
                "sequence_length_group": str(length_group),
                "all_users": len(group),
                "included_users": int(group["included"].sum()),
                "coverage": float(group["included"].mean()),
            })
    pd.DataFrame.from_records(coverage_records).to_csv(
        output_dir / "coverage_by_sequence_length.csv", index=False
    )
    passing = summary.loc[
        summary["coverage"].ge(MINIMUM_COVERAGE)
        & summary["pair_kendall_mismatch_median"].le(TOLERANCE)
        & summary["selected_adjacency_gap_median"].ge(MINIMUM_ADJACENCY_GAP)
        & summary["selected_local_lower_adjacency_fraction"].ge(MINIMUM_DIRECTION_FRACTION)
    ]
    selected_budget = float(passing["target_kendall_distance"].max()) if len(passing) else None
    if selected_budget is not None:
        pd.Series(selected_users_by_budget[selected_budget], name="user_id").to_csv(
            output_dir / "included_user_ids.csv", index=False
        )
    _write_json(output_dir / "calibration_config.json", {
        "dataset": "MovieTweetings", "formal_split": str(FORMAL / "split_assignments.csv"),
        "formal_split_sha256": _file_hash(FORMAL / "split_assignments.csv"),
        "candidate_budgets": list(BUDGETS), "kendall_tolerance": TOLERANCE,
        "local_window_rule": "4-item windows for histories of length <=32; 8-item windows for longer histories",
        "local_family": "permutation inside disjoint local windows; inversion count allocated across windows",
        "broad_family": "stable swap of two adjacent intact blocks with tunable block lengths and random position",
        "selection_rule": (
            "Largest candidate budget with >=80% all-three-seed paired-user coverage, "
            "per-user-seed Local and Broad target errors and pair mismatch <=0.01, "
            "both changed for every retained user/seed, median Broad-minus-Local adjacency "
            ">=0.20, and Local adjacency lower in >=75% of retained user-seed cells. "
            "No model score or outcome is read."
        ),
        "all_seed_user_eligibility": True,
        "perturbation_seeds": list(PERTURBATION_SEEDS),
        "source_history": "frozen original-order train+validation positives only",
        "calibration_history": (
            "Structural-only iterations: fixed four-item windows lacked 80% coverage; "
            "adaptive 4/8-item windows improved coverage; the broad tie-break was "
            "changed to prioritize Kendall-target accuracy. No model scores were read."
        ),
        "selected_primary_budget": selected_budget,
        "structurally_eligible_users": len(selected_users_by_budget[selected_budget]) if selected_budget is not None else 0,
        "future_model_run_constraint": (
            "To reuse frozen Original and Full scores, train new Local and Broad conditions "
            "on all 11269 formal users with the frozen model settings; use the closest "
            "structural perturbation for ineligible users, and restrict primary evaluation "
            "and paired inference to included_user_ids.csv. Do not train new conditions "
            "only on the subset while comparing them with full-population checkpoints."
        ),
        "model_training_or_scoring_performed": False,
    })
    _write_json(output_dir / "integrity.json", {
        "formal_users": len(users), "frozen_split_rows": len(split),
        "audited_user_seed_budget_cells": len(paired),
        "item_multisets_and_lengths_preserved": True,
        "local_window_boundaries_preserved": True,
        "broad_blocks_internally_ordered": True,
        "targets_and_candidates_unchanged_by_permutations": True,
        "test_items_excluded_from_original_and_perturbed_histories": True,
        "pilot_metrics_opened": False,
        "models_run": False,
    })
    print(summary[["target_kendall_distance", "local_valid_users_all_seeds",
                   "broad_valid_users_all_seeds", "both_valid_users_all_seeds", "coverage",
                   "pair_kendall_mismatch_median", "selected_adjacency_gap_median",
                   "selected_local_lower_adjacency_fraction"]].to_string(index=False))
    print(f"Selected budget: {selected_budget}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_calibration(args.output_dir)


if __name__ == "__main__":
    main()
