"""Predeclared block-order audit and non-scientific MovieTweetings pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from movie_recommender.ml1m_model_feasibility import FPMC
from movie_recommender.ml1m_order_experiment import build_order_histories
from movie_recommender.ml1m_revised_sequence_feasibility import RevisedConfig, revised_chronological_split
from movie_recommender.movietweetings_audit import DATA_DIR, ROOT, load_movies, load_ratings
from movie_recommender.movietweetings_experiment import (
    compact_sets, evaluate, fpmc_scorer, neural_scorer, train_neural,
)


FORMAL = ROOT / "outputs" / "movietweetings_experiment" / "formal_rating8_5init_20261002"
DEFAULT_OUTPUT = ROOT / "outputs" / "movietweetings_structured_perturbation"
BLOCK_SIZES = (4, 8, 16)
PERTURBATION_SEEDS = (20760, 20761, 20762)
PILOT_INITIALIZATION_SEED = 760
PILOT_PERTURBATION_SEED = 20760
PILOT_USERS = 1500
CONDITIONS = ("original_order", "within_block_shuffle", "block_permutation", "fully_shuffled_pretest")


def block_ranges(length: int, block_size: int) -> list[tuple[int, int]]:
    """Return contiguous ranges, retaining a final partial block."""
    if length < 1 or block_size < 2:
        raise ValueError("length must be positive and block size at least two")
    return [(start, min(start + block_size, length)) for start in range(0, length, block_size)]


def perturb_indices(length: int, block_size: int, condition: str, seed: int) -> np.ndarray:
    """Return original-position indices in the deterministic perturbed order."""
    blocks = [np.arange(start, end, dtype=np.int32) for start, end in block_ranges(length, block_size)]
    rng = np.random.default_rng(seed)
    if condition == "within_block_shuffle":
        return np.concatenate([rng.permutation(block) for block in blocks])
    if condition == "block_permutation":
        return np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])
    raise ValueError(f"Unknown block perturbation: {condition}")


def perturb_sequence(
    sequence: np.ndarray, block_size: int, condition: str, seed: int
) -> np.ndarray:
    """Reorder a copy of the history without changing its item multiset."""
    return np.asarray(sequence)[perturb_indices(len(sequence), block_size, condition, seed)].copy()


def perturbation_strength(original: np.ndarray, perturbed: np.ndarray) -> dict[str, float | bool]:
    """Measure adjacency, displacement, ordering, and fixed positions."""
    if len(original) != len(perturbed) or Counter(original) != Counter(perturbed):
        raise ValueError("Perturbation changed sequence length or item multiset")
    if len(set(original)) != len(original):
        raise ValueError("Position-based audit requires unique user-item events")
    positions = {int(item): index for index, item in enumerate(original)}
    reordered_positions = np.fromiter(
        (positions[int(item)] for item in perturbed), dtype=np.int32, count=len(original)
    )
    length = len(original)
    preserved = np.mean(np.diff(reordered_positions) == 1) if length > 1 else 1.0
    displacement = np.mean(
        np.abs(reordered_positions - np.arange(length)) / max(length - 1, 1)
    )
    rank_correlation = (
        1 - 6 * np.square(reordered_positions - np.arange(length)).sum()
        / (length * (length**2 - 1)) if length > 1 else 1.0
    )
    return {
        "preserved_adjacency_rate": float(preserved),
        "normalized_displacement": float(displacement),
        "rank_correlation": float(rank_correlation),
        "fixed_position_rate": float(np.mean(reordered_positions == np.arange(length))),
        "effectively_unchanged": bool(np.array_equal(original, perturbed)),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _frozen_histories() -> tuple[np.ndarray, list[np.ndarray], pd.DataFrame]:
    split_path = FORMAL / "split_assignments.csv"
    table = pd.read_csv(split_path)
    required = {"UserID", "MovieID", "Timestamp", "_source_order", "split"}
    if not required.issubset(table.columns):
        raise ValueError("Frozen split artifact has an unexpected schema")
    pretest = table.loc[table["split"].isin(("train", "validation"))].copy()
    pretest = pretest.sort_values(
        ["UserID", "Timestamp", "MovieID", "_source_order"], kind="stable"
    )
    users = np.sort(table["UserID"].unique())
    groups = {int(user): frame["MovieID"].to_numpy(dtype=np.int64)
              for user, frame in pretest.groupby("UserID", sort=True)}
    if len(users) != 11269 or set(users) != set(groups):
        raise ValueError("Frozen formal user set is incomplete")
    histories = [groups[int(user)] for user in users]
    if any(len(history) < 2 or len(set(history)) != len(history) for history in histories):
        raise ValueError("Frozen histories violate minimum length or no-repeat policy")
    test_items = table.loc[table["split"].eq("test")].groupby("UserID")["MovieID"].agg(set)
    if any(set(history) & test_items.loc[int(user)] for user, history in zip(users, histories, strict=True)):
        raise ValueError("A test item appears in a pre-test history")
    return users, histories, table


def _user_seed(user_id: int, seed: int) -> int:
    return seed + user_id * 1009


def run_structural_audit(output_dir: Path) -> Path:
    """Audit B=4/8/16 without reading any recommendation performance."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    users, histories, split = _frozen_histories()
    output_dir.mkdir(parents=True)
    records: list[dict] = []
    for user_index, (user_id, original) in enumerate(zip(users, histories, strict=True)):
        length = len(original)
        records.append({
            "user_id": int(user_id), "block_size": 0, "condition": "original_order",
            "perturbation_seed": -1, "sequence_length": length,
            "block_count": 0, **perturbation_strength(original, original),
        })
        for seed in PERTURBATION_SEEDS:
            shuffled = np.random.default_rng(seed + user_index * 1009).permutation(original)
            records.append({
                "user_id": int(user_id), "block_size": 0,
                "condition": "fully_shuffled_pretest", "perturbation_seed": seed,
                "sequence_length": length, "block_count": 0,
                **perturbation_strength(original, shuffled),
            })
            for block_size in BLOCK_SIZES:
                for condition in ("within_block_shuffle", "block_permutation"):
                    candidate = perturb_sequence(
                        original, block_size, condition, _user_seed(int(user_id), seed)
                    )
                    records.append({
                        "user_id": int(user_id), "block_size": block_size,
                        "condition": condition, "perturbation_seed": seed,
                        "sequence_length": length,
                        "block_count": len(block_ranges(length, block_size)),
                        **perturbation_strength(original, candidate),
                    })
    user_audit = pd.DataFrame.from_records(records)
    summaries = []
    for (block_size, condition, seed), frame in user_audit.groupby(
        ["block_size", "condition", "perturbation_seed"], sort=True
    ):
        summaries.append({
            "block_size": int(block_size), "condition": condition,
            "perturbation_seed": int(seed), "users": len(frame),
            "users_at_least_two_blocks": int(frame["block_count"].ge(2).sum()),
            "users_at_least_three_blocks": int(frame["block_count"].ge(3).sum()),
            "fraction_at_least_two_blocks": float(frame["block_count"].ge(2).mean()),
            "fraction_at_least_three_blocks": float(frame["block_count"].ge(3).mean()),
            "preserved_adjacency_rate": float(frame["preserved_adjacency_rate"].mean()),
            "normalized_displacement": float(frame["normalized_displacement"].mean()),
            "rank_correlation": float(frame["rank_correlation"].mean()),
            "fixed_position_rate": float(frame["fixed_position_rate"].mean()),
            "effectively_unchanged_users": int(frame["effectively_unchanged"].sum()),
        })
    structural = pd.DataFrame.from_records(summaries)
    length_bins = pd.cut(
        user_audit["sequence_length"], [0, 8, 16, 32, 64, np.inf],
        labels=["2-8", "9-16", "17-32", "33-64", "65+"],
    )
    user_audit.insert(5, "sequence_length_group", length_bins)
    user_audit.to_csv(output_dir / "user_perturbation_audit.csv", index=False)
    structural.to_csv(output_dir / "structural_audit.csv", index=False)
    by_length = user_audit.groupby(
        ["block_size", "condition", "sequence_length_group"], observed=True
    ).agg(
        users=("user_id", "nunique"),
        preserved_adjacency_rate=("preserved_adjacency_rate", "mean"),
        normalized_displacement=("normalized_displacement", "mean"),
        rank_correlation=("rank_correlation", "mean"),
        effectively_unchanged_rate=("effectively_unchanged", "mean"),
    ).reset_index()
    by_length.to_csv(output_dir / "structural_audit_by_length.csv", index=False)
    config = {
        "formal_output": str(FORMAL), "formal_split_sha256": _file_hash(FORMAL / "split_assignments.csv"),
        "formal_metric_sha256": _file_hash(FORMAL / "per_user_metrics.csv"),
        "block_sizes_audited": list(BLOCK_SIZES),
        "perturbation_seeds": list(PERTURBATION_SEEDS),
        "selection_rule": (
            "Among B=4,8,16, choose the largest B with at least 80% of users having "
            "three blocks; if none, choose B=4. No ranking metrics are read."
        ),
        "history": "frozen Original train+validation positive sequence; test excluded",
        "full_shuffle_audit": "reconstructed from frozen formal seed and full-user index; formal scores untouched",
        "block_seed_rule": "perturbation_seed + user_id * 1009",
        "pilot_users": PILOT_USERS, "pilot_initialization_seed": PILOT_INITIALIZATION_SEED,
        "pilot_perturbation_seed": PILOT_PERTURBATION_SEED,
        "pilot_policy": "all four conditions retrained on the same 1500-user subset; no tuning",
        "full_run_policy": "reuse frozen formal Original/Full metrics; train only new block conditions",
    }
    eligible = []
    for block_size in BLOCK_SIZES:
        fraction = structural.loc[
            structural["block_size"].eq(block_size)
            & structural["condition"].eq("within_block_shuffle"),
            "fraction_at_least_three_blocks",
        ].iloc[0]
        if fraction >= 0.8:
            eligible.append(block_size)
    config["primary_block_size"] = max(eligible) if eligible else 4
    config["primary_block_size_reason"] = (
        "Largest audited block meeting the predeclared three-block coverage rule"
        if eligible else "Predeclared B=4 fallback because no candidate met coverage"
    )
    config["formal_users"] = len(users)
    config["formal_positive_split_rows"] = len(split)
    _write_json(output_dir / "perturbation_config.json", config)
    print(json.dumps({"primary_block_size": config["primary_block_size"],
                      "structural_summary": summaries}, indent=2), flush=True)
    return output_dir


def run_pilot(output_dir: Path) -> Path:
    """Run a fixed one-init/one-perturbation-seed correctness pilot."""
    started = time.perf_counter()
    config_path = output_dir / "perturbation_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Run the structural audit before the pilot")
    for name in ("pilot_metrics.csv", "runtime_summary.json", "pilot_integrity.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing pilot output: {name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    block_size = int(config["primary_block_size"])
    if _file_hash(FORMAL / "split_assignments.csv") != config["formal_split_sha256"]:
        raise ValueError("Frozen split artifact changed after block-size selection")
    if _file_hash(FORMAL / "per_user_metrics.csv") != config["formal_metric_sha256"]:
        raise ValueError("Frozen metric artifact changed after block-size selection")
    selected_path = FORMAL / f"selected_parameters_init{PILOT_INITIALIZATION_SEED}.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    ratings = load_ratings(DATA_DIR / "ratings.dat")
    movies = load_movies(DATA_DIR / "movies.dat")
    split = revised_chronological_split(
        ratings, RevisedConfig(relevance_threshold=8, user_limit=PILOT_USERS, random_seed=40760)
    )
    users = split.eligible_users["user_id"].to_numpy(dtype=np.int64)
    if len(users) != PILOT_USERS:
        raise ValueError("Pilot user count differs from the predeclared sample")
    frozen_split = pd.read_csv(FORMAL / "split_assignments.csv")
    frozen_selected = frozen_split.loc[frozen_split["UserID"].isin(users)]
    columns = ["UserID", "MovieID", "Rating", "Timestamp", "_source_order", "split"]
    left = split.split_table[columns].sort_values(columns).reset_index(drop=True)
    right = frozen_selected[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    movie_ids = pd.read_csv(FORMAL / "item_mapping.csv")["MovieID"].to_numpy(dtype=np.int64)
    if not np.array_equal(movie_ids, np.sort(movies["MovieID"].unique())):
        raise ValueError("Formal catalog differs from raw metadata")
    mapping = {int(movie): index for index, movie in enumerate(movie_ids)}
    histories = build_order_histories(split, users, mapping, [PILOT_PERTURBATION_SEED])
    test = compact_sets(split, split.test_positive, movie_ids)
    sequences = {
        "original_order": histories.original,
        "fully_shuffled_pretest": histories.fully_shuffled[PILOT_PERTURBATION_SEED],
        "within_block_shuffle": [
            perturb_sequence(sequence, block_size, "within_block_shuffle", _user_seed(int(user), PILOT_PERTURBATION_SEED))
            for user, sequence in zip(users, histories.original, strict=True)
        ],
        "block_permutation": [
            perturb_sequence(sequence, block_size, "block_permutation", _user_seed(int(user), PILOT_PERTURBATION_SEED))
            for user, sequence in zip(users, histories.original, strict=True)
        ],
    }
    mismatch_count = 0
    for condition in CONDITIONS:
        for original, candidate in zip(histories.original, sequences[condition], strict=True):
            mismatch_count += int(len(original) != len(candidate) or not np.array_equal(np.sort(original), np.sort(candidate)))
    if mismatch_count:
        raise AssertionError("Pilot perturbation changed an item multiset or sequence length")
    test_targets = split.test_positive.groupby("UserID")["MovieID"].agg(set)
    leakage = sum(
        bool(set(movie_ids[sequence]) & test_targets.loc[int(user)])
        for user, sequence in zip(users, histories.original, strict=True)
    )
    if leakage or any(np.intersect1d(excluded, targets).size for excluded, targets in zip(test.excluded, test.targets, strict=True)):
        raise AssertionError("Pilot test-target leakage or candidate exclusion error")
    if not np.array_equal(test.user_ids, users):
        raise AssertionError("Candidate sets changed pilot user order")
    if any(not np.array_equal(
        perturb_sequence(sequence, block_size, condition, _user_seed(int(user), PILOT_PERTURBATION_SEED)),
        replay,
    ) for condition in ("within_block_shuffle", "block_permutation")
          for user, sequence, replay in zip(users, histories.original, sequences[condition], strict=True)):
        raise AssertionError("Pilot perturbations are not deterministic")
    torch.set_num_threads(4)
    frames = []
    runtimes = []
    for model_name in ("fpmc", "lstm", "sasrec"):
        choice = selected[model_name]
        for condition in CONDITIONS:
            started_cell = time.perf_counter()
            sequence_rows = sequences[condition]
            if model_name == "fpmc":
                model = FPMC(
                    len(users), len(movie_ids), choice["factors"], .03, .002,
                    PILOT_INITIALIZATION_SEED, 17000 + PILOT_INITIALIZATION_SEED,
                ).fit(sequence_rows, choice["epochs"])
                scorer = fpmc_scorer(model, sequence_rows, choice["transition_weight"])
            else:
                model, _, _, _ = train_neural(
                    model_name, sequence_rows, len(movie_ids), choice["parameters"],
                    choice["epochs"], f"pilot_{model_name}_{condition}",
                    initialization_seed=PILOT_INITIALIZATION_SEED,
                )
                length = choice["parameters"].get("max_sequence_length", 64)
                scorer = neural_scorer(model, sequence_rows, model_name, length)
            frame = evaluate(test, scorer)
            if len(frame) != PILOT_USERS or not np.isfinite(frame["ndcg_10"]).all():
                raise AssertionError("Pilot produced missing or nonfinite ranking metrics")
            frames.append(frame.assign(
                model_name=model_name, order_condition=condition,
                model_initialization_seed=PILOT_INITIALIZATION_SEED,
                sequence_shuffle_seed=(np.nan if condition == "original_order" else PILOT_PERTURBATION_SEED),
            ))
            elapsed = time.perf_counter() - started_cell
            runtimes.append({"model_name": model_name, "condition": condition, "seconds": elapsed})
            print(f"Pilot {model_name} {condition}: {frame.ndcg_10.mean():.6f}, {elapsed:.1f}s", flush=True)
            del model, scorer
    metrics = pd.concat(frames, ignore_index=True)
    if metrics.duplicated(["user_id", "model_name", "order_condition"]).any():
        raise AssertionError("Pilot has duplicate user/model/condition rows")
    metrics.to_csv(output_dir / "pilot_metrics.csv", index=False)
    runtime = pd.DataFrame(runtimes)
    formal_runtime = {}
    for model_name in ("fpmc", "lstm", "sasrec"):
        times = []
        for initialization_seed in range(760, 765):
            for seed in PERTURBATION_SEEDS:
                path = FORMAL / f"{model_name}_fully_shuffled_pretest_{seed}_init{initialization_seed}_runtime.json"
                times.append(json.loads(path.read_text(encoding="utf-8"))["seconds"])
        formal_runtime[model_name] = {
            "pilot_four_conditions_seconds": float(runtime.loc[runtime["model_name"].eq(model_name), "seconds"].sum()),
            "formal_reference_mean_condition_seconds": float(np.mean(times)),
            "projected_new_full_cells": 2 * 5 * 3,
            "projected_new_full_seconds": float(np.mean(times) * 2 * 5 * 3),
        }
    runtime_payload = {
        "pilot_total_seconds": time.perf_counter() - started,
        "pilot_cell_runtimes": runtimes,
        "per_model": formal_runtime,
        "projected_full_new_conditions_seconds": sum(row["projected_new_full_seconds"] for row in formal_runtime.values()),
        "projection_note": "Formal full-shuffle per-cell times scaled to 30 new block cells per model; Original/Full scores reused later, not retrained.",
        "peak_process_rss_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)),
    }
    _write_json(output_dir / "runtime_summary.json", runtime_payload)
    integrity = {
        "complete": True, "users": len(users), "catalog_items": len(movie_ids),
        "models": ["fpmc", "lstm", "sasrec"], "conditions": list(CONDITIONS),
        "metric_rows": len(metrics), "all_ndcg_finite": bool(np.isfinite(metrics["ndcg_10"]).all()),
        "frozen_split_matches_subset": True, "candidate_set_shared_across_conditions": True,
        "test_targets_shared_across_conditions": True, "pretest_test_item_leakage_users": leakage,
        "multiset_or_length_mismatches": mismatch_count,
        "perturbations_deterministic": True, "block_size_selected_before_scores": block_size,
        "frozen_selected_parameters_sha256": _file_hash(selected_path),
        "frozen_selected_parameters": selected,
        "pilot_is_not_scientific_inference": True,
        "formal_original_and_full_checkpoints_reused_in_pilot": False,
        "reason": "Formal checkpoints have 11269-user training; all pilot conditions must share the 1500-user training population.",
    }
    _write_json(output_dir / "pilot_integrity.json", integrity)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "pilot"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.phase == "audit":
        run_structural_audit(args.output_dir)
    else:
        run_pilot(args.output_dir)


if __name__ == "__main__":
    main()
