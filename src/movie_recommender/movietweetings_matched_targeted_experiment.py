"""Run the frozen matched Local/Broad MovieTweetings experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from movie_recommender.ml1m_model_feasibility import FPMC
from movie_recommender.ml1m_revised_sequence_feasibility import RevisedConfig, revised_chronological_split
from movie_recommender.movietweetings_audit import DATA_DIR, ROOT, load_movies, load_ratings
from movie_recommender.movietweetings_experiment import (
    compact_sets, evaluate, fpmc_scorer, neural_scorer, train_neural,
)
from movie_recommender.movietweetings_kendall_calibration import (
    OUTPUT_DIR as CALIBRATION_DEFAULT, broad_budget_indices, local_budget_indices,
    normalized_kendall_distance,
)
from movie_recommender.movietweetings_structured_perturbation import (
    FORMAL, _file_hash, _write_json, perturbation_strength,
)


CALIBRATION = ROOT / "outputs" / "movietweetings_kendall_calibration_20261003_v6"
OUTPUT = ROOT / "outputs" / "movietweetings_matched_targeted_formal_20261003"
INITIALIZATIONS = (760, 761, 762, 763, 764)
PERTURBATIONS = (20760, 20761, 20762)
MODELS = ("fpmc", "lstm", "sasrec")
NEW_CONDITIONS = ("matched_local", "matched_broad")
ALL_CONDITIONS = ("original_order", *NEW_CONDITIONS, "fully_shuffled_pretest")
TARGET_BUDGET = 0.10
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 30760


def expected_new_cells() -> list[tuple[str, int, int, str]]:
    """Return the frozen 90-cell grid in deterministic execution order."""
    return [
        (model, initialization, perturbation, condition)
        for model in MODELS
        for initialization in INITIALIZATIONS
        for perturbation in PERTURBATIONS
        for condition in NEW_CONDITIONS
    ]


def _git_hash() -> str:
    safe = str(ROOT).replace("\\", "/")
    return subprocess.run(
        ["git", "-c", f"safe.directory={safe}", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _git_status() -> str:
    safe = str(ROOT).replace("\\", "/")
    return subprocess.run(
        ["git", "-c", f"safe.directory={safe}", "status", "--short"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout


def _read_selected_parameters() -> tuple[dict[int, dict], dict[int, str]]:
    settings = {}
    hashes = {}
    for initialization in INITIALIZATIONS:
        path = FORMAL / f"selected_parameters_init{initialization}.json"
        settings[initialization] = json.loads(path.read_text(encoding="utf-8"))
        hashes[initialization] = _file_hash(path)
        if set(settings[initialization]) != set(MODELS):
            raise ValueError(f"Frozen settings are incomplete: {path}")
    return settings, hashes


def _load_frozen_design() -> tuple[dict, np.ndarray, pd.DataFrame]:
    config_path = CALIBRATION / "calibration_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["selected_primary_budget"] != TARGET_BUDGET:
        raise ValueError("Frozen Kendall budget changed")
    if config["structurally_eligible_users"] != 9287:
        raise ValueError("Frozen matched user count changed")
    if _file_hash(FORMAL / "split_assignments.csv") != config["formal_split_sha256"]:
        raise ValueError("Frozen formal split changed")
    matched = pd.read_csv(CALIBRATION / "included_user_ids.csv")["user_id"].to_numpy(dtype=np.int64)
    if len(matched) != 9287 or len(np.unique(matched)) != 9287 or not np.array_equal(matched, np.sort(matched)):
        raise ValueError("Matched user IDs are incomplete, duplicated, or unsorted")
    audit = pd.read_csv(CALIBRATION / "user_seed_budget_audit.csv")
    audit = audit.loc[audit["target_kendall_distance"].eq(TARGET_BUDGET)]
    if len(audit) != 11269 * len(PERTURBATIONS):
        raise ValueError("Frozen per-user structural audit is incomplete")
    return config, matched, audit


def _preflight() -> dict:
    """Reconstruct the formal population, split, catalog, and perturbations."""
    started = time.perf_counter()
    design, matched_users, frozen_audit = _load_frozen_design()
    settings, settings_hashes = _read_selected_parameters()
    for model_name in MODELS:
        for initialization in INITIALIZATIONS:
            original_path = FORMAL / f"{model_name}_original_order_None_init{initialization}_metrics.csv"
            if not original_path.is_file():
                raise FileNotFoundError(f"Frozen Original cell is missing: {original_path}")
            for perturbation in PERTURBATIONS:
                full_path = FORMAL / f"{model_name}_fully_shuffled_pretest_{perturbation}_init{initialization}_metrics.csv"
                if not full_path.is_file():
                    raise FileNotFoundError(f"Frozen Full Shuffle cell is missing: {full_path}")
    ratings = load_ratings(DATA_DIR / "ratings.dat")
    movies = load_movies(DATA_DIR / "movies.dat")
    formal_protocol = json.loads((FORMAL / "protocol.json").read_text(encoding="utf-8"))
    if _file_hash(DATA_DIR / "ratings.dat") != formal_protocol["source_files"]["ratings.dat"]["sha256"]:
        raise ValueError("Pinned MovieTweetings ratings changed")
    if _file_hash(DATA_DIR / "movies.dat") != formal_protocol["source_files"]["movies.dat"]["sha256"]:
        raise ValueError("Pinned MovieTweetings catalog changed")
    split = revised_chronological_split(
        ratings, RevisedConfig(relevance_threshold=8, random_seed=40760)
    )
    users = split.eligible_users["user_id"].to_numpy(dtype=np.int64)
    if len(users) != 11269 or not np.array_equal(users, np.sort(users)):
        raise ValueError("Formal training population changed")
    if not np.isin(matched_users, users).all():
        raise ValueError("Matched users are not a subset of formal users")
    columns = ["UserID", "MovieID", "Rating", "Timestamp", "_source_order", "split"]
    frozen_split = pd.read_csv(FORMAL / "split_assignments.csv", usecols=columns)
    actual_split = split.split_table[columns]
    pd.testing.assert_frame_equal(
        actual_split.sort_values(columns).reset_index(drop=True),
        frozen_split.sort_values(columns).reset_index(drop=True),
        check_dtype=False,
    )
    movie_ids = pd.read_csv(FORMAL / "item_mapping.csv")["MovieID"].to_numpy(dtype=np.int64)
    if len(movie_ids) != 38013 or not np.array_equal(movie_ids, np.sort(movies["MovieID"].unique())):
        raise ValueError("Formal 38,013-item catalog changed")
    mapping = {int(movie): index for index, movie in enumerate(movie_ids)}
    pretest = split.split_table.loc[split.split_table["split"].isin(("train", "validation"))]
    pretest = pretest.sort_values(["UserID", "Timestamp", "MovieID", "_source_order"], kind="stable")
    grouped = {int(user): frame["MovieID"].to_numpy(dtype=np.int64)
               for user, frame in pretest.groupby("UserID", sort=True)}
    original_movie_ids = [grouped[int(user)] for user in users]
    original = [np.asarray([mapping[int(movie)] for movie in history], dtype=np.int32)
                for history in original_movie_ids]
    test = compact_sets(split, split.test_positive, movie_ids)
    if not np.array_equal(test.user_ids, users):
        raise ValueError("Test candidate user order differs from formal population")
    formal_reference = pd.read_csv(
        FORMAL / "fpmc_original_order_None_init760_metrics.csv",
        usecols=["user_id", "number_of_candidates", "number_of_test_targets"],
    )
    if not np.array_equal(formal_reference["user_id"], users):
        raise ValueError("Formal score users differ from reconstructed users")
    if not np.array_equal(formal_reference["number_of_candidates"],
                          np.asarray([len(movie_ids) - len(excluded) for excluded in test.excluded])):
        raise ValueError("Reconstructed candidate counts differ from formal")
    if not np.array_equal(formal_reference["number_of_test_targets"],
                          np.asarray([len(targets) for targets in test.targets])):
        raise ValueError("Reconstructed target counts differ from formal")
    test_targets = split.test_positive.groupby("UserID")["MovieID"].agg(set)
    if any(set(history) & test_targets.loc[int(user)]
           for user, history in zip(users, original_movie_ids, strict=True)):
        raise ValueError("Test item leaked into original pre-test histories")
    audit_lookup = frozen_audit.set_index(["user_id", "perturbation_seed"]).sort_index()
    matched_set = set(matched_users)
    sequences = {}
    structural_records = []
    for perturbation in PERTURBATIONS:
        local_rows = []
        broad_rows = []
        for user_id, history in zip(users, original, strict=True):
            seed = int(user_id) * 1009 + perturbation + 1_000_003
            local = history[local_budget_indices(len(history), TARGET_BUDGET, seed)]
            broad = history[broad_budget_indices(len(history), TARGET_BUDGET, seed + 100_000_019)]
            if len(local) != len(history) or len(broad) != len(history):
                raise AssertionError("Perturbation changed sequence length")
            if not np.array_equal(np.sort(local), np.sort(history)) or not np.array_equal(np.sort(broad), np.sort(history)):
                raise AssertionError("Perturbation changed an item multiset")
            frozen = audit_lookup.loc[(int(user_id), perturbation)]
            local_kendall = normalized_kendall_distance(history, local)
            broad_kendall = normalized_kendall_distance(history, broad)
            if not np.isclose(local_kendall, frozen["local_kendall_distance"], atol=1e-12):
                raise AssertionError("Local sequence differs from frozen calibration")
            if not np.isclose(broad_kendall, frozen["broad_kendall_distance"], atol=1e-12):
                raise AssertionError("Broad sequence differs from frozen calibration")
            if int(user_id) in matched_set:
                local_strength = perturbation_strength(history, local)
                broad_strength = perturbation_strength(history, broad)
                for family, strength in (("local", local_strength), ("broad", broad_strength)):
                    for measure in ("preserved_adjacency_rate", "normalized_displacement",
                                    "rank_correlation", "fixed_position_rate"):
                        if not np.isclose(strength[measure], frozen[f"{family}_{measure}"], atol=1e-12):
                            raise AssertionError(f"Frozen {family} {measure} changed")
                if max(abs(local_kendall - TARGET_BUDGET), abs(broad_kendall - TARGET_BUDGET),
                       abs(local_kendall - broad_kendall)) > 0.01 + 1e-12:
                    raise AssertionError("Matched user exceeded Kendall tolerance")
                structural_records.append({
                    "user_id": int(user_id), "perturbation_seed": perturbation,
                    "local_kendall_distance": local_kendall,
                    "broad_kendall_distance": broad_kendall,
                    "pair_kendall_mismatch": abs(local_kendall - broad_kendall),
                    **{f"local_{key}": value for key, value in local_strength.items()},
                    **{f"broad_{key}": value for key, value in broad_strength.items()},
                })
            local_rows.append(local)
            broad_rows.append(broad)
        sequences[perturbation] = {
            "matched_local": local_rows, "matched_broad": broad_rows,
        }
    structural = pd.DataFrame.from_records(structural_records)
    if len(structural) != len(matched_users) * len(PERTURBATIONS):
        raise AssertionError("Matched structural audit has missing user-seed cells")
    if structural["pair_kendall_mismatch"].max() > 0.01 + 1e-12:
        raise AssertionError("Structural matching changed")
    means = structural[[
        "local_kendall_distance", "broad_kendall_distance",
        "local_preserved_adjacency_rate", "broad_preserved_adjacency_rate",
    ]].mean()
    expected_means = (0.10075, 0.10093, 0.16220, 0.80454)
    if not np.allclose(means.to_numpy(), expected_means, atol=0.0001):
        raise AssertionError("Matched-subset structural means differ materially from calibration")
    return {
        "users": users, "matched_users": matched_users, "movie_ids": movie_ids,
        "original": original, "split": split, "test": test,
        "sequences": sequences, "structural": structural,
        "settings": settings, "settings_hashes": settings_hashes,
        "design": design, "preflight_seconds": time.perf_counter() - started,
    }


def run(output_dir: Path = OUTPUT) -> Path:
    """Train exactly the 90 new cells after immutable preflight checks."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    started = time.perf_counter()
    context = _preflight()
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "run_config.json", {
        "checkpoint_commit": _git_hash(),
        "git_status_before_run": _git_status(),
        "formal_output": str(FORMAL), "calibration_output": str(CALIBRATION),
        "formal_split_sha256": _file_hash(FORMAL / "split_assignments.csv"),
        "formal_per_user_metrics_sha256": _file_hash(FORMAL / "per_user_metrics.csv"),
        "calibration_config_sha256": _file_hash(CALIBRATION / "calibration_config.json"),
        "included_user_ids_sha256": _file_hash(CALIBRATION / "included_user_ids.csv"),
        "settings_sha256_by_initialization": context["settings_hashes"],
        "settings_by_initialization": context["settings"],
        "training_users": len(context["users"]),
        "matched_evaluation_users": len(context["matched_users"]),
        "catalog_items": len(context["movie_ids"]),
        "kendall_budget": TARGET_BUDGET,
        "initialization_seeds": list(INITIALIZATIONS),
        "perturbation_seeds": list(PERTURBATIONS),
        "new_conditions": list(NEW_CONDITIONS),
        "expected_new_cells": len(expected_new_cells()),
        "model_selection": "frozen Original-validation selections only; no new tuning",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    })
    context["structural"].to_csv(output_dir / "matched_structural_verification.csv", index=False)
    context["structural"].drop(columns=["user_id", "perturbation_seed"]).mean(numeric_only=True).to_frame(
        "mean"
    ).to_csv(output_dir / "matched_structural_summary.csv")
    pd.DataFrame(expected_new_cells(), columns=[
        "model_name", "model_initialization_seed", "sequence_shuffle_seed", "order_condition"
    ]).to_csv(output_dir / "expected_new_cells.csv", index=False)
    _write_json(output_dir / "pre_run_integrity.json", {
        "training_users": len(context["users"]),
        "matched_evaluation_users": len(context["matched_users"]),
        "catalog_items": len(context["movie_ids"]),
        "formal_split_identical": True, "targets_identical": True,
        "candidate_counts_identical": True, "no_test_leakage": True,
        "all_perturbations_match_frozen_structural_audit": True,
        "expected_new_cells": len(expected_new_cells()),
        "preflight_seconds": context["preflight_seconds"],
    })
    torch.set_num_threads(4)
    timings = []
    for model_name, initialization, perturbation, condition in expected_new_cells():
        cell_started = time.perf_counter()
        choice = context["settings"][initialization][model_name]
        sequences = context["sequences"][perturbation][condition]
        label = f"{model_name}_{condition}_{perturbation}_init{initialization}"
        print(f"Training {label} using frozen settings", flush=True)
        if model_name == "fpmc":
            model = FPMC(
                len(context["users"]), len(context["movie_ids"]), choice["factors"],
                .03, .002, initialization, 17000 + initialization,
            ).fit(sequences, choice["epochs"])
            scorer = fpmc_scorer(model, sequences, choice["transition_weight"])
        else:
            model, curve, _, _ = train_neural(
                model_name, sequences, len(context["movie_ids"]),
                choice["parameters"], choice["epochs"], label,
                initialization_seed=initialization,
            )
            pd.DataFrame(curve).to_csv(output_dir / f"{label}_curve.csv", index=False)
            max_length = choice["parameters"].get("max_sequence_length", 64)
            scorer = neural_scorer(model, sequences, model_name, max_length)
        frame = evaluate(context["test"], scorer)
        if not np.array_equal(frame["user_id"], context["users"]):
            raise AssertionError(f"User grid changed in {label}")
        if not np.isfinite(frame[["ndcg_10", "precision_10", "recall_10"]]).all().all():
            raise AssertionError(f"Nonfinite metrics in {label}")
        frame = frame.assign(
            model_name=model_name, order_condition=condition,
            model_initialization_seed=initialization,
            sequence_shuffle_seed=perturbation,
        )
        frame.to_csv(output_dir / f"{label}_metrics.csv", index=False)
        if model_name != "fpmc":
            torch.save(model.state_dict(), output_dir / f"{label}.pt")
        elapsed = time.perf_counter() - cell_started
        timings.append({
            "model_name": model_name, "model_initialization_seed": initialization,
            "sequence_shuffle_seed": perturbation, "order_condition": condition,
            "seconds": elapsed, "mean_ndcg_10_all_users": float(frame["ndcg_10"].mean()),
        })
        pd.DataFrame(timings).to_csv(output_dir / "cell_runtime.csv", index=False)
        print(f"Finished {label}: {elapsed:.1f}s", flush=True)
        del model, scorer, frame
    summarize(output_dir, context, started)
    return output_dir


def summarize(output_dir: Path, context: dict, started: float) -> None:
    """Validate and analyze the fixed, completed 90-cell grid."""
    columns = [
        "user_id", "model_name", "order_condition", "model_initialization_seed",
        "sequence_shuffle_seed", "ndcg_10", "number_of_candidates",
        "number_of_test_targets",
    ]
    new_frames = []
    reference_candidates = np.asarray([
        len(context["movie_ids"]) - len(excluded) for excluded in context["test"].excluded
    ])
    reference_targets = np.asarray([len(targets) for targets in context["test"].targets])
    for model_name, initialization, perturbation, condition in expected_new_cells():
        label = f"{model_name}_{condition}_{perturbation}_init{initialization}"
        path = output_dir / f"{label}_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete targeted cell: {path}")
        frame = pd.read_csv(path, usecols=columns)
        if len(frame) != len(context["users"]) or not np.array_equal(frame["user_id"], context["users"]):
            raise AssertionError(f"Missing or reordered users in {label}")
        if not frame["model_name"].eq(model_name).all() or not frame["order_condition"].eq(condition).all():
            raise AssertionError(f"Incorrect model or condition in {label}")
        if not frame["model_initialization_seed"].eq(initialization).all() or not frame["sequence_shuffle_seed"].eq(perturbation).all():
            raise AssertionError(f"Incorrect seed in {label}")
        if not np.isfinite(frame["ndcg_10"]).all():
            raise AssertionError(f"Nonfinite NDCG in {label}")
        if not np.array_equal(frame["number_of_candidates"], reference_candidates):
            raise AssertionError(f"Candidate count mismatch in {label}")
        if not np.array_equal(frame["number_of_test_targets"], reference_targets):
            raise AssertionError(f"Target count mismatch in {label}")
        new_frames.append(frame)
    new_metrics = pd.concat(new_frames, ignore_index=True)
    formal_metrics = pd.read_csv(FORMAL / "per_user_metrics.csv", usecols=columns)
    formal_metrics = formal_metrics.loc[
        formal_metrics["model_name"].isin(MODELS)
        & formal_metrics["order_condition"].isin(("original_order", "fully_shuffled_pretest"))
    ].copy()
    expected_formal_rows = len(context["users"]) * len(MODELS) * len(INITIALIZATIONS) * (1 + len(PERTURBATIONS))
    if len(formal_metrics) != expected_formal_rows:
        raise AssertionError("Frozen Original/Full metrics are incomplete")
    formal_metrics = formal_metrics.sort_values(
        ["model_name", "order_condition", "model_initialization_seed", "sequence_shuffle_seed", "user_id"],
        kind="stable",
    )
    if not np.isfinite(formal_metrics["ndcg_10"]).all():
        raise AssertionError("Frozen Original/Full metrics are nonfinite")
    combined = pd.concat((formal_metrics, new_metrics), ignore_index=True)
    keys = ["user_id", "model_name", "order_condition", "model_initialization_seed", "sequence_shuffle_seed"]
    if combined.duplicated(keys).any():
        raise AssertionError("Duplicate user/model/condition/seed metrics")
    if len(combined) != len(context["users"]) * (len(expected_new_cells()) + len(MODELS) * len(INITIALIZATIONS) * 4):
        raise AssertionError("Four-condition result grid is incomplete")
    combined.to_csv(output_dir / "per_user_metrics.csv", index=False)
    matched = combined.loc[combined["user_id"].isin(context["matched_users"])].copy()
    matched.to_csv(output_dir / "matched_subset_per_user_metrics.csv", index=False)
    index = pd.MultiIndex.from_product(
        [context["matched_users"], INITIALIZATIONS], names=["user_id", "model_initialization_seed"]
    )
    effect_rows = []
    interval_rows = []
    initialization_rows = []
    perturbation_rows = []
    for model_index, model_name in enumerate(MODELS):
        model = matched.loc[matched["model_name"].eq(model_name)]
        original_rows = model.loc[model["order_condition"].eq("original_order")]
        if len(original_rows) != len(index) or original_rows["sequence_shuffle_seed"].notna().any():
            raise AssertionError(f"Original seed grid is incomplete for {model_name}")
        original = original_rows.set_index(["user_id", "model_initialization_seed"])["ndcg_10"].reindex(index).to_numpy()
        original = original.reshape(len(context["matched_users"]), len(INITIALIZATIONS))
        if not np.isfinite(original).all():
            raise AssertionError(f"Original seed pairing failed for {model_name}")
        condition_arrays = {}
        for condition in (*NEW_CONDITIONS, "fully_shuffled_pretest"):
            rows = model.loc[model["order_condition"].eq(condition)]
            if len(rows) != len(index) * len(PERTURBATIONS):
                raise AssertionError(f"{condition} seed grid is incomplete for {model_name}")
            pivot = rows.pivot(index=["user_id", "model_initialization_seed"],
                               columns="sequence_shuffle_seed", values="ndcg_10")
            values = pivot.reindex(index=index, columns=PERTURBATIONS).to_numpy()
            if not np.isfinite(values).all():
                raise AssertionError(f"{condition} pairing failed for {model_name}")
            condition_arrays[condition] = values.reshape(
                len(context["matched_users"]), len(INITIALIZATIONS), len(PERTURBATIONS)
            )
        original_cube = np.broadcast_to(original[:, :, None], condition_arrays["matched_local"].shape)
        local = condition_arrays["matched_local"]
        broad = condition_arrays["matched_broad"]
        full = condition_arrays["fully_shuffled_pretest"]
        effects = np.stack((original_cube - local, original_cube - broad,
                            original_cube - full, broad - local))
        effect_names = ("delta_local", "delta_broad", "delta_full", "broad_minus_local")
        lower, upper = hierarchical_paired_intervals(
            effects, samples=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED + model_index
        )
        original_mean = float(original.mean())
        condition_means = {
            "original_ndcg_10": original_mean,
            "local_ndcg_10": float(local.mean()),
            "broad_ndcg_10": float(broad.mean()),
            "full_shuffle_ndcg_10": float(full.mean()),
        }
        effect_means = effects.mean(axis=(1, 2, 3))
        effect_rows.append({
            "population": "matched_subset_primary", "model_name": model_name,
            "user_count": len(context["matched_users"]), **condition_means,
            **dict(zip(effect_names, effect_means, strict=True)),
            "relative_local_drop_percent": 100 * effect_means[0] / original_mean,
            "relative_broad_drop_percent": 100 * effect_means[1] / original_mean,
            "relative_full_drop_percent": 100 * effect_means[2] / original_mean,
        })
        for effect_index, name in enumerate(effect_names):
            interval_rows.append({
                "population": "matched_subset_primary", "model_name": model_name,
                "effect": name, "mean_effect": float(effect_means[effect_index]),
                "ci_95_lower": float(lower[effect_index]),
                "ci_95_upper": float(upper[effect_index]),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_unit": "paired users with resampled initialization and perturbation seed clusters",
            })
        for initialization_index, initialization in enumerate(INITIALIZATIONS):
            row = {
                "model_name": model_name, "model_initialization_seed": initialization,
                "original_ndcg_10": float(original[:, initialization_index].mean()),
                "local_ndcg_10": float(local[:, initialization_index, :].mean()),
                "broad_ndcg_10": float(broad[:, initialization_index, :].mean()),
                "full_shuffle_ndcg_10": float(full[:, initialization_index, :].mean()),
            }
            row.update(zip(effect_names, effects[:, :, initialization_index, :].mean(axis=(1, 2)), strict=True))
            initialization_rows.append(row)
        for perturbation_index, perturbation in enumerate(PERTURBATIONS):
            row = {
                "model_name": model_name, "perturbation_seed": perturbation,
                "original_ndcg_10": original_mean,
                "local_ndcg_10": float(local[:, :, perturbation_index].mean()),
                "broad_ndcg_10": float(broad[:, :, perturbation_index].mean()),
                "full_shuffle_ndcg_10": float(full[:, :, perturbation_index].mean()),
            }
            row.update(zip(effect_names, effects[:, :, :, perturbation_index].mean(axis=(1, 2)), strict=True))
            perturbation_rows.append(row)
    pd.DataFrame(effect_rows).to_csv(output_dir / "aggregate_effects.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(output_dir / "bootstrap_intervals.csv", index=False)
    pd.DataFrame(initialization_rows).to_csv(output_dir / "per_initialization_effects.csv", index=False)
    pd.DataFrame(perturbation_rows).to_csv(output_dir / "per_perturbation_seed_effects.csv", index=False)
    population = describe_matched_and_excluded(context, formal_metrics)
    population.to_csv(output_dir / "matched_vs_excluded.csv", index=False)
    runtime = pd.read_csv(output_dir / "cell_runtime.csv")
    if len(runtime) != len(expected_new_cells()):
        raise AssertionError("Cell runtime log is incomplete")
    _write_json(output_dir / "integrity_validation.json", {
        "valid": True, "new_cells_completed": len(runtime),
        "expected_new_cells": len(expected_new_cells()),
        "new_metric_rows": len(new_metrics),
        "reused_formal_metric_rows": len(formal_metrics),
        "matched_metric_rows": len(matched),
        "matched_users": len(context["matched_users"]),
        "training_users": len(context["users"]),
        "duplicate_metric_cells": 0, "nonfinite_ndcg_rows": 0,
        "candidate_mismatches": 0, "target_mismatches": 0,
        "item_multiset_mismatches": 0, "test_leakage_users": 0,
    })
    process_memory = psutil.Process().memory_info()
    _write_json(output_dir / "runtime_summary.json", {
        "total_wall_seconds": time.perf_counter() - started,
        "new_cell_seconds_by_model": runtime.groupby("model_name")["seconds"].sum().to_dict(),
        "new_cell_seconds_total": float(runtime["seconds"].sum()),
        "peak_process_rss_bytes": int(getattr(process_memory, "peak_wset", process_memory.rss)),
        "output_size_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
        "failed_cells": [], "retries": 0,
        "warnings": ["PyTorch may emit the existing nested-tensor norm_first warning"],
    })
    _write_json(output_dir / "completion.json", {
        "complete": True, "new_cells": len(runtime),
        "training_users": len(context["users"]),
        "primary_evaluation_users": len(context["matched_users"]),
        "formal_original_and_full_reused": True,
        "models_retuned": False,
        "checkpoint_commit": _git_hash(),
    })


def hierarchical_paired_intervals(
    effects: np.ndarray, *, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap users and global seed clusters with pairing across effects."""
    if effects.ndim != 4 or effects.shape[2:] != (len(INITIALIZATIONS), len(PERTURBATIONS)):
        raise ValueError("Expected effect, user, initialization, perturbation tensor")
    effect_count, users, _, _ = effects.shape
    flattened = effects.reshape(effect_count * users, -1)
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, effect_count), dtype=np.float64)
    for replicate in range(samples):
        initialization_indices = rng.integers(0, len(INITIALIZATIONS), len(INITIALIZATIONS))
        perturbation_indices = rng.integers(0, len(PERTURBATIONS), len(PERTURBATIONS))
        initialization_weights = np.bincount(initialization_indices, minlength=len(INITIALIZATIONS)) / len(INITIALIZATIONS)
        perturbation_weights = np.bincount(perturbation_indices, minlength=len(PERTURBATIONS)) / len(PERTURBATIONS)
        weights = np.outer(initialization_weights, perturbation_weights).ravel()
        seed_average = (flattened @ weights).reshape(effect_count, users)
        user_indices = rng.integers(0, users, users)
        estimates[replicate] = seed_average[:, user_indices].mean(axis=1)
    return np.quantile(estimates, 0.025, axis=0), np.quantile(estimates, 0.975, axis=0)


def describe_matched_and_excluded(context: dict, formal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Describe the predeclared subset without changing primary inference."""
    matched = set(context["matched_users"])
    lengths = pd.DataFrame({
        "user_id": context["users"],
        "sequence_length": [len(sequence) for sequence in context["original"]],
    })
    formal = formal_metrics.copy()
    formal["population"] = np.where(
        formal["user_id"].isin(matched), "matched_subset_primary", "excluded_structural_users"
    )
    records = []
    for model_name in MODELS:
        model = formal.loc[formal["model_name"].eq(model_name)]
        original = model.loc[model["order_condition"].eq("original_order")]
        shuffled = model.loc[model["order_condition"].eq("fully_shuffled_pretest")]
        original_user = original.groupby(["population", "user_id"])["ndcg_10"].mean()
        shuffled_user = shuffled.groupby(["population", "user_id"])["ndcg_10"].mean()
        paired = pd.concat((original_user.rename("original"), shuffled_user.rename("full")), axis=1)
        paired["delta_full"] = paired["original"] - paired["full"]
        for population, frame in paired.groupby(level=0):
            users = frame.index.get_level_values("user_id")
            lengths_subset = lengths.set_index("user_id").loc[users, "sequence_length"]
            records.append({
                "model_name": model_name, "population": population,
                "user_count": len(frame),
                "median_sequence_length": float(lengths_subset.median()),
                "mean_sequence_length": float(lengths_subset.mean()),
                "original_ndcg_10": float(frame["original"].mean()),
                "full_shuffle_ndcg_10": float(frame["full"].mean()),
                "delta_full": float(frame["delta_full"].mean()),
            })
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.phase == "preflight":
        context = _preflight()
        print(json.dumps({
            "training_users": len(context["users"]),
            "matched_users": len(context["matched_users"]),
            "catalog_items": len(context["movie_ids"]),
            "structural_cells": len(context["structural"]),
            "expected_new_cells": len(expected_new_cells()),
            "preflight_seconds": context["preflight_seconds"],
        }, indent=2))
    else:
        run(args.output_dir)


if __name__ == "__main__":
    main()
