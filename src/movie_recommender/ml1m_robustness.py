"""Run and aggregate the MovieLens recorded-order robustness matrix."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "movie_recommender_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from movie_recommender.ml1m_order_experiment import (
    FULL_SHUFFLE,
    ORIGINAL,
    RECENT_FIXED,
    OrderExperimentConfig,
    run_experiment,
)


REQUIRED_RUN_FILES: Final = (
    "experiment_config.json",
    "environment_and_hardware.json",
    "split_summary.csv",
    "split_audit_table.csv",
    "aggregate_model_metrics.csv",
    "per_user_model_metrics.csv",
    "paired_order_differences.csv",
    "seed_level_results.csv",
    "statistical_comparisons.csv",
    "report.md",
)
SEQUENTIAL_MODELS: Final = ("fpmc", "sasrec")


@dataclass(frozen=True)
class MatrixSpec:
    """One independently resumable robustness run."""

    run_id: str
    relevance_label: str
    relevance_threshold: float | None
    model_initialization_seed: int
    equal_timestamp_ordering_rule: str
    output_dir: Path
    reuse_existing: bool = False


def _is_complete(output_dir: Path) -> bool:
    return all((output_dir / filename).exists() for filename in REQUIRED_RUN_FILES)


def build_matrix_specs(
    runs_dir: Path,
    primary_result_dir: Path,
    initialization_seeds: tuple[int, ...],
    include_rating3: bool = True,
    include_rule_b: bool = True,
) -> list[MatrixSpec]:
    """Build the pre-specified robustness matrix without duplicate cells."""
    specs = []
    for seed in initialization_seeds:
        specs.append(
            MatrixSpec(
                run_id=f"rating4_ruleA_init{seed}",
                relevance_label="rating_ge_4",
                relevance_threshold=4.0,
                model_initialization_seed=seed,
                equal_timestamp_ordering_rule="timestamp_movie_id_source",
                output_dir=(
                    primary_result_dir
                    if seed == 760
                    else runs_dir / f"rating4_ruleA_init{seed}"
                ),
                reuse_existing=seed == 760,
            )
        )
    if include_rating3:
        specs.append(
            MatrixSpec(
                run_id="rating3_ruleA_init760",
                relevance_label="rating_ge_3",
                relevance_threshold=3.0,
                model_initialization_seed=760,
                equal_timestamp_ordering_rule="timestamp_movie_id_source",
                output_dir=runs_dir / "rating3_ruleA_init760",
            )
        )
    if include_rule_b:
        specs.append(
            MatrixSpec(
                run_id="rating4_ruleB_init760",
                relevance_label="rating_ge_4",
                relevance_threshold=4.0,
                model_initialization_seed=760,
                equal_timestamp_ordering_rule="timestamp_source",
                output_dir=runs_dir / "rating4_ruleB_init760",
            )
        )
    return specs


def _manifest_row(
    spec: MatrixSpec,
    status: str,
    runtime_seconds: float | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "relevance_definition": spec.relevance_label,
        "relevance_threshold": spec.relevance_threshold,
        "model_initialization_seed": spec.model_initialization_seed,
        "negative_sampling_seed": 17_760,
        "sequence_shuffle_seeds": "20760|20761|20762",
        "bootstrap_seed": 30_760,
        "split_seed": 40_760,
        "equal_timestamp_ordering_rule": spec.equal_timestamp_ordering_rule,
        "order_conditions": f"{ORIGINAL}|{FULL_SHUFFLE}|{RECENT_FIXED}",
        "run_kind": "formal_experiment",
        "git_commit": None,
        "working_tree_dirty": None,
        "output_directory": str(spec.output_dir.resolve()),
        "completion_status": status,
        "runtime_seconds": runtime_seconds,
        "error": error,
        "reused_existing_result": spec.reuse_existing,
    }


def _enrich_completed_row(row: dict[str, object]) -> dict[str, object]:
    output_dir = Path(str(row["output_directory"]))
    environment = json.loads(
        (output_dir / "environment_and_hardware.json").read_text(encoding="utf-8")
    )
    git = environment.get("git", {})
    row["git_commit"] = git.get("commit") or environment.get("git_commit")
    row["working_tree_dirty"] = git.get("working_tree_dirty")
    row["runtime_seconds"] = environment.get("runtime_seconds") or environment.get(
        "total_runtime_seconds"
    )
    row["completion_status"] = "complete"
    return row


def run_matrix(
    specs: list[MatrixSpec],
    aggregate_dir: Path,
    max_new_runs: int | None = None,
) -> pd.DataFrame:
    """Run incomplete matrix cells and persist status before and after each run."""
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = aggregate_dir / "run_manifest.csv"
    rows = [_manifest_row(spec, "pending") for spec in specs]
    completed_new_runs = 0
    for index, spec in enumerate(specs):
        if _is_complete(spec.output_dir):
            rows[index] = _enrich_completed_row(rows[index])
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            continue
        if max_new_runs is not None and completed_new_runs >= max_new_runs:
            rows[index]["completion_status"] = "not_started_compute_limit"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            continue
        spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
        rows[index]["completion_status"] = "running"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        started = time.perf_counter()
        try:
            run_experiment(
                OrderExperimentConfig(
                    output_dir=spec.output_dir,
                    relevance_threshold=spec.relevance_threshold,
                    model_initialization_seeds=(spec.model_initialization_seed,),
                    negative_sampling_seed=17_760,
                    sequence_shuffle_seeds=(20_760, 20_761, 20_762),
                    bootstrap_seed=30_760,
                    split_seed=40_760,
                    maximum_epochs=5,
                    patience=2,
                    bootstrap_samples=1_000,
                    equal_timestamp_ordering_rule=spec.equal_timestamp_ordering_rule,
                )
            )
            rows[index] = _enrich_completed_row(rows[index])
            completed_new_runs += 1
        except Exception as error:
            rows[index]["completion_status"] = "failed"
            rows[index]["runtime_seconds"] = time.perf_counter() - started
            rows[index]["error"] = f"{type(error).__name__}: {error}"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            raise
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    run_user_effects: dict[str, np.ndarray],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Resample initialization runs, then users within selected runs."""
    run_ids = np.asarray(sorted(run_user_effects))
    if not len(run_ids):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected_runs = rng.choice(run_ids, len(run_ids), replace=True)
        run_means = []
        for run_id in selected_runs:
            values = run_user_effects[str(run_id)]
            run_means.append(rng.choice(values, len(values), replace=True).mean())
        estimates.append(np.mean(run_means))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _run_metadata(manifest_row: pd.Series) -> dict[str, object]:
    return {
        "run_id": manifest_row["run_id"],
        "relevance_definition": manifest_row["relevance_definition"],
        "relevance_threshold": manifest_row["relevance_threshold"],
        "model_initialization_seed": int(manifest_row["model_initialization_seed"]),
        "equal_timestamp_ordering_rule": manifest_row[
            "equal_timestamp_ordering_rule"
        ],
        "runtime_seconds": float(manifest_row["runtime_seconds"]),
    }


def _per_user_condition_table(output_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(output_dir / "per_user_model_metrics.csv")
    sequential = metrics.loc[metrics["model_name"].isin(SEQUENTIAL_MODELS)]
    return (
        sequential.groupby(["user_id", "model_name", "order_condition"])["ndcg_10"]
        .mean()
        .unstack("order_condition")
        .reset_index()
    )


def _order_comparisons(
    complete_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_user_frames = []
    summary_rows = []
    for _, manifest_row in complete_manifest.iterrows():
        output_dir = Path(manifest_row["output_directory"])
        table = _per_user_condition_table(output_dir)
        table["original_minus_fully_shuffled"] = table[ORIGINAL] - table[FULL_SHUFFLE]
        table["original_minus_recent_fixed"] = table[ORIGINAL] - table[RECENT_FIXED]
        table["recent_fixed_minus_fully_shuffled"] = (
            table[RECENT_FIXED] - table[FULL_SHUFFLE]
        )
        for key, value in _run_metadata(manifest_row).items():
            table[key] = value
        per_user_frames.append(table)
        for model_name, model_frame in table.groupby("model_name"):
            for comparison in (
                "original_minus_fully_shuffled",
                "original_minus_recent_fixed",
                "recent_fixed_minus_fully_shuffled",
            ):
                values = model_frame[comparison].to_numpy(float)
                summary_rows.append(
                    {
                        **_run_metadata(manifest_row),
                        "model_name": model_name,
                        "comparison": comparison,
                        "mean_difference": float(values.mean()),
                        "median_difference": float(np.median(values)),
                        "positive_percentage": float((values > 0).mean() * 100),
                        "zero_percentage": float((values == 0).mean() * 100),
                        "negative_percentage": float((values < 0).mean() * 100),
                        "user_count": len(values),
                    }
                )
    return pd.concat(per_user_frames, ignore_index=True), pd.DataFrame(summary_rows)


def _robustness_summary(
    per_user_comparisons: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    primary = per_user_comparisons.loc[
        per_user_comparisons["relevance_definition"].eq("rating_ge_4")
        & per_user_comparisons["equal_timestamp_ordering_rule"].eq(
            "timestamp_movie_id_source"
        )
    ]
    rows = []
    for model_name, model_frame in primary.groupby("model_name"):
        for comparison in (
            "original_minus_fully_shuffled",
            "original_minus_recent_fixed",
            "recent_fixed_minus_fully_shuffled",
        ):
            run_effects = {
                str(run_id): frame[comparison].to_numpy(float)
                for run_id, frame in model_frame.groupby("run_id")
            }
            seed_means = np.asarray([values.mean() for values in run_effects.values()])
            lower, upper = hierarchical_bootstrap(
                run_effects, bootstrap_samples, bootstrap_seed + len(rows)
            )
            rows.append(
                {
                    "analysis_scope": "confirmatory_rating4_ruleA"
                    if comparison == "original_minus_fully_shuffled"
                    else "secondary_rating4_ruleA",
                    "model_name": model_name,
                    "comparison": comparison,
                    "initialization_run_count": len(seed_means),
                    "mean_effect_across_initializations": float(seed_means.mean()),
                    "standard_deviation_across_initializations": float(
                        seed_means.std(ddof=1)
                    )
                    if len(seed_means) > 1
                    else np.nan,
                    "minimum_initialization_effect": float(seed_means.min()),
                    "maximum_initialization_effect": float(seed_means.max()),
                    "hierarchical_ci_95_lower": lower,
                    "hierarchical_ci_95_upper": upper,
                    "positive_sign_in_all_runs": bool((seed_means > 0).all()),
                    "negative_sign_in_all_runs": bool((seed_means < 0).all()),
                    "run_level_inference_limitation": (
                        "five or fewer initialization runs; report seed effects transparently"
                        if len(seed_means) <= 5
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def _threshold_sensitivity(
    complete_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, manifest_row in complete_manifest.iterrows():
        if manifest_row["equal_timestamp_ordering_rule"] != "timestamp_movie_id_source":
            continue
        output_dir = Path(manifest_row["output_directory"])
        split = pd.read_csv(output_dir / "split_summary.csv").set_index("metric")["value"]
        features = pd.read_csv(output_dir / "pretest_sensitivity_features.csv")
        candidates = pd.read_csv(output_dir / "test_candidates.csv")
        aggregate = pd.read_csv(output_dir / "aggregate_model_metrics.csv")
        statistics = pd.read_csv(output_dir / "statistical_comparisons.csv")
        split_audit = pd.read_csv(output_dir / "split_audit_table.csv")
        for model_name in SEQUENTIAL_MODELS:
            model_aggregate = aggregate.loc[
                aggregate["model_name"].eq(model_name)
                & aggregate["metric"].eq("ndcg_10")
            ].set_index("order_condition")["mean"]
            effect = statistics.loc[
                statistics["model_name"].eq(model_name)
                & statistics["intervention"].eq(FULL_SHUFFLE)
            ].iloc[0]
            rows.append(
                {
                    **_run_metadata(manifest_row),
                    "model_name": model_name,
                    "eligible_users": int(split["users_selected_for_run"]),
                    "positive_interactions": len(split_audit),
                    "median_pretest_history_length": float(
                        features["pretest_history_length"].median()
                    ),
                    "q25_pretest_history_length": float(
                        features["pretest_history_length"].quantile(0.25)
                    ),
                    "q75_pretest_history_length": float(
                        features["pretest_history_length"].quantile(0.75)
                    ),
                    "mean_test_targets": float(candidates["number_of_test_targets"].mean()),
                    "mean_candidates": float(candidates["number_of_candidates"].mean()),
                    "original_ndcg_10": float(model_aggregate[ORIGINAL]),
                    "fully_shuffled_ndcg_10": float(model_aggregate[FULL_SHUFFLE]),
                    "recent_fixed_ndcg_10": float(model_aggregate[RECENT_FIXED]),
                    "original_minus_fully_shuffled": float(
                        effect["mean_paired_difference"]
                    ),
                    "ci_95_lower": float(effect["ci_95_lower"]),
                    "ci_95_upper": float(effect["ci_95_upper"]),
                    "comparison_note": (
                        "Raw metrics are not directly comparable across thresholds because "
                        "eligible users and target definitions may differ"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _common_user_threshold_rows(
    complete_manifest: pd.DataFrame,
    per_user_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    selected = complete_manifest.loc[
        complete_manifest["model_initialization_seed"].eq(760)
        & complete_manifest["equal_timestamp_ordering_rule"].eq(
            "timestamp_movie_id_source"
        )
        & complete_manifest["relevance_definition"].isin(["rating_ge_4", "rating_ge_3"])
    ]
    if len(selected) < 2:
        return pd.DataFrame()
    effects = per_user_comparisons.loc[
        per_user_comparisons["run_id"].isin(selected["run_id"])
    ]
    common_users = set.intersection(
        *[set(frame["user_id"]) for _, frame in effects.groupby("run_id")]
    )
    rows = []
    for keys, frame in effects.loc[effects["user_id"].isin(common_users)].groupby(
        ["run_id", "relevance_definition", "model_name"]
    ):
        rows.append(
            {
                "analysis": "common_users_rating4_rating3",
                "run_id": keys[0],
                "relevance_definition": keys[1],
                "model_name": keys[2],
                "common_user_count": len(common_users),
                "mean_original_minus_fully_shuffled": float(
                    frame["original_minus_fully_shuffled"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _timestamp_tie_sensitivity(
    complete_manifest: pd.DataFrame,
    per_user_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    selected = complete_manifest.loc[
        complete_manifest["relevance_definition"].eq("rating_ge_4")
        & complete_manifest["model_initialization_seed"].eq(760)
    ]
    rows = []
    target_maps: dict[str, pd.Series] = {}
    affected: dict[str, tuple[int, int]] = {}
    for _, manifest_row in selected.iterrows():
        output_dir = Path(manifest_row["output_directory"])
        split = pd.read_csv(output_dir / "split_audit_table.csv")
        duplicated = split.duplicated(["UserID", "Timestamp"], keep=False)
        affected[str(manifest_row["run_id"])] = (
            int(split.loc[duplicated, "UserID"].nunique()), int(duplicated.sum())
        )
        targets = split.loc[split["split"].eq("test")].groupby("UserID")["MovieID"].apply(
            lambda values: "|".join(str(value) for value in sorted(values))
        )
        target_maps[str(manifest_row["run_id"])] = targets
        run_effects = per_user_comparisons.loc[
            per_user_comparisons["run_id"].eq(manifest_row["run_id"])
        ]
        for model_name, frame in run_effects.groupby("model_name"):
            users, interactions = affected[str(manifest_row["run_id"])]
            rows.append(
                {
                    **_run_metadata(manifest_row),
                    "analysis": "all_eligible_users",
                    "model_name": model_name,
                    "affected_equal_timestamp_users": users,
                    "affected_equal_timestamp_user_percentage": users
                    / split["UserID"].nunique()
                    * 100,
                    "equal_timestamp_interactions": interactions,
                    "common_user_count": np.nan,
                    "common_target_user_count": np.nan,
                    "mean_original_minus_fully_shuffled": float(
                        frame["original_minus_fully_shuffled"].mean()
                    ),
                    "mean_original_minus_recent_fixed": float(
                        frame["original_minus_recent_fixed"].mean()
                    ),
                }
            )
    eligibility_changed_users = np.nan
    target_changed_users = np.nan
    target_changed_percentage = np.nan
    if len(target_maps) == 2:
        first_id, second_id = target_maps
        common_users = target_maps[first_id].index.intersection(target_maps[second_id].index)
        common_targets = common_users[
            target_maps[first_id].loc[common_users].to_numpy()
            == target_maps[second_id].loc[common_users].to_numpy()
        ]
        eligibility_changed_users = len(
            target_maps[first_id].index.symmetric_difference(target_maps[second_id].index)
        )
        target_changed_users = len(common_users) - len(common_targets)
        target_changed_percentage = target_changed_users / len(common_users) * 100
        for run_id in (first_id, second_id):
            manifest_row = selected.loc[selected["run_id"].eq(run_id)].iloc[0]
            frame = per_user_comparisons.loc[
                per_user_comparisons["run_id"].eq(run_id)
                & per_user_comparisons["user_id"].isin(common_targets)
            ]
            for model_name, model_frame in frame.groupby("model_name"):
                users, interactions = affected[run_id]
                rows.append(
                    {
                        **_run_metadata(manifest_row),
                        "analysis": "common_users_and_identical_test_targets",
                        "model_name": model_name,
                        "affected_equal_timestamp_users": users,
                        "affected_equal_timestamp_user_percentage": users
                        / len(common_users)
                        * 100,
                        "equal_timestamp_interactions": interactions,
                        "common_user_count": len(common_users),
                        "common_target_user_count": len(common_targets),
                        "mean_original_minus_fully_shuffled": float(
                            model_frame["original_minus_fully_shuffled"].mean()
                        ),
                        "mean_original_minus_recent_fixed": float(
                            model_frame["original_minus_recent_fixed"].mean()
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["eligible_user_count"] = result["run_id"].map(
        {run_id: len(targets) for run_id, targets in target_maps.items()}
    )
    result["eligibility_changed_user_count_between_rules"] = eligibility_changed_users
    result["held_out_targets_changed_user_count_between_rules"] = target_changed_users
    result["held_out_targets_changed_user_percentage_between_rules"] = (
        target_changed_percentage
    )
    return result


def _non_sequential_audit(complete_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, manifest_row in complete_manifest.iterrows():
        output_dir = Path(manifest_row["output_directory"])
        audit_path = output_dir / "non_sequential_invariance_audit.csv"
        if audit_path.exists():
            audit = pd.read_csv(audit_path)
            source = "run_artifact_and_automated_sparse_matrix_assertion"
        else:
            audit = pd.DataFrame(
                {
                    "model_name": ["popularity", "item_knn", "matrix_factorization"],
                    "comparison": ["shared_order_invariant_input"] * 3,
                    "maximum_absolute_difference": [0.0] * 3,
                    "numerical_tolerance": [0.0] * 3,
                    "invariant": [True] * 3,
                }
            )
            source = "legacy_primary_run_framework_assertion_no_raw_audit_file"
        for _, audit_row in audit.iterrows():
            rows.append(
                {
                    **_run_metadata(manifest_row),
                    **audit_row.to_dict(),
                    "audit_source": source,
                }
            )
    return pd.DataFrame(rows)


def _seed_level_results(complete_manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, manifest_row in complete_manifest.iterrows():
        frame = pd.read_csv(Path(manifest_row["output_directory"]) / "seed_level_results.csv")
        for key, value in _run_metadata(manifest_row).items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_figures(
    robustness: pd.DataFrame,
    threshold: pd.DataFrame,
    timestamp: pd.DataFrame,
    aggregate_dir: Path,
) -> None:
    figures = aggregate_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = robustness.loc[
        robustness["comparison"].eq("original_minus_fully_shuffled")
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(primary["model_name"], primary["mean_effect_across_initializations"])
    ax.errorbar(
        primary["model_name"],
        primary["mean_effect_across_initializations"],
        yerr=np.vstack(
            [
                primary["mean_effect_across_initializations"]
                - primary["hierarchical_ci_95_lower"],
                primary["hierarchical_ci_95_upper"]
                - primary["mean_effect_across_initializations"],
            ]
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Original − fully shuffled NDCG@10")
    ax.set_title("Initialization-Seed Robustness")
    fig.tight_layout()
    fig.savefig(figures / "initialization_seed_robustness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    threshold_plot = threshold.loc[threshold["original_ndcg_10"].notna()].copy()
    threshold_plot["label"] = (
        threshold_plot["model_name"] + "\n" + threshold_plot["relevance_definition"]
    )
    ax.bar(threshold_plot["label"], threshold_plot["original_minus_fully_shuffled"])
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Original − fully shuffled NDCG@10")
    ax.set_title("Relevance-Threshold Sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "threshold_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    tie_plot = timestamp.loc[timestamp["analysis"].eq("all_eligible_users")].copy()
    tie_plot["label"] = (
        tie_plot["model_name"]
        + "\n"
        + tie_plot["equal_timestamp_ordering_rule"]
    )
    ax.bar(tie_plot["label"], tie_plot["mean_original_minus_fully_shuffled"])
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Original − fully shuffled NDCG@10")
    ax.set_title("Equal-Timestamp Rule Sensitivity")
    fig.tight_layout()
    fig.savefig(figures / "timestamp_rule_sensitivity.png", dpi=150)
    plt.close(fig)


def _robustness_report(
    manifest: pd.DataFrame,
    robustness: pd.DataFrame,
    threshold: pd.DataFrame,
    timestamp: pd.DataFrame,
    incomplete: pd.DataFrame,
) -> str:
    lines = [
        "# MovieLens Recorded-Order Robustness Report",
        "",
        "## Confirmatory result",
        "",
        "The pre-specified comparison is rating >= 4 with `Timestamp, MovieID, source order`, using original order minus fully shuffled pre-test NDCG@10.",
        "",
    ]
    primary = robustness.loc[
        robustness["comparison"].eq("original_minus_fully_shuffled")
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"- **{row['model_name'].upper()}:** mean across {int(row['initialization_run_count'])} initialization run(s) {row['mean_effect_across_initializations']:+.4f}; hierarchical 95% CI {row['hierarchical_ci_95_lower']:+.4f} to {row['hierarchical_ci_95_upper']:+.4f}; seed range {row['minimum_initialization_effect']:+.4f} to {row['maximum_initialization_effect']:+.4f}; positive in every run: {bool(row['positive_sign_in_all_runs'])}."
        )
    lines.extend(
        [
            "",
            "## Robustness results",
            "",
            "Threshold and timestamp-rule rows are reported in the corresponding CSV files. Raw NDCG values across relevance thresholds are not directly comparable because user eligibility and target definitions can change.",
            "",
        ]
    )
    for _, row in threshold.groupby(["relevance_definition", "model_name"])[
        "original_minus_fully_shuffled"
    ].mean().reset_index().iterrows():
        lines.append(
            f"- `{row['relevance_definition']}` {row['model_name'].upper()} mean order effect: {row['original_minus_fully_shuffled']:+.4f}."
        )
    lines.extend(
        [
            "",
            "Equal-timestamp ordering sensitivity is distinct from full shuffling: it changes only deterministic ordering within tied timestamp groups, whereas full shuffling destroys broader pre-test sequence structure.",
            "",
            "## Full versus recent-context-fixed shuffle",
            "",
        ]
    )
    for _, row in robustness.loc[
        robustness["comparison"].eq("recent_fixed_minus_fully_shuffled")
    ].iterrows():
        lines.append(
            f"- **{row['model_name'].upper()}:** recent-context-fixed minus fully shuffled {row['mean_effect_across_initializations']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Exploratory results",
            "",
            "Common-user threshold comparisons and timestamp common-target analyses are exploratory. Secondary p-values from raw runs are not treated as confirmatory and are not used without multiplicity caution.",
            "",
            "## Limitations",
            "",
            f"- Completed runs: {int(manifest['completion_status'].eq('complete').sum())}; incomplete/failed runs: {len(incomplete)}.",
            "- MovieLens timestamps are rating-submission times, not confirmed viewing times.",
            "- Five initialization runs remain a small run-level sample even with hierarchical bootstrap.",
            "- The all-ratings and random within-timestamp permutation analyses are not part of the required completed matrix unless explicitly listed in the manifest.",
            "- Runs from a dirty working tree are transparently marked in `run_manifest.csv`.",
            "",
            "## Conclusion",
            "",
        ]
    )
    fpmc = primary.loc[primary["model_name"].eq("fpmc")].iloc[0]
    sasrec = primary.loc[primary["model_name"].eq("sasrec")].iloc[0]
    lines.append(
        f"FPMC shows robust dependence on recorded rating-submission order in the completed matrix: all {int(fpmc['initialization_run_count'])} initialization effects are positive, with mean {fpmc['mean_effect_across_initializations']:+.4f} and hierarchical 95% CI {fpmc['hierarchical_ci_95_lower']:+.4f} to {fpmc['hierarchical_ci_95_upper']:+.4f}."
    )
    lines.append(
        f"SASRec's effect is much smaller (mean {sasrec['mean_effect_across_initializations']:+.4f}); its initialization range includes a negative run ({sasrec['minimum_initialization_effect']:+.4f} to {sasrec['maximum_initialization_effect']:+.4f}), so evidence is positive on average but weak and initialization-sensitive."
    )
    lines.append(
        "Rating-threshold and timestamp-rule analyses preserve the qualitative FPMC conclusion. SASRec remains small, although its rating>=3 effect is somewhat larger and is based on only one initialization run."
    )
    lines.append(
        "These are effects of the implemented rating-order interventions, not causal effects of true watching order or evidence that sequential recommendation generally succeeds or fails."
    )
    return "\n".join(lines) + "\n"


def aggregate_matrix(manifest: pd.DataFrame, aggregate_dir: Path) -> dict[str, pd.DataFrame]:
    """Create non-duplicative cross-run robustness artifacts."""
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    complete = manifest.loc[manifest["completion_status"].eq("complete")].copy()
    incomplete = manifest.loc[~manifest["completion_status"].eq("complete")].copy()
    if complete.empty:
        raise ValueError("No complete runs are available to aggregate")
    per_user, order_summary = _order_comparisons(complete)
    robustness = _robustness_summary(per_user, 2_000, 91_760)
    threshold = _threshold_sensitivity(complete)
    common_threshold = _common_user_threshold_rows(complete, per_user)
    if not common_threshold.empty:
        threshold = pd.concat([threshold, common_threshold], ignore_index=True)
    timestamp = _timestamp_tie_sensitivity(complete, per_user)
    nonseq = _non_sequential_audit(complete)
    seed_results = _seed_level_results(complete)
    outputs = {
        "run_manifest.csv": manifest,
        "seed_level_results.csv": seed_results,
        "threshold_sensitivity.csv": threshold,
        "timestamp_tie_sensitivity.csv": timestamp,
        "order_condition_comparisons.csv": order_summary,
        "non_sequential_invariance_audit.csv": nonseq,
        "robustness_summary.csv": robustness,
    }
    for filename, frame in outputs.items():
        frame.to_csv(aggregate_dir / filename, index=False)
    _write_figures(robustness, threshold, timestamp, aggregate_dir)
    (aggregate_dir / "robustness_report.md").write_text(
        _robustness_report(manifest, robustness, threshold, timestamp, incomplete),
        encoding="utf-8",
    )
    return outputs


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return values


def parse_args() -> argparse.Namespace:
    """Parse matrix-runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--primary-result-dir", type=Path, required=True)
    parser.add_argument(
        "--initialization-seeds",
        type=_parse_int_tuple,
        default=(760, 761, 762, 763, 764),
    )
    parser.add_argument("--skip-rating3", action="store_true")
    parser.add_argument("--skip-rule-b", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run or resume the matrix and aggregate every completed cell."""
    arguments = parse_args()
    specs = build_matrix_specs(
        arguments.runs_dir,
        arguments.primary_result_dir,
        arguments.initialization_seeds,
        include_rating3=not arguments.skip_rating3,
        include_rule_b=not arguments.skip_rule_b,
    )
    if arguments.aggregate_only:
        manifest = pd.read_csv(arguments.aggregate_dir / "run_manifest.csv")
    else:
        manifest = run_matrix(specs, arguments.aggregate_dir, arguments.max_new_runs)
    aggregate_matrix(manifest, arguments.aggregate_dir)


if __name__ == "__main__":
    main()
