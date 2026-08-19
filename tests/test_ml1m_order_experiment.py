from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import torch

from movie_recommender.ml1m_order_experiment import (
    FULL_SHUFFLE,
    ORIGINAL,
    RECENT_FIXED,
    OrderExperimentConfig,
    assert_split_and_candidates,
    build_order_histories,
    run_experiment,
)
from movie_recommender.ml1m_model_feasibility import build_interaction_matrix
from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig,
    SASRec,
    SASRecParameters,
    build_ranking_sets,
    revised_chronological_split,
    set_ranking_metrics,
)
from movie_recommender.ml1m_robustness import (
    build_matrix_specs,
    hierarchical_bootstrap,
)


def _synthetic_data(
    user_count: int = 8, interactions_per_user: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    movie_count = 50
    for user_id in range(1, user_count + 1):
        for position in range(interactions_per_user):
            rows.append(
                {
                    "UserID": user_id,
                    "MovieID": 1 + (user_id * 7 + position * 3) % movie_count,
                    "Rating": 5 if position % 5 else 3,
                    "Timestamp": 1_000_000 + user_id * 10_000 + position // 2,
                }
            )
    movies = pd.DataFrame(
        {
            "MovieID": np.arange(1, movie_count + 1),
            "Title": [f"Movie {index}" for index in range(1, movie_count + 1)],
            "Genres": ["Drama"] * movie_count,
        }
    )
    return pd.DataFrame(rows), movies


def _split_and_histories(tmp_path: Path):
    ratings, movies = _synthetic_data()
    split = revised_chronological_split(
        ratings,
        RevisedConfig(
            data_dir=tmp_path,
            output_dir=tmp_path,
            relevance_threshold=4,
            minimum_training_interactions=5,
            random_seed=99,
        ),
    )
    movie_ids = movies["MovieID"].to_numpy()
    item_to_index = {int(item): index for index, item in enumerate(movie_ids)}
    user_ids = split.eligible_users["user_id"].to_numpy(np.int64)
    histories = build_order_histories(split, user_ids, item_to_index, (101, 202))
    return split, histories, movie_ids, item_to_index


def test_split_is_deterministic_and_uses_equal_timestamp_tie_breaking(
    tmp_path: Path,
) -> None:
    ratings, _ = _synthetic_data(user_count=1)
    config = RevisedConfig(data_dir=tmp_path, output_dir=tmp_path, random_seed=7)

    first = revised_chronological_split(ratings, config)
    second = revised_chronological_split(ratings.sample(frac=1, random_state=5), config)

    columns = ["UserID", "MovieID", "Timestamp"]
    pdt.assert_frame_equal(
        first.split_table[columns].reset_index(drop=True),
        second.split_table[columns].reset_index(drop=True),
    )
    for _, frame in first.split_table.groupby("Timestamp"):
        assert frame["MovieID"].tolist() == sorted(frame["MovieID"].tolist())


def test_equal_timestamp_rules_are_distinct_and_deterministic(tmp_path: Path) -> None:
    movie_ids = [12, 3, 18, 7, 14, 1, 19, 8, 16, 2, 20, 4]
    ratings = pd.DataFrame(
        {
            "UserID": [1] * len(movie_ids),
            "MovieID": movie_ids,
            "Rating": [5] * len(movie_ids),
            "Timestamp": [100] * len(movie_ids),
        }
    )
    common = dict(
        data_dir=tmp_path,
        output_dir=tmp_path,
        minimum_training_interactions=5,
    )
    rule_a = revised_chronological_split(
        ratings,
        RevisedConfig(
            **common, equal_timestamp_ordering_rule="timestamp_movie_id_source"
        ),
    )
    rule_b = revised_chronological_split(
        ratings,
        RevisedConfig(**common, equal_timestamp_ordering_rule="timestamp_source"),
    )

    assert rule_a.test_positive["MovieID"].tolist() == [19, 20]
    assert rule_b.test_positive["MovieID"].tolist() == [20, 4]
    assert rule_b.split_table["MovieID"].tolist() == movie_ids


def test_no_test_leakage_and_targets_are_common(tmp_path: Path) -> None:
    split, _, movie_ids, _ = _split_and_histories(tmp_path)
    validation_sets, _ = build_ranking_sets(
        split, split.validation_positive, movie_ids
    )
    test_sets, _ = build_ranking_sets(split, split.test_positive, movie_ids)

    assert_split_and_candidates(split, validation_sets, test_sets)
    train_rows = set(
        zip(split.train_positive["UserID"], split.train_positive["_source_order"])
    )
    test_rows = set(
        zip(split.test_positive["UserID"], split.test_positive["_source_order"])
    )
    assert train_rows.isdisjoint(test_rows)


def test_full_shuffle_preserves_multisets_and_destroys_pretest_order(
    tmp_path: Path,
) -> None:
    _, histories, _, _ = _split_and_histories(tmp_path)

    assert all(
        np.array_equal(np.sort(original), np.sort(shuffled))
        for original, shuffled in zip(
            histories.original, histories.fully_shuffled[101], strict=True
        )
    )
    assert any(
        not np.array_equal(original, shuffled)
        for original, shuffled in zip(
            histories.original, histories.fully_shuffled[101], strict=True
        )
    )
    assert any(
        not np.array_equal(shuffled[-len(recent) :], recent)
        for shuffled, recent in zip(
            histories.fully_shuffled[101], histories.validation, strict=True
        )
    )


def test_recent_context_fixed_shuffle_preserves_suffix(tmp_path: Path) -> None:
    _, histories, _, _ = _split_and_histories(tmp_path)

    for original, shuffled, recent in zip(
        histories.original,
        histories.recent_context_fixed[101],
        histories.validation,
        strict=True,
    ):
        assert np.array_equal(np.sort(original), np.sort(shuffled))
        assert np.array_equal(shuffled[-len(recent) :], recent)


def test_shuffle_seeds_are_deterministic_and_distinct(tmp_path: Path) -> None:
    split, first, _, item_to_index = _split_and_histories(tmp_path)
    user_ids = split.eligible_users["user_id"].to_numpy(np.int64)
    second = build_order_histories(split, user_ids, item_to_index, (101, 202))

    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first.fully_shuffled[101], second.fully_shuffled[101], strict=True
        )
    )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            first.fully_shuffled[101], first.fully_shuffled[202], strict=True
        )
    )


def test_non_sequential_inputs_are_order_invariant(tmp_path: Path) -> None:
    _, histories, movie_ids, _ = _split_and_histories(tmp_path)
    original = build_interaction_matrix(histories.original, len(movie_ids))
    fully_shuffled = build_interaction_matrix(
        histories.fully_shuffled[101], len(movie_ids)
    )
    recent_fixed = build_interaction_matrix(
        histories.recent_context_fixed[101], len(movie_ids)
    )

    assert (original != fully_shuffled).nnz == 0
    assert (original != recent_fixed).nnz == 0


def test_metric_ties_use_item_id_and_multiple_targets() -> None:
    metrics, top_items = set_ranking_metrics(
        np.asarray([1.0, 1.0, 0.5, 0.5]),
        np.asarray([20, 10, 40, 30]),
        {10, 30},
        k=3,
    )

    assert top_items.tolist() == [10, 20, 30]
    assert metrics["precision_10"] == 2 / 3
    assert metrics["recall_10"] == 1


def test_sasrec_attention_is_causal() -> None:
    torch.manual_seed(3)
    model = SASRec(20, SASRecParameters(8, 5, 1, 2, 0.0, 0.001)).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 9, 10]])

    with torch.no_grad():
        first_hidden = model(first)
        changed_hidden = model(changed_future)

    assert torch.allclose(first_hidden[:, :3], changed_hidden[:, :3], atol=1e-6)
    assert not torch.allclose(first_hidden[:, -1], changed_hidden[:, -1])


def test_small_end_to_end_outputs_and_reproducibility(tmp_path: Path) -> None:
    ratings, movies = _synthetic_data()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    common = dict(
        relevance_threshold=4,
        model_initialization_seeds=(11,),
        negative_sampling_seed=22,
        sequence_shuffle_seeds=(33,),
        bootstrap_seed=44,
        split_seed=55,
        maximum_epochs=1,
        patience=1,
        batch_size=8,
        bootstrap_samples=20,
        fast_test=True,
        fast_user_limit=8,
    )
    first = run_experiment(
        OrderExperimentConfig(output_dir=first_dir, **common), ratings, movies
    )
    second = run_experiment(
        OrderExperimentConfig(output_dir=second_dir, **common), ratings, movies
    )

    expected = {
        "experiment_config.json",
        "environment_and_hardware.json",
        "dataset_metadata.json",
        "split_summary.csv",
        "eligible_users.csv",
        "excluded_users.csv",
        "split_audit_table.csv",
        "validation_results.csv",
        "selected_hyperparameters.json",
        "per_user_model_metrics.csv",
        "aggregate_model_metrics.csv",
        "paired_order_differences.csv",
        "seed_level_results.csv",
        "statistical_comparisons.csv",
        "training_curves.csv",
        "report.md",
    }
    assert expected.issubset({path.name for path in first_dir.iterdir()})
    assert (first_dir / "non_sequential_invariance_audit.csv").exists()
    assert (first_dir / "figures" / "primary_effect_summary.png").exists()
    assert "NON-FINAL SMOKE TEST" in (first_dir / "report.md").read_text()

    metric_columns = {
        "user_id",
        "model_name",
        "order_condition",
        "ndcg_10",
        "sequence_shuffle_seed",
    }
    assert metric_columns.issubset(first["per_user_model_metrics.csv"].columns)
    per_user = first["per_user_model_metrics.csv"]
    candidate_counts = per_user.pivot_table(
        index="user_id",
        columns=["model_name", "order_condition"],
        values="number_of_candidates",
        aggfunc="first",
    )
    target_counts = per_user.pivot_table(
        index="user_id",
        columns=["model_name", "order_condition"],
        values="number_of_test_targets",
        aggfunc="first",
    )
    assert candidate_counts.nunique(axis=1).eq(1).all()
    assert target_counts.nunique(axis=1).eq(1).all()
    for filename in (
        "per_user_model_metrics.csv",
        "aggregate_model_metrics.csv",
        "statistical_comparisons.csv",
    ):
        pdt.assert_frame_equal(
            first[filename].reset_index(drop=True),
            second[filename].reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


def test_hierarchical_bootstrap_and_matrix_specs_are_reproducible(
    tmp_path: Path,
) -> None:
    effects = {
        "run_a": np.asarray([0.1, 0.2, 0.3]),
        "run_b": np.asarray([0.2, 0.3, 0.4]),
    }
    first = hierarchical_bootstrap(effects, samples=100, seed=77)
    second = hierarchical_bootstrap(effects, samples=100, seed=77)
    specs = build_matrix_specs(
        tmp_path / "runs",
        tmp_path / "primary",
        (760, 761, 762, 763, 764),
    )

    assert first == second
    assert first[0] < first[1]
    assert len(specs) == 7
    assert len({spec.run_id for spec in specs}) == len(specs)
    assert specs[0].reuse_existing
