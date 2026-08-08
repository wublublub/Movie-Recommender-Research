from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movie_recommender.ml1m_model_feasibility import (
    ExperimentConfig,
    _rank_metrics,
    build_advantage_table,
    chronological_split,
    sample_candidates,
    shuffle_sequences,
)


def _ratings() -> pd.DataFrame:
    rows = []
    for user_id in (1, 2):
        for position in range(8):
            rows.append(
                {
                    "UserID": user_id,
                    "MovieID": user_id * 100 + position,
                    "Rating": 5 if user_id == 1 or position < 6 else 2,
                    "Timestamp": user_id * 1000 + position,
                }
            )
    return pd.DataFrame(rows)


def test_chronological_split_is_leakage_safe_and_reports_exclusions(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        minimum_training_interactions=5,
    )
    split = chronological_split(_ratings(), config)

    assert split.eligible_users["user_id"].tolist() == [1]
    assert split.excluded_users["user_id"].tolist() == [2]
    assert split.train["MovieID"].tolist() == [100, 101, 102, 103, 104, 105]
    assert split.validation.iloc[0]["MovieID"] == 106
    assert split.test.iloc[0]["MovieID"] == 107
    assert split.train["Timestamp"].max() < split.test.iloc[0]["Timestamp"]


def test_candidate_sampling_excludes_all_items_seen_before_target(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        minimum_training_interactions=5,
        negative_samples=3,
    )
    split = chronological_split(_ratings(), config)
    candidates, _ = sample_candidates(
        split,
        split.test,
        np.arange(100, 120),
        negative_samples=3,
        seed=760,
        split_name="test",
    )

    candidate_items = set(candidates[1].candidate_movie_ids)
    assert candidates[1].target_movie_id in candidate_items
    assert candidate_items.isdisjoint(set(range(100, 107)))
    assert len(candidate_items) == 4


def test_rank_metrics_uses_deterministic_item_id_tie_breaking() -> None:
    scores = np.asarray([1.0, 1.0, 0.5])
    candidate_ids = np.asarray([10, 20, 30])

    hit, recall, ndcg, mrr, rank = _rank_metrics(
        scores, candidate_ids, target_movie_id=20
    )

    assert rank == 2
    assert hit == recall == 1
    assert ndcg == pytest.approx(1 / np.log2(3))
    assert mrr == pytest.approx(0.5)


def test_shuffle_preserves_items_and_is_deterministic() -> None:
    sequences = [np.asarray([1, 2, 3, 4]), np.asarray([5, 6, 7])]

    first = shuffle_sequences(sequences, seed=760)
    second = shuffle_sequences(sequences, seed=760)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(
        sorted(original.tolist()) == sorted(shuffled.tolist())
        for original, shuffled in zip(sequences, first)
    )


def test_advantage_table_reports_best_model_ties_explicitly() -> None:
    rows = []
    values = {
        1: {
            "popularity": 0.0,
            "item_knn": 0.0,
            "matrix_factorization": 0.0,
            "fpmc_original": 0.0,
            "fpmc_shuffled_seed_1": 0.0,
        },
        2: {
            "popularity": 0.1,
            "item_knn": 0.2,
            "matrix_factorization": 0.3,
            "fpmc_original": 0.4,
            "fpmc_shuffled_seed_1": 0.2,
        },
    }
    for user_id, model_values in values.items():
        for model_name, ndcg in model_values.items():
            rows.append(
                {"user_id": user_id, "model_name": model_name, "ndcg_10": ndcg}
            )
    features = pd.DataFrame({"user_id": [1, 2], "rating_count": [5, 5]})

    matrix, _ = build_advantage_table(pd.DataFrame(rows), features)

    assert matrix.loc[matrix["user_id"].eq(1), "best_model"].iloc[0] == "tie"
    assert matrix.loc[matrix["user_id"].eq(1), "all_core_models_zero"].iloc[0]
    assert (
        matrix.loc[matrix["user_id"].eq(2), "best_model"].iloc[0]
        == "fpmc_original"
    )
