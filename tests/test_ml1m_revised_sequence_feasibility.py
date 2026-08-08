from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig,
    build_ranking_sets,
    revised_chronological_split,
    set_ranking_metrics,
)


def _ratings(user_count: int = 2, positives: int = 20) -> pd.DataFrame:
    rows = []
    for user_id in range(1, user_count + 1):
        for position in range(positives):
            rows.append(
                {
                    "UserID": user_id,
                    "MovieID": user_id * 100 + position,
                    "Rating": 5,
                    "Timestamp": user_id * 1000 + position // 2,
                }
            )
    return pd.DataFrame(rows)


def test_hybrid_split_uses_timestamp_then_movie_id_and_preserves_minimums(
    tmp_path: Path,
) -> None:
    config = RevisedConfig(data_dir=tmp_path, output_dir=tmp_path)
    split = revised_chronological_split(_ratings(user_count=1), config)

    assert len(split.train_positive) == 16
    assert len(split.validation_positive) == 2
    assert len(split.test_positive) == 2
    assert split.validation_positive["MovieID"].tolist() == [116, 117]
    assert split.test_positive["MovieID"].tolist() == [118, 119]
    assert split.training_history["Timestamp"].max() <= split.train_positive[
        "Timestamp"
    ].max()


def test_split_reports_users_who_cannot_support_multi_target_evaluation(
    tmp_path: Path,
) -> None:
    config = RevisedConfig(data_dir=tmp_path, output_dir=tmp_path)
    split = revised_chronological_split(_ratings(user_count=1, positives=8), config)

    assert split.eligible_users.empty
    assert split.excluded_users.iloc[0]["exclusion_reason"] == (
        "insufficient_for_hybrid_multi_target_split"
    )


def test_full_ranking_candidates_include_targets_and_exclude_seen_items(
    tmp_path: Path,
) -> None:
    config = RevisedConfig(data_dir=tmp_path, output_dir=tmp_path)
    split = revised_chronological_split(_ratings(user_count=1), config)
    ranking_sets, _ = build_ranking_sets(
        split, split.test_positive, np.arange(100, 130)
    )

    candidate_items = set(ranking_sets[1].candidate_movie_ids)
    assert ranking_sets[1].relevant_movie_ids == {118, 119}
    assert {118, 119}.issubset(candidate_items)
    assert candidate_items.isdisjoint(set(range(100, 118)))


def test_set_metrics_use_multiple_relevant_targets() -> None:
    candidates = np.asarray([1, 2, 3, 4, 5])
    scores = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    metrics, top_items = set_ranking_metrics(
        scores, candidates, relevant_movie_ids={1, 3}, k=3
    )

    assert top_items.tolist() == [1, 2, 3]
    assert metrics["precision_10"] == pytest.approx(2 / 3)
    assert metrics["recall_10"] == 1
    assert metrics["hit_rate_10"] == 1
    assert metrics["mrr_10"] == 1
