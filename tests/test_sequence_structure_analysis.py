import numpy as np
import pandas as pd
import pytest

from movie_recommender.sequence_structure_analysis import (
    assign_quantile_groups,
    bootstrap_mean_interval,
    compute_user_order_effects,
    extract_feature_table,
    extract_user_sequence_features,
    feature_applicability_audit,
    load_movietweetings_artifacts,
)


def test_sequence_scale_repeat_entropy_and_recurrence_features() -> None:
    features = extract_user_sequence_features(
        [1, 2, 1, 2, 1], [0, 10, 20, 30, 40]
    )

    assert features["sequence_length"] == 5
    assert features["unique_item_count"] == 2
    assert features["unique_item_ratio"] == pytest.approx(0.4)
    assert features["repeat_event_rate"] == pytest.approx(0.6)
    assert features["immediate_repeat_rate"] == 0
    assert features["normalized_item_entropy"] == pytest.approx(
        -(0.6 * np.log(0.6) + 0.4 * np.log(0.4)) / np.log(2)
    )
    assert features["repeated_bigram_rate"] == pytest.approx(0.5)
    assert features["local_transition_concentration"] == 1.0
    assert features["repeated_trigram_rate"] == pytest.approx(1 / 3)


def test_timestamp_gap_ties_short_windows_and_burstiness() -> None:
    features = extract_user_sequence_features(
        [1, 2, 3, 4], [0, 0, 60, 3660]
    )

    assert features["timestamp_tie_rate"] == pytest.approx(1 / 3)
    assert features["median_gap_seconds"] == 60
    assert features["mean_gap_seconds"] == 1220
    assert features["gap_standard_deviation_seconds"] == pytest.approx(
        np.std([0, 60, 3600])
    )
    assert features["short_gap_rate_1_minute"] == pytest.approx(2 / 3)
    assert features["short_gap_rate_1_hour"] == 1.0
    positive = np.asarray([60, 3600], dtype=float)
    expected = (positive.std() - positive.mean()) / (
        positive.std() + positive.mean()
    )
    assert features["burstiness"] == pytest.approx(expected)


def test_feature_extraction_is_deterministic_and_excludes_test() -> None:
    split = pd.DataFrame(
        {
            "UserID": [1, 1, 1, 1],
            "MovieID": [2, 1, 2, 99],
            "Timestamp": [20, 10, 30, 40],
            "_user_order_rank": [1, 0, 2, 3],
            "split": ["train", "train", "validation", "test"],
        }
    )

    first = extract_feature_table(split, "example")
    second = extract_feature_table(split.sample(frac=1, random_state=3), "example")

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "sequence_length"] == 3
    assert first.loc[0, "unique_item_count"] == 2


def test_paired_effect_averages_shuffles_then_initializations() -> None:
    rows = []
    for initialization_seed, original, shuffled in [
        (10, 0.6, [0.2, 0.4]),
        (11, 0.4, [0.1, 0.3]),
    ]:
        rows.append(
            {
                "user_id": 1,
                "model_name": "fpmc",
                "ndcg_10": original,
                "order_condition": "original_order",
                "model_initialization_seed": initialization_seed,
                "sequence_shuffle_seed": np.nan,
            }
        )
        for shuffle_seed, score in enumerate(shuffled):
            rows.append(
                {
                    "user_id": 1,
                    "model_name": "fpmc",
                    "ndcg_10": score,
                    "order_condition": "fully_shuffled_pretest",
                    "model_initialization_seed": initialization_seed,
                    "sequence_shuffle_seed": shuffle_seed,
                }
            )

    aggregate, paired = compute_user_order_effects(pd.DataFrame(rows), "example")

    assert paired.sort_values("model_initialization_seed")["delta_ndcg_10"].tolist() == pytest.approx([0.3, 0.2])
    assert aggregate.loc[0, "delta_ndcg_10"] == pytest.approx(0.25)
    assert aggregate.loc[0, "original_ndcg_10"] == pytest.approx(0.5)
    assert aggregate.loc[0, "shuffled_ndcg_10"] == pytest.approx(0.25)
    assert aggregate.loc[0, "initialization_seed_count"] == 2
    assert aggregate.loc[0, "shuffle_seed_count"] == 2


def test_unmatched_initialization_seed_is_rejected() -> None:
    metrics = pd.DataFrame(
        [
            {
                "user_id": 1,
                "model_name": "fpmc",
                "ndcg_10": 0.5,
                "order_condition": "original_order",
                "model_initialization_seed": 10,
                "sequence_shuffle_seed": np.nan,
            },
            {
                "user_id": 1,
                "model_name": "fpmc",
                "ndcg_10": 0.2,
                "order_condition": "fully_shuffled_pretest",
                "model_initialization_seed": 11,
                "sequence_shuffle_seed": 20,
            },
        ]
    )

    with pytest.raises(ValueError, match="pairing failed"):
        compute_user_order_effects(metrics, "example")


def test_quantile_assignment_preserves_groups_and_constant_is_missing() -> None:
    groups = assign_quantile_groups(pd.Series(range(9)))
    assert groups.value_counts().to_dict() == {"Low": 3, "Medium": 3, "High": 3}
    assert assign_quantile_groups(pd.Series([1, 1, 1])).isna().all()


def test_bootstrap_is_reproducible() -> None:
    values = [0.0, 0.1, 0.2, 0.3]
    first = bootstrap_mean_interval(values, samples=100, seed=123)
    second = bootstrap_mean_interval(values, samples=100, seed=123)
    assert first == second


def test_movietweetings_adapter_preserves_pretest_history_and_pairing(tmp_path) -> None:
    split_rows = []
    metric_rows = []
    for user_id in (17, 29):
        for index, (movie_id, split) in enumerate(
            ((10, "train"), (20, "train"), (30, "validation"), (40, "test"))
        ):
            split_rows.append({
                "UserID": user_id, "MovieID": movie_id + user_id,
                "Timestamp": 100 + index * 60, "_source_order": index,
                "split": split,
            })
        for model in ("fpmc", "lstm", "sasrec"):
            for initialization_seed in range(760, 765):
                metric_rows.append({
                    "user_id": user_id, "model_name": model,
                    "order_condition": "original_order", "ndcg_10": 0.4,
                    "model_initialization_seed": initialization_seed,
                    "sequence_shuffle_seed": np.nan,
                })
                for shuffle_seed in (20760, 20761, 20762):
                    metric_rows.append({
                        "user_id": user_id, "model_name": model,
                        "order_condition": "fully_shuffled_pretest", "ndcg_10": 0.1,
                        "model_initialization_seed": initialization_seed,
                        "sequence_shuffle_seed": shuffle_seed,
                    })
                metric_rows.append({
                    "user_id": user_id, "model_name": model,
                    "order_condition": "recent_context_fixed", "ndcg_10": 0.9,
                    "model_initialization_seed": initialization_seed,
                    "sequence_shuffle_seed": np.nan,
                })
    metric_path = tmp_path / "metrics.csv"
    split_path = tmp_path / "split.csv"
    pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
    pd.DataFrame(split_rows).sample(frac=1, random_state=4).to_csv(split_path, index=False)

    metrics, features = load_movietweetings_artifacts(metric_path, split_path)
    assert set(features["user_id"]) == {17, 29}
    assert features["sequence_length"].eq(3).all()
    assert features["unique_item_ratio"].eq(1).all()
    assert features["mean_gap_seconds"].eq(60).all()
    assert set(metrics["order_condition"]) == {
        "original_order", "fully_shuffled_pretest", "recent_context_fixed"
    }
    effects, paired = compute_user_order_effects(metrics, "MovieTweetings")
    assert len(paired) == 30
    assert effects["delta_ndcg_10"].to_numpy() == pytest.approx([0.3] * 6)
    audit = feature_applicability_audit(features)
    assert not audit.loc[audit["feature"].eq("repeat_event_rate"), "applicable"].any()
    assert audit.loc[audit["feature"].eq("sequence_length"), "support_count"].iloc[0] == 2
    assert not audit.loc[audit["feature"].eq("sequence_length"), "applicable"].iloc[0]


def test_movietweetings_adapter_rejects_incomplete_shuffle_cells(tmp_path) -> None:
    metric_path = tmp_path / "metrics.csv"
    split_path = tmp_path / "split.csv"
    pd.DataFrame({
        "user_id": [1, 1], "model_name": ["fpmc", "fpmc"],
        "order_condition": ["original_order", "fully_shuffled_pretest"],
        "ndcg_10": [0.4, 0.1], "model_initialization_seed": [760, 760],
        "sequence_shuffle_seed": [np.nan, 20760],
    }).to_csv(metric_path, index=False)
    pd.DataFrame({
        "UserID": [1, 1], "MovieID": [1, 2], "Timestamp": [1, 2],
        "_source_order": [0, 1], "split": ["train", "validation"],
    }).to_csv(split_path, index=False)
    with pytest.raises(ValueError, match="initialization seeds"):
        load_movietweetings_artifacts(metric_path, split_path)
