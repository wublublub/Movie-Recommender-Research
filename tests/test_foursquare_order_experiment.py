from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movie_recommender.foursquare_data_loader import (
    load_foursquare_checkins,
    make_order_experiment_frames,
)
from movie_recommender.ml1m_order_experiment import (
    OrderExperimentConfig,
    assert_split_and_candidates,
    build_order_histories,
    run_experiment,
)
from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig,
    build_ranking_sets,
    event_ranking_metrics,
    revised_chronological_split,
)


def _raw_line(user_id: int, venue_id: str, timestamp: str, category: str) -> str:
    return "\t".join(
        [
            str(user_id),
            venue_id,
            f"category-{venue_id}",
            category,
            "40.7",
            "-74.0",
            "-240",
            timestamp,
        ]
    )


def _write_checkins(path: Path) -> None:
    rows = [
        _raw_line(1, "venue-b", "Tue Apr 03 18:00:09 +0000 2012", "Café"),
        _raw_line(1, "venue-a", "Tue Apr 03 18:00:09 +0000 2012", "Park"),
        _raw_line(1, "venue-b", "Tue Apr 03 19:00:09 +0000 2012", "Café"),
    ]
    path.write_bytes(("\n".join(rows) + "\n").encode("latin-1"))


def _revisit_ratings() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = [1, 2, 1, 3, 4, 5, 6, 7, 1, 8, 1, 1]
    ratings = pd.DataFrame(
        {
            "UserID": [1] * len(events),
            "MovieID": events,
            "Rating": [1] * len(events),
            "Timestamp": np.arange(100, 100 + len(events)),
        }
    )
    items = pd.DataFrame({"MovieID": np.arange(1, 10)})
    return ratings, items


def test_foursquare_parser_uses_tabs_latin1_and_utc(tmp_path: Path) -> None:
    path = tmp_path / "dataset_TSMC2014_NYC.txt"
    _write_checkins(path)

    checkins = load_foursquare_checkins("NYC", tmp_path)

    assert len(checkins) == 3
    assert checkins["venue_category_name"].tolist() == ["Café", "Park", "Café"]
    assert str(checkins["timestamp"].dtype) == "datetime64[us, UTC]"
    assert checkins["source_order"].tolist() == [0, 1, 2]
    assert checkins["item_id"].tolist() == ["venue-b", "venue-a", "venue-b"]


def test_adapter_preserves_repeated_events_and_source_order(tmp_path: Path) -> None:
    path = tmp_path / "dataset_TSMC2014_NYC.txt"
    _write_checkins(path)
    checkins = load_foursquare_checkins("NYC", tmp_path)

    ratings, items, mapping = make_order_experiment_frames(checkins)
    venue_b = int(mapping.loc[mapping["item_id"].eq("venue-b"), "MovieID"].iloc[0])

    assert ratings["MovieID"].tolist().count(venue_b) == 2
    assert ratings["Timestamp"].tolist()[:2] == [1333476009, 1333476009]
    assert len(items) == 2


def test_source_ties_split_revisits_and_eligibility_are_deterministic() -> None:
    ratings, items = _revisit_ratings()
    ratings.loc[8:9, "Timestamp"] = 108
    config = RevisedConfig(
        relevance_threshold=-np.inf,
        equal_timestamp_ordering_rule="timestamp_source",
    )

    first = revised_chronological_split(ratings, config)
    second = revised_chronological_split(ratings, config)

    assert len(first.eligible_users) == 1
    assert first.validation_positive["MovieID"].tolist() == [1, 8]
    assert first.test_positive["MovieID"].tolist() == [1, 1]
    assert first.split_table["_source_order"].tolist() == second.split_table[
        "_source_order"
    ].tolist()
    train_rows = set(first.train_positive["_source_order"])
    test_rows = set(first.test_positive["_source_order"])
    assert train_rows.isdisjoint(test_rows)

    short = revised_chronological_split(ratings.iloc[:8], config)
    assert short.eligible_users.empty
    assert len(items) == 9


def test_user_limit_is_deterministic_without_fast_mode() -> None:
    ratings = pd.concat(
        [
            _revisit_ratings()[0].assign(UserID=user_id)
            for user_id in range(1, 7)
        ],
        ignore_index=True,
    )
    config = RevisedConfig(
        relevance_threshold=-np.inf,
        random_seed=123,
        user_limit=3,
        fast_test=False,
        equal_timestamp_ordering_rule="timestamp_source",
    )

    first = revised_chronological_split(ratings, config)
    second = revised_chronological_split(ratings, config)

    assert len(first.eligible_users) == 3
    assert first.eligible_users["user_id"].tolist() == second.eligible_users[
        "user_id"
    ].tolist()


def test_all_item_candidates_keep_seen_revisit_targets_and_events() -> None:
    ratings, items = _revisit_ratings()
    split = revised_chronological_split(
        ratings,
        RevisedConfig(
            relevance_threshold=-np.inf,
            equal_timestamp_ordering_rule="timestamp_source",
        ),
    )
    validation, _ = build_ranking_sets(
        split,
        split.validation_positive,
        items["MovieID"].to_numpy(),
        candidate_policy="all_items",
    )
    test, _ = build_ranking_sets(
        split,
        split.test_positive,
        items["MovieID"].to_numpy(),
        candidate_policy="all_items",
    )

    assert_split_and_candidates(split, validation, test)
    assert test[1].candidate_movie_ids.tolist() == items["MovieID"].tolist()
    assert test[1].relevant_movie_ids == {1}
    assert test[1].relevant_movie_events.tolist() == [1, 1]
    assert 1 in set(split.train_positive["MovieID"])


def test_shuffle_preserves_duplicate_multisets_lengths_and_seeds() -> None:
    ratings, items = _revisit_ratings()
    split = revised_chronological_split(
        ratings,
        RevisedConfig(
            relevance_threshold=-np.inf,
            equal_timestamp_ordering_rule="timestamp_source",
        ),
    )
    user_ids = split.eligible_users["user_id"].to_numpy(np.int64)
    item_to_index = {
        int(item): index for index, item in enumerate(items["MovieID"].to_numpy())
    }
    first = build_order_histories(split, user_ids, item_to_index, (101, 202))
    second = build_order_histories(split, user_ids, item_to_index, (101, 202))

    assert len(first.original[0]) == len(first.fully_shuffled[101][0])
    assert np.array_equal(
        np.sort(first.original[0]), np.sort(first.fully_shuffled[101][0])
    )
    assert np.array_equal(first.fully_shuffled[101][0], second.fully_shuffled[101][0])
    assert not np.array_equal(first.fully_shuffled[101][0], first.fully_shuffled[202][0])
    repeated_index = item_to_index[1]
    assert list(first.original[0]).count(repeated_index) == 3


def test_event_metrics_weight_duplicate_target_occurrences() -> None:
    candidates = np.asarray([1, 2, 3, 4])
    scores = np.asarray([4.0, 2.0, 3.0, 1.0])
    events = np.asarray([1, 2, 2])

    metrics, top_items = event_ranking_metrics(scores, candidates, events, k=2)

    assert top_items.tolist() == [1, 3]
    assert metrics["recall_10"] == pytest.approx(1 / 3)
    assert metrics["ndcg_10"] == pytest.approx(1 / 3)
    assert metrics["mrr_10"] == pytest.approx(1 / 3)


def test_small_foursquare_semantics_run_all_sequential_models(tmp_path: Path) -> None:
    rows = []
    for user_id in range(1, 5):
        for position in range(12):
            rows.append(
                {
                    "UserID": user_id,
                    "MovieID": 1 + (user_id + position) % 12,
                    "Rating": 1,
                    "Timestamp": 1_000 + user_id * 100 + position,
                }
            )
    ratings = pd.DataFrame(rows)
    items = pd.DataFrame({"MovieID": np.arange(1, 14)})
    output = tmp_path / "output"
    results = run_experiment(
        OrderExperimentConfig(
            data_dir=tmp_path,
            output_dir=output,
            relevance_threshold=None,
            model_initialization_seeds=(7,),
            sequence_shuffle_seeds=(11,),
            maximum_epochs=1,
            batch_size=4,
            lstm_batch_size=2,
            bootstrap_samples=20,
            fast_test=True,
            fast_user_limit=4,
            equal_timestamp_ordering_rule="timestamp_source",
            models=("fpmc", "lstm", "sasrec"),
            candidate_policy="all_items",
            target_semantics="events",
            dataset_name="Synthetic Foursquare",
            interaction_filename="missing.txt",
            fpmc_parameter_options=((4, 1),),
            lstm_parameter_options=((4, 4, 8, 0.001),),
            sasrec_parameter_options=((4, 8, 1, 1, 0.0, 0.001),),
        ),
        ratings=ratings,
        movies=items,
    )

    per_user = results["per_user_model_metrics.csv"]
    assert set(per_user["model_name"]) == {"fpmc", "lstm", "sasrec"}
    assert {"original_order", "fully_shuffled_pretest"}.issubset(
        set(per_user["order_condition"])
    )
    controlled_counts = per_user.pivot_table(
        index=["user_id", "model_name", "model_initialization_seed"],
        columns="order_condition",
        values=["number_of_test_targets", "number_of_candidates"],
        aggfunc="first",
    )
    for column in ("number_of_test_targets", "number_of_candidates"):
        assert np.array_equal(
            controlled_counts[(column, "original_order")],
            controlled_counts[(column, "fully_shuffled_pretest")],
        )
    statistics = results["statistical_comparisons.csv"]
    assert statistics["relative_order_drop_percent"].notna().all()
    assert set(results["model_runtime.csv"]["model_name"]) == {
        "fpmc",
        "lstm",
        "sasrec",
    }
    assert (output / "dataset_metadata.json").exists()
    environment = pd.read_json(output / "environment_and_hardware.json", typ="series")
    assert environment["peak_process_rss_bytes"] > 0
