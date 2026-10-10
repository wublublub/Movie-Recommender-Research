"""Controlled order-sensitivity experiment for TSMC2014 Foursquare NYC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from movie_recommender.foursquare_data_loader import (
    DEFAULT_FOURSQUARE_DATA_DIR,
    load_foursquare_checkins,
    make_order_experiment_frames,
)
from movie_recommender.ml1m_order_experiment import (
    OrderExperimentConfig,
    run_experiment,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "foursquare_nyc_order_experiment"
)


def run_foursquare_experiment(
    data_dir: Path,
    output_dir: Path,
    fast_test: bool,
    fast_user_limit: int,
    user_limit: int | None,
    maximum_epochs: int,
    batch_size: int,
    lstm_batch_size: int,
    models: tuple[str, ...],
    model_initialization_seeds: tuple[int, ...],
    sequence_shuffle_seeds: tuple[int, ...],
) -> dict[str, object]:
    """Load NYC check-ins and run the shared controlled-order pipeline."""
    checkins = load_foursquare_checkins("NYC", data_dir)
    ratings, items, mapping = make_order_experiment_frames(checkins)
    config = OrderExperimentConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        relevance_threshold=None,
        model_initialization_seeds=model_initialization_seeds,
        negative_sampling_seed=17_760,
        sequence_shuffle_seeds=(
            sequence_shuffle_seeds[:1] if fast_test else sequence_shuffle_seeds
        ),
        bootstrap_seed=30_760,
        split_seed=40_760,
        maximum_epochs=maximum_epochs,
        patience=2,
        batch_size=batch_size,
        lstm_batch_size=lstm_batch_size,
        bootstrap_samples=1_000,
        fast_test=fast_test,
        fast_user_limit=fast_user_limit,
        user_limit=user_limit,
        equal_timestamp_ordering_rule="timestamp_source",
        models=models,
        candidate_policy="all_items",
        target_semantics="events",
        dataset_name="Foursquare TSMC2014 NYC",
        interaction_filename="dataset_TSMC2014_NYC.txt",
        timestamp_interpretation=(
            "recorded UTC check-in time; local-time features are not used"
        ),
        fpmc_parameter_options=((16, 3), (32, 3)),
        lstm_parameter_options=(
            (16, 32, 64, 0.001),
            (32, 64, 100, 0.0005),
        ),
        sasrec_parameter_options=(
            (32, 50, 1, 2, 0.1, 0.001),
            (64, 100, 2, 4, 0.2, 0.0005),
        ),
    )
    outputs = run_experiment(config, ratings=ratings, movies=items)
    mapping.to_csv(output_dir / "foursquare_item_mapping.csv", index=False)
    repeated = checkins.duplicated(["user_id", "item_id"], keep="first")
    tied = checkins.duplicated(["user_id", "timestamp"], keep="first")
    summary = {
        "dataset": "Foursquare TSMC2014 NYC",
        "raw_rows": len(checkins),
        "users": int(checkins["user_id"].nunique()),
        "venues": int(checkins["item_id"].nunique()),
        "repeated_user_venue_events_beyond_first": int(repeated.sum()),
        "repeated_user_venue_event_percentage": float(repeated.mean() * 100),
        "per_user_tied_timestamp_events_beyond_first": int(tied.sum()),
        "per_user_tied_timestamp_event_percentage": float(tied.mean() * 100),
        "candidate_policy": "all catalog venues, including previously visited venues",
        "target_semantics": "event-weighted; duplicate held-out visits are retained",
        "compatibility_adapter": (
            "Canonical user_id/item_id/timestamp fields are temporarily mapped to "
            "the shared runner's UserID/MovieID/Rating/Timestamp schema."
        ),
    }
    (output_dir / "foursquare_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {**outputs, "foursquare_item_mapping.csv": mapping, "summary": summary}


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"fpmc", "lstm", "sasrec"}
    unknown = set(models) - allowed
    if not models or unknown:
        raise argparse.ArgumentTypeError(
            f"Models must be a non-empty subset of {sorted(allowed)}"
        )
    return models


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return values


def parse_args() -> argparse.Namespace:
    """Parse Foursquare experiment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_FOURSQUARE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fast-test", action="store_true")
    parser.add_argument("--fast-user-limit", type=int, default=24)
    parser.add_argument(
        "--user-limit",
        type=int,
        help="Deterministically sample eligible users without enabling fast-test shortcuts.",
    )
    parser.add_argument("--maximum-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lstm-batch-size", type=int, default=4)
    parser.add_argument(
        "--models", type=_parse_models, default=("fpmc", "lstm", "sasrec")
    )
    parser.add_argument(
        "--model-initialization-seeds", type=_parse_int_tuple, default=(760,)
    )
    parser.add_argument(
        "--sequence-shuffle-seeds",
        type=_parse_int_tuple,
        default=(20_760, 20_761, 20_762),
    )
    return parser.parse_args()


def main() -> None:
    """Run the NYC experiment from the command line."""
    arguments = parse_args()
    run_foursquare_experiment(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        fast_test=arguments.fast_test,
        fast_user_limit=arguments.fast_user_limit,
        user_limit=arguments.user_limit,
        maximum_epochs=arguments.maximum_epochs,
        batch_size=arguments.batch_size,
        lstm_batch_size=arguments.lstm_batch_size,
        models=arguments.models,
        model_initialization_seeds=arguments.model_initialization_seeds,
        sequence_shuffle_seeds=arguments.sequence_shuffle_seeds,
    )


if __name__ == "__main__":
    main()
