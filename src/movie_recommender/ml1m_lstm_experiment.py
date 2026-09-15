"""Compatibility CLI for running LSTM through the shared formal experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from movie_recommender.lstm_model import make_history_batch, make_padded_batch
from movie_recommender.ml1m_order_experiment import (
    DEFAULT_OUTPUT_DIR,
    OrderExperimentConfig,
    run_experiment,
)


def run_lstm_experiment(
    fast_test: bool = False, output_dir: Path | None = None
) -> dict[str, object]:
    """Run LSTM through the authoritative shared order pipeline.

    Args:
        fast_test: Whether to execute the explicitly non-final smoke configuration.
        output_dir: Destination for the common formal artifact schema.

    Returns:
        In-memory tables written by the shared experiment runner.
    """
    destination = output_dir or DEFAULT_OUTPUT_DIR.with_name("lstm_order_experiment")
    shuffle_seeds = (20_760,) if fast_test else (20_760, 20_761, 20_762)
    return run_experiment(
        OrderExperimentConfig(
            output_dir=destination,
            models=("lstm",),
            model_initialization_seeds=(760,),
            sequence_shuffle_seeds=shuffle_seeds,
            maximum_epochs=2 if fast_test else 5,
            fast_test=fast_test,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse compatibility-command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-test", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the compatibility command."""
    arguments = parse_args()
    run_lstm_experiment(arguments.fast_test, arguments.output_dir)


if __name__ == "__main__":
    main()
