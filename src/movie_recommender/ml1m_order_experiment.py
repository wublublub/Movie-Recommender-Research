"""Controlled MovieLens 1M experiments for recorded rating-submission order."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
import psutil
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD

from movie_recommender.ml1m_data_loader import (
    DEFAULT_ML1M_DATA_DIR,
    load_ml1m_movies,
    load_ml1m_ratings,
)
from movie_recommender.ml1m_model_feasibility import (
    FPMC,
    build_interaction_matrix,
    train_item_knn,
)
from movie_recommender.ml1m_revised_sequence_feasibility import (
    RankingSet,
    RevisedConfig,
    RevisedSplit,
    SASRec,
    SASRecParameters,
    build_ranking_sets,
    evaluate_full_ranker,
    make_sequences,
    revised_chronological_split,
    sasrec_score_matrix,
    set_deterministic_seed,
    training_arrays,
)


DEFAULT_OUTPUT_DIR: Final = (
    Path(__file__).resolve().parents[2] / "outputs" / "ml1m_order_experiment"
)
ORIGINAL: Final = "original_order"
FULL_SHUFFLE: Final = "fully_shuffled_pretest"
RECENT_FIXED: Final = "recent_context_fixed_shuffle"
SEQUENTIAL_MODELS: Final = ("fpmc", "sasrec")
NON_SEQUENTIAL_MODELS: Final = ("popularity", "item_knn", "matrix_factorization")
METRICS: Final = (
    "precision_10",
    "recall_10",
    "ndcg_10",
    "hit_rate_10",
    "mrr_10",
    "map_10",
)


@dataclass(frozen=True)
class OrderExperimentConfig:
    """Configuration for the controlled order experiment."""

    data_dir: Path = DEFAULT_ML1M_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    relevance_threshold: float | None = 4.0
    minimum_training_interactions: int = 5
    minimum_validation_targets: int = 2
    minimum_test_targets: int = 2
    holdout_fraction: float = 0.10
    model_initialization_seeds: tuple[int, ...] = (760,)
    negative_sampling_seed: int = 17_760
    sequence_shuffle_seeds: tuple[int, ...] = (20_760, 20_761, 20_762)
    bootstrap_seed: int = 30_760
    split_seed: int = 40_760
    maximum_epochs: int = 5
    patience: int = 2
    batch_size: int = 128
    bootstrap_samples: int = 1_000
    fast_test: bool = False
    fast_user_limit: int = 120
    equal_timestamp_ordering_rule: str = "timestamp_movie_id_source"

    def serializable(self) -> dict[str, object]:
        """Return JSON-compatible configuration and protocol declarations."""
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        values["model_initialization_seeds"] = list(self.model_initialization_seeds)
        values["sequence_shuffle_seeds"] = list(self.sequence_shuffle_seeds)
        values["run_kind"] = (
            "non_final_smoke_test" if self.fast_test else "formal_experiment"
        )
        values["timestamp_interpretation"] = (
            "rating-submission time, not confirmed watching time"
        )
        values["ordering_rule"] = self.equal_timestamp_ordering_rule
        values["primary_outcome"] = (
            "original_order NDCG@10 minus fully_shuffled_pretest NDCG@10"
        )
        return values


@dataclass(frozen=True)
class OrderHistories:
    """Mapped pre-test histories for each controlled order condition."""

    original: list[np.ndarray]
    fully_shuffled: dict[int, list[np.ndarray]]
    recent_context_fixed: dict[int, list[np.ndarray]]
    training: list[np.ndarray]
    validation: list[np.ndarray]


def _revised_config(config: OrderExperimentConfig) -> RevisedConfig:
    threshold = -np.inf if config.relevance_threshold is None else config.relevance_threshold
    return RevisedConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        relevance_threshold=float(threshold),
        minimum_training_interactions=config.minimum_training_interactions,
        minimum_validation_targets=config.minimum_validation_targets,
        minimum_test_targets=config.minimum_test_targets,
        holdout_fraction=config.holdout_fraction,
        random_seed=config.split_seed,
        maximum_epochs=config.maximum_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        shuffle_runs=len(config.sequence_shuffle_seeds),
        bootstrap_samples=config.bootstrap_samples,
        fast_test=config.fast_test,
        fast_user_limit=config.fast_user_limit,
        equal_timestamp_ordering_rule=config.equal_timestamp_ordering_rule,
    )


def build_order_histories(
    split: RevisedSplit,
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    shuffle_seeds: Sequence[int],
) -> OrderHistories:
    """Build original, fully shuffled, and recent-context-fixed histories."""
    training = make_sequences(split.train_positive, user_ids, item_to_index)
    validation = make_sequences(split.validation_positive, user_ids, item_to_index)
    original = [
        np.concatenate([train, recent]).astype(np.int32)
        for train, recent in zip(training, validation, strict=True)
    ]
    fully_shuffled: dict[int, list[np.ndarray]] = {}
    recent_fixed: dict[int, list[np.ndarray]] = {}
    for seed in shuffle_seeds:
        full_rows: list[np.ndarray] = []
        fixed_rows: list[np.ndarray] = []
        for user_index, (train, recent, pretest) in enumerate(
            zip(training, validation, original, strict=True)
        ):
            rng = np.random.default_rng(seed + user_index * 1_009)
            full_rows.append(rng.permutation(pretest).astype(np.int32))
            fixed_rows.append(
                np.concatenate([rng.permutation(train), recent]).astype(np.int32)
            )
        fully_shuffled[int(seed)] = full_rows
        recent_fixed[int(seed)] = fixed_rows
    histories = OrderHistories(
        original=original,
        fully_shuffled=fully_shuffled,
        recent_context_fixed=recent_fixed,
        training=training,
        validation=validation,
    )
    assert_order_invariants(histories)
    return histories


def assert_order_invariants(histories: OrderHistories) -> None:
    """Assert exact item-multiset, user, length, and suffix preservation."""
    user_count = len(histories.original)
    for condition_rows in (
        *histories.fully_shuffled.values(),
        *histories.recent_context_fixed.values(),
    ):
        if len(condition_rows) != user_count:
            raise AssertionError("Order intervention changed the user set")
    for seed, shuffled_rows in histories.fully_shuffled.items():
        for original, shuffled in zip(histories.original, shuffled_rows, strict=True):
            if len(original) != len(shuffled) or not np.array_equal(
                np.sort(original), np.sort(shuffled)
            ):
                raise AssertionError(f"Full shuffle seed {seed} changed an item multiset")
    for seed, shuffled_rows in histories.recent_context_fixed.items():
        for original, shuffled, recent in zip(
            histories.original,
            shuffled_rows,
            histories.validation,
            strict=True,
        ):
            if not np.array_equal(np.sort(original), np.sort(shuffled)):
                raise AssertionError(f"Recent-fixed seed {seed} changed an item multiset")
            if len(recent) and not np.array_equal(shuffled[-len(recent) :], recent):
                raise AssertionError(f"Recent-fixed seed {seed} changed the suffix")


def assert_split_and_candidates(
    split: RevisedSplit,
    validation_sets: dict[int, RankingSet],
    test_sets: dict[int, RankingSet],
) -> None:
    """Assert no leakage and complete, common targets/candidates."""
    train_keys = set(
        zip(
            split.train_positive["UserID"],
            split.train_positive["MovieID"],
            split.train_positive["_source_order"],
        )
    )
    test_keys = set(
        zip(
            split.test_positive["UserID"],
            split.test_positive["MovieID"],
            split.test_positive["_source_order"],
        )
    )
    if train_keys & test_keys:
        raise AssertionError("A test row occurs in training")
    expected_users = set(split.eligible_users["user_id"].astype(int))
    if set(validation_sets) != expected_users or set(test_sets) != expected_users:
        raise AssertionError("Candidate sets do not preserve all eligible users")
    for user_id, ranking_set in test_sets.items():
        expected = set(
            split.test_positive.loc[
                split.test_positive["UserID"].eq(user_id), "MovieID"
            ].astype(int)
        )
        if ranking_set.relevant_movie_ids != expected:
            raise AssertionError(f"Test targets changed for user {user_id}")
        if not expected.issubset(set(ranking_set.candidate_movie_ids.astype(int))):
            raise AssertionError(f"A test target is absent for user {user_id}")


def _condition_sequences(
    histories: OrderHistories, condition: str, shuffle_seed: int | None
) -> list[np.ndarray]:
    if condition == ORIGINAL:
        return histories.original
    if shuffle_seed is None:
        raise ValueError(f"Condition {condition} requires a shuffle seed")
    if condition == FULL_SHUFFLE:
        return histories.fully_shuffled[shuffle_seed]
    if condition == RECENT_FIXED:
        return histories.recent_context_fixed[shuffle_seed]
    raise ValueError(f"Unknown order condition: {condition}")


def _evaluate_non_sequential(
    histories: OrderHistories,
    test_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    config: OrderExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    matrices = {
        ORIGINAL: build_interaction_matrix(histories.original, item_count),
        FULL_SHUFFLE: build_interaction_matrix(
            histories.fully_shuffled[config.sequence_shuffle_seeds[0]], item_count
        ),
        RECENT_FIXED: build_interaction_matrix(
            histories.recent_context_fixed[config.sequence_shuffle_seeds[0]], item_count
        ),
    }
    reference = matrices[ORIGINAL]
    for condition, matrix in matrices.items():
        if (reference != matrix).nnz:
            raise AssertionError(f"Non-sequential input changed under {condition}")

    matrix = reference
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    neighbors = 30 if config.fast_test else 50
    dimensions = 16 if config.fast_test else 32
    similarity = train_item_knn(matrix, neighbors)
    svd = TruncatedSVD(
        n_components=min(dimensions, min(matrix.shape) - 1),
        random_state=config.model_initialization_seeds[0],
    )
    user_embeddings = svd.fit_transform(matrix)
    frames = [
        evaluate_full_ranker(
            "popularity",
            test_sets,
            user_ids,
            item_to_index,
            lambda _user, candidates: popularity[candidates],
        ),
        evaluate_full_ranker(
            "item_knn",
            test_sets,
            user_ids,
            item_to_index,
            lambda user, candidates: similarity[candidates]
            .dot(matrix[user].T)
            .toarray()
            .ravel(),
        ),
        evaluate_full_ranker(
            "matrix_factorization",
            test_sets,
            user_ids,
            item_to_index,
            lambda user, candidates: user_embeddings[user]
            @ svd.components_[:, candidates],
        ),
    ]
    result = pd.concat(frames, ignore_index=True)
    result["order_condition"] = "not_applicable_order_invariant"
    result["model_initialization_seed"] = config.model_initialization_seeds[0]
    result["negative_sampling_seed"] = np.nan
    result["sequence_shuffle_seed"] = np.nan
    return result, {
        "item_knn": {"neighbors": neighbors},
        "matrix_factorization": {"dimensions": dimensions},
    }


def _tune_fpmc(
    histories: OrderHistories,
    validation_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    config: OrderExperimentConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    factors = 16 if config.fast_test else 24
    epochs = 1 if config.fast_test else min(3, config.maximum_epochs)
    init_seed = config.model_initialization_seeds[0]
    model = FPMC(
        len(user_ids),
        item_count,
        factors,
        0.03,
        0.002,
        init_seed,
        config.negative_sampling_seed,
    ).fit(histories.training, epochs)
    previous = np.asarray([sequence[-1] for sequence in histories.training])
    rows: list[dict[str, object]] = []
    for transition_weight in (0.5, 1.0):
        metrics = evaluate_full_ranker(
            "fpmc_validation",
            validation_sets,
            user_ids,
            item_to_index,
            lambda user, candidates, weight=transition_weight: model.score(
                user, int(previous[user]), candidates, weight
            ),
        )
        rows.append(
            {
                "model_name": "fpmc",
                "parameters": json.dumps(
                    {
                        "factors": factors,
                        "epochs": epochs,
                        "transition_weight": transition_weight,
                    }
                ),
                "validation_ndcg_10": float(metrics["ndcg_10"].mean()),
                "model_initialization_seed": init_seed,
                "negative_sampling_seed": config.negative_sampling_seed,
            }
        )
    best = max(rows, key=lambda row: float(row["validation_ndcg_10"]))
    parameters = json.loads(str(best["parameters"]))
    return parameters, rows


def _train_sasrec_controlled(
    sequences: list[np.ndarray],
    validation_sets: dict[int, RankingSet] | None,
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    parameters: SASRecParameters,
    config: OrderExperimentConfig,
    device: torch.device,
    initialization_seed: int,
    run_label: str,
    fixed_epochs: int | None = None,
) -> tuple[SASRec, list[dict[str, object]], int, float]:
    set_deterministic_seed(initialization_seed)
    model = SASRec(item_count, parameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
    maximum_epochs = fixed_epochs or (
        min(2, config.maximum_epochs) if config.fast_test else config.maximum_epochs
    )
    best_epoch = maximum_epochs
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rows: list[dict[str, object]] = []
    for epoch in range(1, maximum_epochs + 1):
        epoch_started = time.perf_counter()
        inputs, positives, negatives = training_arrays(
            sequences,
            parameters.max_sequence_length,
            item_count,
            config.negative_sampling_seed + epoch,
        )
        order = np.random.default_rng(initialization_seed + epoch).permutation(
            len(sequences)
        )
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            inputs_batch = torch.as_tensor(inputs[indices], device=device)
            positives_batch = torch.as_tensor(positives[indices], device=device)
            negatives_batch = torch.as_tensor(negatives[indices], device=device)
            mask = positives_batch.gt(0)
            hidden = model(inputs_batch)
            positive_logits = (
                hidden * model.item_embedding(positives_batch)
            ).sum(dim=-1)
            negative_logits = (
                hidden * model.item_embedding(negatives_batch)
            ).sum(dim=-1)
            loss = (
                loss_function(positive_logits, torch.ones_like(positive_logits))
                + loss_function(negative_logits, torch.zeros_like(negative_logits))
            )
            loss = (loss * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_ndcg = np.nan
        if validation_sets is not None:
            scores = sasrec_score_matrix(model, sequences, config.batch_size, device)
            validation_metrics = evaluate_full_ranker(
                "sasrec_validation",
                validation_sets,
                user_ids,
                item_to_index,
                lambda user, candidates: scores[user, candidates],
            )
            validation_ndcg = float(validation_metrics["ndcg_10"].mean())
            if validation_ndcg > best_score + 1e-5:
                best_score = validation_ndcg
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        rows.append(
            {
                "run_label": run_label,
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_ndcg_10": validation_ndcg,
                "model_initialization_seed": initialization_seed,
                "negative_sampling_seed": config.negative_sampling_seed,
                "epoch_runtime_seconds": time.perf_counter() - epoch_started,
            }
        )
        if validation_sets is not None and stale >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows, best_epoch, best_score


def _tune_sasrec(
    histories: OrderHistories,
    validation_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    config: OrderExperimentConfig,
    device: torch.device,
) -> tuple[SASRecParameters, int, list[dict[str, object]], list[dict[str, object]]]:
    options = [
        SASRecParameters(32, 50, 1, 2, 0.1, 0.001),
        SASRecParameters(64, 100, 2, 4, 0.2, 0.0005),
    ]
    if config.fast_test:
        options = options[:1]
    validation_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    best: tuple[float, SASRecParameters, int] | None = None
    for index, parameters in enumerate(options):
        model, rows, epoch, score = _train_sasrec_controlled(
            histories.training,
            validation_sets,
            user_ids,
            item_to_index,
            item_count,
            parameters,
            config,
            device,
            config.model_initialization_seeds[0],
            f"sasrec_tuning_{index}",
        )
        training_rows.extend(rows)
        validation_rows.append(
            {
                "model_name": "sasrec",
                "parameters": json.dumps(asdict(parameters)),
                "best_epoch": epoch,
                "validation_ndcg_10": score,
                "model_initialization_seed": config.model_initialization_seeds[0],
                "negative_sampling_seed": config.negative_sampling_seed,
            }
        )
        candidate = (score, parameters, epoch)
        if best is None or candidate[0] > best[0]:
            best = candidate
        del model
    if best is None:
        raise RuntimeError("SASRec tuning produced no candidate")
    return best[1], best[2], validation_rows, training_rows


def _evaluate_fpmc_conditions(
    histories: OrderHistories,
    test_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    parameters: dict[str, object],
    config: OrderExperimentConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    conditions = [(ORIGINAL, None)] + [
        (condition, seed)
        for condition in (FULL_SHUFFLE, RECENT_FIXED)
        for seed in config.sequence_shuffle_seeds
    ]
    for init_seed in config.model_initialization_seeds:
        for condition, shuffle_seed in conditions:
            sequences = _condition_sequences(histories, condition, shuffle_seed)
            model = FPMC(
                len(user_ids),
                item_count,
                int(parameters["factors"]),
                0.03,
                0.002,
                init_seed,
                config.negative_sampling_seed,
            ).fit(sequences, int(parameters["epochs"]))
            previous = np.asarray([sequence[-1] for sequence in sequences])
            metrics = evaluate_full_ranker(
                f"fpmc_{condition}",
                test_sets,
                user_ids,
                item_to_index,
                lambda user, candidates: model.score(
                    user,
                    int(previous[user]),
                    candidates,
                    float(parameters["transition_weight"]),
                ),
            )
            metrics["model_name"] = "fpmc"
            metrics["order_condition"] = condition
            metrics["model_initialization_seed"] = init_seed
            metrics["negative_sampling_seed"] = config.negative_sampling_seed
            metrics["sequence_shuffle_seed"] = shuffle_seed
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def _evaluate_sasrec_conditions(
    histories: OrderHistories,
    test_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    parameters: SASRecParameters,
    epochs: int,
    config: OrderExperimentConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    conditions = [(ORIGINAL, None)] + [
        (condition, seed)
        for condition in (FULL_SHUFFLE, RECENT_FIXED)
        for seed in config.sequence_shuffle_seeds
    ]
    for init_seed in config.model_initialization_seeds:
        for condition, shuffle_seed in conditions:
            sequences = _condition_sequences(histories, condition, shuffle_seed)
            label = f"sasrec_{condition}_init_{init_seed}_shuffle_{shuffle_seed}"
            model, rows, _, _ = _train_sasrec_controlled(
                sequences,
                None,
                user_ids,
                item_to_index,
                item_count,
                parameters,
                config,
                device,
                init_seed,
                label,
                fixed_epochs=epochs,
            )
            training_rows.extend(rows)
            scores = sasrec_score_matrix(model, sequences, config.batch_size, device)
            metrics = evaluate_full_ranker(
                f"sasrec_{condition}",
                test_sets,
                user_ids,
                item_to_index,
                lambda user, candidates: scores[user, candidates],
            )
            metrics["model_name"] = "sasrec"
            metrics["order_condition"] = condition
            metrics["model_initialization_seed"] = init_seed
            metrics["negative_sampling_seed"] = config.negative_sampling_seed
            metrics["sequence_shuffle_seed"] = shuffle_seed
            metric_rows.append(metrics)
            del model, scores
    return pd.concat(metric_rows, ignore_index=True), pd.DataFrame(training_rows)


def _bootstrap_interval(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate_model_metrics(
    per_user: pd.DataFrame, config: OrderExperimentConfig
) -> pd.DataFrame:
    """Aggregate metrics after averaging repeated runs within each user."""
    grouped = (
        per_user.groupby(["user_id", "model_name", "order_condition"], dropna=False)[
            list(METRICS)
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for keys, frame in grouped.groupby(["model_name", "order_condition"]):
        model_name, condition = keys
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            lower, upper = _bootstrap_interval(
                values,
                min(100, config.bootstrap_samples)
                if config.fast_test
                else config.bootstrap_samples,
                config.bootstrap_seed + len(rows),
            )
            rows.append(
                {
                    "model_name": model_name,
                    "order_condition": condition,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "user_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_order_results(
    per_user: pd.DataFrame, config: OrderExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate primary and secondary paired order effects by seed and overall."""
    sequential = per_user.loc[per_user["model_name"].isin(SEQUENTIAL_MODELS)].copy()
    originals = sequential.loc[sequential["order_condition"].eq(ORIGINAL), [
        "user_id", "model_name", "model_initialization_seed", "ndcg_10"
    ]].rename(columns={"ndcg_10": "original_ndcg_10"})
    comparisons = sequential.loc[
        sequential["order_condition"].isin([FULL_SHUFFLE, RECENT_FIXED])
    ].merge(
        originals,
        on=["user_id", "model_name", "model_initialization_seed"],
        validate="many_to_one",
    )
    comparisons = comparisons.rename(columns={"ndcg_10": "intervention_ndcg_10"})
    comparisons["paired_difference_ndcg_10"] = (
        comparisons["original_ndcg_10"] - comparisons["intervention_ndcg_10"]
    )
    seed_rows: list[dict[str, object]] = []
    for keys, frame in comparisons.groupby(
        ["model_name", "order_condition", "model_initialization_seed", "sequence_shuffle_seed"]
    ):
        model_name, condition, init_seed, shuffle_seed = keys
        values = frame["paired_difference_ndcg_10"].to_numpy(float)
        lower, upper = _bootstrap_interval(
            values,
            min(100, config.bootstrap_samples)
            if config.fast_test
            else config.bootstrap_samples,
            config.bootstrap_seed + len(seed_rows),
        )
        nonzero = values[values != 0]
        test = wilcoxon(nonzero) if len(nonzero) else None
        seed_rows.append(
            {
                "model_name": model_name,
                "intervention": condition,
                "model_initialization_seed": int(init_seed),
                "sequence_shuffle_seed": int(shuffle_seed),
                "mean_paired_difference": float(values.mean()),
                "median_paired_difference": float(np.median(values)),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "standardized_effect": float(values.mean() / values.std(ddof=1))
                if values.std(ddof=1) > 0
                else 0.0,
                "positive_percentage": float((values > 0).mean() * 100),
                "zero_percentage": float((values == 0).mean() * 100),
                "negative_percentage": float((values < 0).mean() * 100),
                "wilcoxon_statistic": float(test.statistic) if test else np.nan,
                "wilcoxon_p_value": float(test.pvalue) if test else np.nan,
                "user_count": len(values),
            }
        )
    seed_results = pd.DataFrame(seed_rows)
    per_user_mean = (
        comparisons.groupby(["user_id", "model_name", "order_condition"])[
            ["original_ndcg_10", "intervention_ndcg_10"]
        ]
        .mean()
        .reset_index()
    )
    per_user_mean["paired_difference_ndcg_10"] = (
        per_user_mean["original_ndcg_10"]
        - per_user_mean["intervention_ndcg_10"]
    )
    overall_rows: list[dict[str, object]] = []
    for keys, frame in per_user_mean.groupby(["model_name", "order_condition"]):
        values = frame["paired_difference_ndcg_10"].to_numpy(float)
        lower, upper = _bootstrap_interval(
            values,
            min(100, config.bootstrap_samples)
            if config.fast_test
            else config.bootstrap_samples,
            config.bootstrap_seed + 10_000 + len(overall_rows),
        )
        nonzero = values[values != 0]
        test = wilcoxon(nonzero) if len(nonzero) else None
        overall_rows.append(
            {
                "model_name": keys[0],
                "intervention": keys[1],
                "comparison_level": "per_user_mean_across_configured_seeds",
                "mean_paired_difference": float(values.mean()),
                "median_paired_difference": float(np.median(values)),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "standardized_effect": float(values.mean() / values.std(ddof=1))
                if values.std(ddof=1) > 0
                else 0.0,
                "positive_percentage": float((values > 0).mean() * 100),
                "zero_percentage": float((values == 0).mean() * 100),
                "negative_percentage": float((values < 0).mean() * 100),
                "wilcoxon_statistic": float(test.statistic) if test else np.nan,
                "wilcoxon_p_value": float(test.pvalue) if test else np.nan,
                "user_count": len(values),
            }
        )
    return comparisons, seed_results, pd.DataFrame(overall_rows)


def _pretest_sensitivity_features(
    split: RevisedSplit, histories: OrderHistories, user_ids: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full = pd.concat([split.train_positive, split.validation_positive]).sort_values(
        ["UserID", "_user_order_rank"], kind="mergesort"
    )
    grouped = {int(user): frame for user, frame in full.groupby("UserID")}
    for index, user_id in enumerate(user_ids):
        frame = grouped[int(user_id)]
        timestamp_counts = frame["Timestamp"].value_counts()
        rows.append(
            {
                "user_id": int(user_id),
                "pretest_history_length": len(histories.original[index]),
                "history_span_days": (
                    frame["Timestamp"].max() - frame["Timestamp"].min()
                )
                / 86_400,
                "largest_equal_timestamp_ratio": float(timestamp_counts.max() / len(frame)),
                "equal_timestamp_interaction_ratio": float(
                    frame["Timestamp"].duplicated(keep=False).mean()
                ),
            }
        )
    features = pd.DataFrame(rows)
    def rank_tertiles(series: pd.Series, labels: list[str]) -> pd.Series:
        if series.nunique() < 2:
            return pd.Series(labels[1], index=series.index, dtype="category")
        return pd.cut(
            series.rank(method="average", pct=True),
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=labels,
            include_lowest=True,
        )

    features["history_length_group"] = rank_tertiles(
        features["pretest_history_length"], ["short", "medium", "long"]
    )
    features["temporal_span_group"] = rank_tertiles(
        features["history_span_days"], ["short", "medium", "long"]
    )
    features["timestamp_concentration_group"] = rank_tertiles(
        features["largest_equal_timestamp_ratio"], ["low", "medium", "high"]
    )
    features["has_equal_timestamp_events"] = np.where(
        features["equal_timestamp_interaction_ratio"].gt(0), "yes", "no"
    )
    return features


def _sensitivity_results(
    paired: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    merged = paired.merge(features, on="user_id", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for group_variable in (
        "history_length_group",
        "temporal_span_group",
        "timestamp_concentration_group",
        "has_equal_timestamp_events",
    ):
        for keys, frame in merged.groupby(
            ["model_name", "order_condition", group_variable], observed=True
        ):
            rows.append(
                {
                    "model_name": keys[0],
                    "intervention": keys[1],
                    "group_variable": group_variable,
                    "group_value": str(keys[2]),
                    "mean_paired_difference": float(
                        frame["paired_difference_ndcg_10"].mean()
                    ),
                    "median_paired_difference": float(
                        frame["paired_difference_ndcg_10"].median()
                    ),
                    "user_count": int(frame["user_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _minimum_history_sensitivity(
    ratings: pd.DataFrame, config: OrderExperimentConfig
) -> pd.DataFrame:
    threshold = -np.inf if config.relevance_threshold is None else config.relevance_threshold
    counts = ratings.loc[ratings["Rating"].ge(threshold)].groupby("UserID").size()
    rows = []
    for minimum in sorted({5, 10, 20, config.minimum_training_interactions}):
        validation = np.maximum(
            config.minimum_validation_targets,
            np.floor(counts.to_numpy() * config.holdout_fraction).astype(int),
        )
        test = np.maximum(
            config.minimum_test_targets,
            np.floor(counts.to_numpy() * config.holdout_fraction).astype(int),
        )
        eligible = counts.to_numpy() - validation - test >= minimum
        rows.append(
            {
                "minimum_training_interactions": minimum,
                "eligible_users": int(eligible.sum()),
                "excluded_users": int((~eligible).sum()),
                "note": "eligibility-only sensitivity; models not refit",
            }
        )
    return pd.DataFrame(rows)


def _dataset_metadata(
    data_dir: Path,
    ratings: pd.DataFrame,
    checksum_local_file: bool = True,
) -> dict[str, object]:
    ratings_path = data_dir / "ratings.dat"
    checksum = None
    if checksum_local_file and ratings_path.exists():
        digest = hashlib.sha256()
        with ratings_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
    return {
        "dataset": "MovieLens 1M",
        "ratings_path": str(ratings_path),
        "ratings_sha256": checksum,
        "rating_rows": len(ratings),
        "users": int(ratings["UserID"].nunique()),
        "movies_in_ratings": int(ratings["MovieID"].nunique()),
    }


def _git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _environment_metadata(device: torch.device) -> dict[str, object]:
    packages = {}
    for name in (
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "psutil",
        "torch",
        "pytest",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gib": psutil.virtual_memory().total / 1024**3,
        "cpu_model": platform.uname().processor or platform.processor() or None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": _git_metadata(),
    }


def _write_plots(
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    statistical: pd.DataFrame,
    output_dir: Path,
) -> None:
    plots = output_dir / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    ndcg = aggregate.loc[aggregate["metric"].eq("ndcg_10")]
    labels = ndcg["model_name"] + "\n" + ndcg["order_condition"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, ndcg["mean"])
    ax.set_ylabel("Mean NDCG@10")
    ax.set_title("Full-Ranking Performance by Order Condition")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots / "aggregate_ndcg.png", dpi=150)
    plt.close(fig)

    primary = paired.loc[paired["order_condition"].eq(FULL_SHUFFLE)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, model_name in zip(axes, SEQUENTIAL_MODELS, strict=True):
        axis.hist(
            primary.loc[primary["model_name"].eq(model_name), "paired_difference_ndcg_10"],
            bins=35,
        )
        axis.axvline(0, color="black", linestyle="--")
        axis.set_title(model_name.upper())
        axis.set_xlabel("Original − fully shuffled NDCG@10")
    axes[0].set_ylabel("User-seed comparisons")
    fig.tight_layout()
    fig.savefig(plots / "primary_paired_differences.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    primary_stats = statistical.loc[statistical["intervention"].eq(FULL_SHUFFLE)]
    ax.bar(primary_stats["model_name"], primary_stats["mean_paired_difference"])
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Mean paired NDCG@10 difference")
    ax.set_title("Primary Recorded-Order Effect")
    fig.tight_layout()
    fig.savefig(plots / "primary_effect_summary.png", dpi=150)
    plt.close(fig)


def _create_report(
    config: OrderExperimentConfig,
    aggregate: pd.DataFrame,
    statistical: pd.DataFrame,
    sensitivity: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    def metric(model: str, condition: str) -> float:
        row = aggregate.loc[
            aggregate["model_name"].eq(model)
            & aggregate["order_condition"].eq(condition)
            & aggregate["metric"].eq("ndcg_10")
        ]
        return float(row.iloc[0]["mean"])

    lines = [
        "# Controlled MovieLens 1M Order Experiment",
        "",
        f"**Run status:** {'NON-FINAL SMOKE TEST' if config.fast_test else 'FORMAL EXPERIMENT'}",
        "",
        "## Confirmatory primary results",
        "",
        "The primary outcome is paired user-level original-order NDCG@10 minus fully shuffled pre-test NDCG@10. Positive values favor recorded rating-submission order.",
        "",
    ]
    for model in SEQUENTIAL_MODELS:
        result = statistical.loc[
            statistical["model_name"].eq(model)
            & statistical["intervention"].eq(FULL_SHUFFLE)
        ].iloc[0]
        lines.append(
            f"- **{model.upper()}:** original {metric(model, ORIGINAL):.4f}; fully shuffled {metric(model, FULL_SHUFFLE):.4f}; mean paired difference {result['mean_paired_difference']:+.4f} (95% CI {result['ci_95_lower']:+.4f} to {result['ci_95_upper']:+.4f}); standardized effect {result['standardized_effect']:+.3f}; Wilcoxon p={result['wilcoxon_p_value']:.3g}."
        )
    lines.extend(
        [
            "",
            "A small p-value is not, by itself, evidence of a practically important order effect.",
            "",
            "## Secondary recent-context-fixed ablation",
            "",
        ]
    )
    for model in SEQUENTIAL_MODELS:
        result = statistical.loc[
            statistical["model_name"].eq(model)
            & statistical["intervention"].eq(RECENT_FIXED)
        ].iloc[0]
        lines.append(
            f"- **{model.upper()}:** mean original-minus-ablation difference {result['mean_paired_difference']:+.4f} (95% CI {result['ci_95_lower']:+.4f} to {result['ci_95_upper']:+.4f})."
        )
    lines.extend(
        [
            "",
            "## Empirical findings",
            "",
            f"Non-sequential controls were fit once on order-invariant user-item data. The run used {len(config.sequence_shuffle_seeds)} shuffle seed(s), {len(config.model_initialization_seeds)} model initialization seed(s), and full ranking over unseen catalog items.",
            "",
            "## Exploratory subgroup findings",
            "",
            f"Subgroup results are available in `sensitivity_group_results.csv` ({len(sensitivity)} rows). They are exploratory and were not used to redefine the primary outcome.",
            "",
            "## Interpretation",
            "",
            "These results estimate sensitivity to the recorded ordering of positive rating events. They do not establish effects of true movie-viewing order.",
            "",
            "## Limitations",
            "",
            "- Timestamps record rating submission, not confirmed viewing.",
            "- Equal timestamps are ordered deterministically by MovieID, then source row.",
            "- Shuffling destroys order but not unobserved confounding in when and why ratings were submitted.",
            "- Sensitivity groups and minimum-history summaries are exploratory.",
            f"- Runtime was {runtime_seconds:.1f} seconds on the recorded environment.",
            "",
            "No automatic selector recommendation is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def _validate_result_rows(
    per_user: pd.DataFrame, eligible_users: pd.DataFrame, config: OrderExperimentConfig
) -> None:
    expected_users = set(eligible_users["user_id"].astype(int))
    if set(per_user["user_id"].astype(int)) != expected_users:
        raise AssertionError("Model outputs do not preserve all eligible users")
    key_columns = [
        "user_id",
        "model_name",
        "order_condition",
        "model_initialization_seed",
        "sequence_shuffle_seed",
    ]
    if per_user.duplicated(key_columns).any():
        raise AssertionError("A model-condition-seed has duplicate user rows")
    if per_user[list(METRICS)].isna().any().any():
        raise AssertionError("A model metric is missing")
    expected_sequential_runs = len(config.model_initialization_seeds) * (
        1 + 2 * len(config.sequence_shuffle_seeds)
    )
    for model in SEQUENTIAL_MODELS:
        rows = per_user.loc[per_user["model_name"].eq(model)]
        if len(rows) != len(expected_users) * expected_sequential_runs:
            raise AssertionError(f"Incomplete result grid for {model}")


def run_experiment(
    config: OrderExperimentConfig,
    ratings: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the controlled order experiment and write self-contained artifacts."""
    started = time.perf_counter()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    ratings_frame = load_ml1m_ratings(config.data_dir) if ratings is None else ratings.copy()
    movies_frame = load_ml1m_movies(config.data_dir) if movies is None else movies.copy()

    split = revised_chronological_split(ratings_frame, _revised_config(config))
    if split.eligible_users.empty:
        raise ValueError("No users satisfy the configured split requirements")
    movie_ids = np.sort(movies_frame["MovieID"].unique()).astype(np.int64)
    item_to_index = {int(item): index for index, item in enumerate(movie_ids)}
    user_ids = split.eligible_users["user_id"].to_numpy(np.int64)
    validation_sets, validation_candidates = build_ranking_sets(
        split, split.validation_positive, movie_ids
    )
    test_sets, test_candidates = build_ranking_sets(
        split, split.test_positive, movie_ids
    )
    assert_split_and_candidates(split, validation_sets, test_sets)
    histories = build_order_histories(
        split, user_ids, item_to_index, config.sequence_shuffle_seeds
    )

    nonseq, nonseq_parameters = _evaluate_non_sequential(
        histories, test_sets, user_ids, item_to_index, len(movie_ids), config
    )
    fpmc_parameters, fpmc_validation = _tune_fpmc(
        histories,
        validation_sets,
        user_ids,
        item_to_index,
        len(movie_ids),
        config,
    )
    sasrec_parameters, sasrec_epochs, sasrec_validation, tuning_curves = _tune_sasrec(
        histories,
        validation_sets,
        user_ids,
        item_to_index,
        len(movie_ids),
        config,
        device,
    )
    fpmc = _evaluate_fpmc_conditions(
        histories,
        test_sets,
        user_ids,
        item_to_index,
        len(movie_ids),
        fpmc_parameters,
        config,
    )
    sasrec, final_curves = _evaluate_sasrec_conditions(
        histories,
        test_sets,
        user_ids,
        item_to_index,
        len(movie_ids),
        sasrec_parameters,
        sasrec_epochs,
        config,
        device,
    )
    per_user = pd.concat([nonseq, fpmc, sasrec], ignore_index=True)
    _validate_result_rows(per_user, split.eligible_users, config)
    aggregate = aggregate_model_metrics(per_user, config)
    paired, seed_results, statistics = paired_order_results(per_user, config)
    features = _pretest_sensitivity_features(split, histories, user_ids)
    sensitivity = _sensitivity_results(paired, features)
    minimum_history = _minimum_history_sensitivity(ratings_frame, config)
    validation_results = pd.DataFrame(fpmc_validation + sasrec_validation)
    training_curves = pd.concat(
        [pd.DataFrame(tuning_curves), final_curves], ignore_index=True
    )
    selected = {
        **nonseq_parameters,
        "fpmc": fpmc_parameters,
        "sasrec": {**asdict(sasrec_parameters), "epochs": sasrec_epochs},
        "selection_protocol": (
            "Selected on original-order validation only, then held fixed across all "
            "test order conditions to isolate order effects"
        ),
    }

    split_audit = split.split_table.copy()
    split_audit["is_test_untouched"] = split_audit["split"].eq("test")
    condition_audit_rows = []
    for condition, seed, sequences in [
        (ORIGINAL, None, histories.original),
        *[
            (FULL_SHUFFLE, seed, histories.fully_shuffled[seed])
            for seed in config.sequence_shuffle_seeds
        ],
        *[
            (RECENT_FIXED, seed, histories.recent_context_fixed[seed])
            for seed in config.sequence_shuffle_seeds
        ],
    ]:
        for user_id, original, sequence, recent in zip(
            user_ids, histories.original, sequences, histories.validation, strict=True
        ):
            condition_audit_rows.append(
                {
                    "user_id": int(user_id),
                    "order_condition": condition,
                    "sequence_shuffle_seed": seed,
                    "sequence_length": len(sequence),
                    "item_multiset_preserved": bool(
                        np.array_equal(np.sort(original), np.sort(sequence))
                    ),
                    "recent_context_preserved": bool(
                        condition != RECENT_FIXED
                        or np.array_equal(sequence[-len(recent) :], recent)
                    ),
                }
            )
    condition_audit = pd.DataFrame(condition_audit_rows)
    non_sequential_audit = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "comparison": "original_vs_fully_shuffled_input_multiset",
                "maximum_absolute_difference": 0.0,
                "numerical_tolerance": 0.0,
                "invariant": True,
            }
            for model_name in NON_SEQUENTIAL_MODELS
        ]
    )

    files: dict[str, pd.DataFrame] = {
        "eligible_users.csv": split.eligible_users,
        "excluded_users.csv": split.excluded_users,
        "split_summary.csv": split.summary,
        "split_audit_table.csv": split_audit,
        "validation_candidates.csv": validation_candidates,
        "test_candidates.csv": test_candidates,
        "order_condition_audit.csv": condition_audit,
        "non_sequential_invariance_audit.csv": non_sequential_audit,
        "validation_results.csv": validation_results,
        "per_user_model_metrics.csv": per_user,
        "aggregate_model_metrics.csv": aggregate,
        "paired_order_differences.csv": paired,
        "seed_level_results.csv": seed_results,
        "statistical_comparisons.csv": statistics,
        "training_curves.csv": training_curves,
        "pretest_sensitivity_features.csv": features,
        "sensitivity_group_results.csv": sensitivity,
        "minimum_history_sensitivity.csv": minimum_history,
    }
    for filename, frame in files.items():
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config.serializable(), indent=2), encoding="utf-8"
    )
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(
            _dataset_metadata(
                config.data_dir,
                ratings_frame,
                checksum_local_file=ratings is None,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_plots(aggregate, paired, statistics, output_dir)
    runtime = time.perf_counter() - started
    (output_dir / "report.md").write_text(
        _create_report(config, aggregate, statistics, sensitivity, runtime),
        encoding="utf-8",
    )
    environment = _environment_metadata(device)
    environment["random_seeds"] = {
        "model_initialization": list(config.model_initialization_seeds),
        "negative_sampling": config.negative_sampling_seed,
        "sequence_shuffle": list(config.sequence_shuffle_seeds),
        "bootstrap": config.bootstrap_seed,
        "split_and_fast_user_sampling": config.split_seed,
    }
    environment["runtime_seconds"] = time.perf_counter() - started
    (output_dir / "environment_and_hardware.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return files


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("At least one integer seed is required")
    return parsed


def parse_args() -> OrderExperimentConfig:
    """Parse command-line configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ML1M_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--all-ratings", action="store_true")
    parser.add_argument("--minimum-training-interactions", type=int, default=5)
    parser.add_argument("--minimum-validation-targets", type=int, default=2)
    parser.add_argument("--minimum-test-targets", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument(
        "--model-initialization-seeds", type=_parse_int_tuple, default=(760,)
    )
    parser.add_argument("--negative-sampling-seed", type=int, default=17_760)
    parser.add_argument(
        "--sequence-shuffle-seeds",
        type=_parse_int_tuple,
        default=(20_760, 20_761, 20_762),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=30_760)
    parser.add_argument("--split-seed", type=int, default=40_760)
    parser.add_argument("--maximum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--fast-test", action="store_true")
    parser.add_argument("--fast-user-limit", type=int, default=120)
    parser.add_argument(
        "--equal-timestamp-ordering-rule",
        choices=("timestamp_movie_id_source", "timestamp_source"),
        default="timestamp_movie_id_source",
    )
    arguments = parser.parse_args()
    return OrderExperimentConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        relevance_threshold=(
            None if arguments.all_ratings else arguments.relevance_threshold
        ),
        minimum_training_interactions=arguments.minimum_training_interactions,
        minimum_validation_targets=arguments.minimum_validation_targets,
        minimum_test_targets=arguments.minimum_test_targets,
        holdout_fraction=arguments.holdout_fraction,
        model_initialization_seeds=arguments.model_initialization_seeds,
        negative_sampling_seed=arguments.negative_sampling_seed,
        sequence_shuffle_seeds=arguments.sequence_shuffle_seeds,
        bootstrap_seed=arguments.bootstrap_seed,
        split_seed=arguments.split_seed,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        bootstrap_samples=arguments.bootstrap_samples,
        fast_test=arguments.fast_test,
        fast_user_limit=arguments.fast_user_limit,
        equal_timestamp_ordering_rule=arguments.equal_timestamp_ordering_rule,
    )


def main() -> None:
    """Run the command-line experiment."""
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
