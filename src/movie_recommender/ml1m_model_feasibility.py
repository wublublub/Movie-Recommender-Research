"""User-level model feasibility experiment for MovieLens 1M."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "movie_recommender_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from movie_recommender.ml1m_data_loader import (
    DEFAULT_ML1M_DATA_DIR,
    load_ml1m_movies,
    load_ml1m_ratings,
)
from movie_recommender.ml1m_temporal_eda import (
    build_genre_evolution_features,
    build_user_temporal_features,
)


DEFAULT_OUTPUT_DIR: Final = (
    Path(__file__).resolve().parents[2] / "outputs" / "ml1m_model_feasibility"
)
CORE_MODELS: Final = ("popularity", "item_knn", "matrix_factorization", "fpmc_original")
NON_SEQUENTIAL_MODELS: Final = ("popularity", "item_knn", "matrix_factorization")
ASSOCIATION_FEATURES: Final = (
    "rating_count",
    "history_span_days",
    "median_gap_minutes",
    "largest_day_ratio",
    "largest_session_ratio_2h",
    "genre_js_divergence",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the reproducible feasibility experiment."""

    data_dir: Path = DEFAULT_ML1M_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    relevance_threshold: float = 4.0
    minimum_training_interactions: int = 5
    negative_samples: int = 99
    random_seed: int = 760
    selected_sequential_model: str = "fpmc"
    maximum_epochs: int = 3
    shuffle_runs: int = 3
    bootstrap_samples: int = 500
    fast_test: bool = False
    fast_user_limit: int = 400

    def serializable(self) -> dict[str, object]:
        """Return JSON-safe configuration values."""
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        values["relevance_rule"] = f"Rating >= {self.relevance_threshold:g}"
        values["candidate_strategy"] = (
            f"1 held-out positive plus {self.negative_samples} deterministic unseen negatives"
        )
        values["sequential_model_description"] = (
            "FPMC trained with pairwise BPR updates over rating-submission order"
        )
        return values


@dataclass
class SplitData:
    """Chronological positive-interaction splits and audit tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    eligible_users: pd.DataFrame
    excluded_users: pd.DataFrame
    summary: pd.DataFrame
    full_ratings: pd.DataFrame


@dataclass
class CandidateSet:
    """A held-out target and its common evaluation candidates."""

    user_id: int
    target_movie_id: int
    target_timestamp: int
    candidate_movie_ids: np.ndarray


class FPMC:
    """Factorizing Personalized Markov Chains with pairwise BPR training."""

    def __init__(
        self,
        user_count: int,
        item_count: int,
        factors: int,
        learning_rate: float,
        regularization: float,
        seed: int,
        sampling_seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.05
        self.user_factors = rng.normal(0, scale, (user_count, factors)).astype(np.float32)
        self.item_preference_factors = rng.normal(
            0, scale, (item_count, factors)
        ).astype(np.float32)
        self.previous_item_factors = rng.normal(
            0, scale, (item_count, factors)
        ).astype(np.float32)
        self.next_item_factors = rng.normal(0, scale, (item_count, factors)).astype(
            np.float32
        )
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        self.sampling_seed = seed if sampling_seed is None else sampling_seed

    def fit(self, sequences: list[np.ndarray], epochs: int) -> FPMC:
        """Fit preference and first-order transition factors."""
        transition_users: list[int] = []
        transition_previous: list[int] = []
        transition_next: list[int] = []
        seen_items: list[set[int]] = []
        for user_index, sequence in enumerate(sequences):
            seen_items.append(set(int(item) for item in sequence))
            if len(sequence) < 2:
                continue
            transition_users.extend([user_index] * (len(sequence) - 1))
            transition_previous.extend(sequence[:-1].tolist())
            transition_next.extend(sequence[1:].tolist())

        users = np.asarray(transition_users, dtype=np.int32)
        previous_items = np.asarray(transition_previous, dtype=np.int32)
        next_items = np.asarray(transition_next, dtype=np.int32)
        rng = np.random.default_rng(self.sampling_seed)
        item_count = self.item_preference_factors.shape[0]

        for _ in range(epochs):
            for position in rng.permutation(len(users)):
                user_index = int(users[position])
                previous_item = int(previous_items[position])
                positive_item = int(next_items[position])
                negative_item = int(rng.integers(item_count))
                while negative_item in seen_items[user_index]:
                    negative_item = int(rng.integers(item_count))

                user_vector = self.user_factors[user_index].copy()
                positive_preference = self.item_preference_factors[positive_item].copy()
                negative_preference = self.item_preference_factors[negative_item].copy()
                previous_vector = self.previous_item_factors[previous_item].copy()
                positive_next = self.next_item_factors[positive_item].copy()
                negative_next = self.next_item_factors[negative_item].copy()

                score_difference = float(
                    user_vector @ (positive_preference - negative_preference)
                    + previous_vector @ (positive_next - negative_next)
                )
                gradient = 1.0 / (1.0 + np.exp(np.clip(score_difference, -30, 30)))
                rate = self.learning_rate
                regularization = self.regularization

                self.user_factors[user_index] += rate * (
                    gradient * (positive_preference - negative_preference)
                    - regularization * user_vector
                )
                self.item_preference_factors[positive_item] += rate * (
                    gradient * user_vector - regularization * positive_preference
                )
                self.item_preference_factors[negative_item] += rate * (
                    -gradient * user_vector - regularization * negative_preference
                )
                self.previous_item_factors[previous_item] += rate * (
                    gradient * (positive_next - negative_next)
                    - regularization * previous_vector
                )
                self.next_item_factors[positive_item] += rate * (
                    gradient * previous_vector - regularization * positive_next
                )
                self.next_item_factors[negative_item] += rate * (
                    -gradient * previous_vector - regularization * negative_next
                )
        return self

    def score(
        self,
        user_index: int,
        previous_item: int,
        candidates: np.ndarray,
        transition_weight: float,
    ) -> np.ndarray:
        """Score candidates using long-term preference and short-term transition."""
        preference = (
            self.item_preference_factors[candidates] @ self.user_factors[user_index]
        )
        transition = (
            self.next_item_factors[candidates] @ self.previous_item_factors[previous_item]
        )
        return preference + transition_weight * transition


def chronological_split(
    ratings: pd.DataFrame,
    config: ExperimentConfig,
) -> SplitData:
    """Create a stable leave-last-two-out split over positive interactions."""
    required = {"UserID", "MovieID", "Rating", "Timestamp"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")

    full = ratings.reset_index(names="_source_order").sort_values(
        ["UserID", "Timestamp", "_source_order"], kind="mergesort"
    )
    positives = full.loc[full["Rating"].ge(config.relevance_threshold)].copy()
    positive_counts = positives.groupby("UserID").size()
    total_counts = full.groupby("UserID").size()
    required_positive_count = config.minimum_training_interactions + 2
    all_users = np.sort(full["UserID"].unique())
    eligible_mask = positive_counts.reindex(all_users, fill_value=0).ge(
        required_positive_count
    )
    eligible_user_ids = all_users[eligible_mask.to_numpy()]

    exclusion_rows = []
    for user_id in all_users[~eligible_mask.to_numpy()]:
        positive_count = int(positive_counts.get(user_id, 0))
        exclusion_rows.append(
            {
                "user_id": int(user_id),
                "total_rating_count": int(total_counts[user_id]),
                "positive_interaction_count": positive_count,
                "exclusion_reason": (
                    f"fewer_than_{required_positive_count}_positive_interactions"
                ),
            }
        )

    selected_user_ids = eligible_user_ids
    if config.fast_test and len(selected_user_ids) > config.fast_user_limit:
        rng = np.random.default_rng(config.random_seed)
        selected_user_ids = np.sort(
            rng.choice(selected_user_ids, config.fast_user_limit, replace=False)
        )

    selected = positives.loc[positives["UserID"].isin(selected_user_ids)].copy()
    selected["positive_position"] = selected.groupby("UserID").cumcount()
    selected["positive_count"] = selected.groupby("UserID")["MovieID"].transform("size")
    train = selected.loc[
        selected["positive_position"] < selected["positive_count"] - 2
    ].copy()
    validation = selected.loc[
        selected["positive_position"] == selected["positive_count"] - 2
    ].copy()
    test = selected.loc[
        selected["positive_position"] == selected["positive_count"] - 1
    ].copy()

    eligible_users = (
        selected.groupby("UserID")
        .size()
        .rename("positive_interaction_count")
        .reset_index()
        .rename(columns={"UserID": "user_id"})
    )
    eligible_users["training_interaction_count"] = (
        eligible_users["positive_interaction_count"] - 2
    )
    eligible_users["selected_for_run"] = True

    summary = pd.DataFrame(
        [
            {"metric": "total_users", "value": len(all_users)},
            {"metric": "eligible_users_full_dataset", "value": len(eligible_user_ids)},
            {"metric": "users_selected_for_run", "value": len(selected_user_ids)},
            {"metric": "excluded_users", "value": len(exclusion_rows)},
            {"metric": "training_interactions", "value": len(train)},
            {"metric": "validation_interactions", "value": len(validation)},
            {"metric": "test_interactions", "value": len(test)},
            {"metric": "relevance_threshold", "value": config.relevance_threshold},
            {
                "metric": "minimum_training_interactions",
                "value": config.minimum_training_interactions,
            },
        ]
    )
    return SplitData(
        train=train,
        validation=validation,
        test=test,
        eligible_users=eligible_users,
        excluded_users=pd.DataFrame(
            exclusion_rows,
            columns=[
                "user_id",
                "total_rating_count",
                "positive_interaction_count",
                "exclusion_reason",
            ],
        ),
        summary=summary,
        full_ratings=full,
    )


def sample_candidates(
    split: SplitData,
    targets: pd.DataFrame,
    all_movie_ids: np.ndarray,
    negative_samples: int,
    seed: int,
    split_name: str,
) -> tuple[dict[int, CandidateSet], pd.DataFrame]:
    """Sample deterministic unseen negatives for each held-out target."""
    by_user = {
        int(user_id): frame
        for user_id, frame in split.full_ratings.groupby("UserID", sort=False)
    }
    candidate_sets: dict[int, CandidateSet] = {}
    audit_rows: list[dict[str, object]] = []
    split_offset = 1_000_003 if split_name == "test" else 0

    for _, target in targets.iterrows():
        user_id = int(target["UserID"])
        history = by_user[user_id]
        before_target = history.loc[
            history["Timestamp"].lt(target["Timestamp"])
            | (
                history["Timestamp"].eq(target["Timestamp"])
                & history["_source_order"].le(target["_source_order"])
            )
        ]
        seen = set(int(item) for item in before_target["MovieID"])
        available = np.asarray(
            [item for item in all_movie_ids if int(item) not in seen],
            dtype=np.int64,
        )
        if len(available) < negative_samples:
            raise ValueError(
                f"User {user_id} has only {len(available)} valid negative candidates"
            )
        rng = np.random.default_rng(seed + split_offset + user_id * 10_007)
        negatives = rng.choice(available, negative_samples, replace=False)
        candidates = np.sort(
            np.concatenate(
                [
                    negatives.astype(np.int64),
                    np.asarray([target["MovieID"]], dtype=np.int64),
                ]
            )
        )
        candidate_sets[user_id] = CandidateSet(
            user_id=user_id,
            target_movie_id=int(target["MovieID"]),
            target_timestamp=int(target["Timestamp"]),
            candidate_movie_ids=candidates,
        )
        audit_rows.append(
            {
                "split": split_name,
                "user_id": user_id,
                "target_movie_id": int(target["MovieID"]),
                "target_timestamp": int(target["Timestamp"]),
                "candidate_movie_ids": "|".join(str(item) for item in candidates),
            }
        )
    return candidate_sets, pd.DataFrame(audit_rows)


def make_sequences(
    frame: pd.DataFrame,
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
) -> list[np.ndarray]:
    """Convert chronologically ordered interactions into indexed sequences."""
    grouped = {
        int(user_id): group.sort_values(
            ["Timestamp", "_source_order"], kind="mergesort"
        )
        for user_id, group in frame.groupby("UserID", sort=False)
    }
    return [
        np.asarray(
            [item_to_index[int(item)] for item in grouped[int(user_id)]["MovieID"]],
            dtype=np.int32,
        )
        for user_id in user_ids
    ]


def append_targets(
    sequences: list[np.ndarray],
    targets: pd.DataFrame,
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
) -> list[np.ndarray]:
    """Append one known validation interaction to each user sequence."""
    target_map = {
        int(row.UserID): item_to_index[int(row.MovieID)]
        for row in targets.itertuples(index=False)
    }
    return [
        np.append(sequence, target_map[int(user_id)]).astype(np.int32)
        for user_id, sequence in zip(user_ids, sequences, strict=True)
    ]


def shuffle_sequences(
    sequences: list[np.ndarray],
    seed: int,
) -> list[np.ndarray]:
    """Shuffle each sequence independently while preserving its item multiset."""
    shuffled: list[np.ndarray] = []
    for user_index, sequence in enumerate(sequences):
        rng = np.random.default_rng(seed + user_index * 1_009)
        shuffled.append(rng.permutation(sequence).astype(np.int32))
    return shuffled


def build_interaction_matrix(
    sequences: list[np.ndarray],
    item_count: int,
) -> sparse.csr_matrix:
    """Build a binary user-item matrix from positive interactions."""
    rows: list[int] = []
    columns: list[int] = []
    for user_index, sequence in enumerate(sequences):
        unique_items = np.unique(sequence)
        rows.extend([user_index] * len(unique_items))
        columns.extend(unique_items.tolist())
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(sequences), item_count),
        dtype=np.float32,
    )


def train_item_knn(
    interaction_matrix: sparse.csr_matrix,
    neighbors: int,
) -> sparse.csr_matrix:
    """Build a sparse cosine item-neighborhood similarity matrix."""
    item_user = interaction_matrix.T.tocsr()
    neighbor_count = min(neighbors + 1, item_user.shape[0])
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(item_user)
    distances, indices = model.kneighbors(item_user)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for item_index in range(item_user.shape[0]):
        for neighbor_index, distance in zip(
            indices[item_index], distances[item_index], strict=True
        ):
            if int(neighbor_index) == item_index:
                continue
            similarity = 1.0 - float(distance)
            if similarity > 0:
                rows.append(item_index)
                columns.append(int(neighbor_index))
                values.append(similarity)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(item_user.shape[0], item_user.shape[0]),
        dtype=np.float32,
    )


def _rank_metrics(
    scores: np.ndarray,
    candidate_movie_ids: np.ndarray,
    target_movie_id: int,
) -> tuple[float, float, float, float, int]:
    target_position = int(np.flatnonzero(candidate_movie_ids == target_movie_id)[0])
    target_score = float(scores[target_position])
    better = scores > target_score
    equal_before = (scores == target_score) & (candidate_movie_ids < target_movie_id)
    rank = 1 + int(better.sum()) + int(equal_before.sum())
    hit = float(rank <= 10)
    ndcg = float(1.0 / np.log2(rank + 1)) if rank <= 10 else 0.0
    reciprocal_rank = float(1.0 / rank) if rank <= 10 else 0.0
    return hit, hit, ndcg, reciprocal_rank, rank


def evaluate_scores(
    model_name: str,
    candidates: dict[int, CandidateSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    scorer: object,
) -> pd.DataFrame:
    """Evaluate a callable scorer on common per-user candidate sets."""
    rows: list[dict[str, object]] = []
    for user_index, user_id in enumerate(user_ids):
        candidate_set = candidates[int(user_id)]
        candidate_indices = np.asarray(
            [item_to_index[int(item)] for item in candidate_set.candidate_movie_ids],
            dtype=np.int32,
        )
        scores = np.asarray(scorer(user_index, candidate_indices), dtype=float)
        hit, recall, ndcg, mrr, rank = _rank_metrics(
            scores,
            candidate_set.candidate_movie_ids,
            candidate_set.target_movie_id,
        )
        rows.append(
            {
                "user_id": int(user_id),
                "model_name": model_name,
                "hit_rate_10": hit,
                "recall_10": recall,
                "ndcg_10": ndcg,
                "mrr_10": mrr,
                "target_movie_id": candidate_set.target_movie_id,
                "target_rank": rank,
                "candidate_count": len(candidate_set.candidate_movie_ids),
            }
        )
    return pd.DataFrame(rows)


def tune_and_evaluate_models(
    split: SplitData,
    movies: pd.DataFrame,
    validation_candidates: dict[int, CandidateSet],
    test_candidates: dict[int, CandidateSet],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Tune models on validation, refit with validation, and evaluate test users."""
    user_ids = split.eligible_users["user_id"].to_numpy(dtype=np.int64)
    movie_ids = np.sort(movies["MovieID"].unique()).astype(np.int64)
    item_to_index = {int(item): index for index, item in enumerate(movie_ids)}
    train_sequences = make_sequences(split.train, user_ids, item_to_index)
    train_matrix = build_interaction_matrix(train_sequences, len(movie_ids))
    epochs = min(config.maximum_epochs, 2) if config.fast_test else config.maximum_epochs
    tuning_rows: list[dict[str, object]] = []

    popularity = np.asarray(train_matrix.sum(axis=0)).ravel()
    popularity_validation = evaluate_scores(
        "popularity",
        validation_candidates,
        user_ids,
        item_to_index,
        lambda _user, candidate_indices: popularity[candidate_indices],
    )
    tuning_rows.append(
        {
            "model_name": "popularity",
            "hyperparameter": "none",
            "value": "global_positive_count",
            "validation_ndcg_10": popularity_validation["ndcg_10"].mean(),
        }
    )

    knn_options = [30, 80] if config.fast_test else [50, 100]
    best_knn: tuple[float, int, sparse.csr_matrix] | None = None
    for neighbors in knn_options:
        similarity = train_item_knn(train_matrix, neighbors)

        def knn_scorer(
            user_index: int,
            candidate_indices: np.ndarray,
            similarity_matrix: sparse.csr_matrix = similarity,
        ) -> np.ndarray:
            return (
                similarity_matrix[candidate_indices]
                .dot(train_matrix[user_index].T)
                .toarray()
                .ravel()
            )

        metrics = evaluate_scores(
            f"item_knn_k{neighbors}",
            validation_candidates,
            user_ids,
            item_to_index,
            knn_scorer,
        )
        score = float(metrics["ndcg_10"].mean())
        tuning_rows.append(
            {
                "model_name": "item_knn",
                "hyperparameter": "neighbors",
                "value": neighbors,
                "validation_ndcg_10": score,
            }
        )
        if best_knn is None or score > best_knn[0]:
            best_knn = (score, neighbors, similarity)
    assert best_knn is not None

    svd_options = [16, 32] if config.fast_test else [32, 64]
    best_svd: tuple[float, int] | None = None
    for factors in svd_options:
        svd = TruncatedSVD(n_components=factors, random_state=config.random_seed)
        user_embeddings = svd.fit_transform(train_matrix)

        def svd_scorer(
            user_index: int,
            candidate_indices: np.ndarray,
            embeddings: np.ndarray = user_embeddings,
            components: np.ndarray = svd.components_,
        ) -> np.ndarray:
            return embeddings[user_index] @ components[:, candidate_indices]

        metrics = evaluate_scores(
            f"matrix_factorization_d{factors}",
            validation_candidates,
            user_ids,
            item_to_index,
            svd_scorer,
        )
        score = float(metrics["ndcg_10"].mean())
        tuning_rows.append(
            {
                "model_name": "matrix_factorization",
                "hyperparameter": "latent_dimensions",
                "value": factors,
                "validation_ndcg_10": score,
            }
        )
        if best_svd is None or score > best_svd[0]:
            best_svd = (score, factors)
    assert best_svd is not None

    fpmc_factors = 16 if config.fast_test else 24
    fpmc_validation_model = FPMC(
        user_count=len(user_ids),
        item_count=len(movie_ids),
        factors=fpmc_factors,
        learning_rate=0.03,
        regularization=0.002,
        seed=config.random_seed,
    ).fit(train_sequences, epochs)
    transition_weights = [0.5, 1.0, 2.0]
    best_fpmc_weight: tuple[float, float] | None = None
    previous_train_items = np.asarray(
        [sequence[-1] for sequence in train_sequences], dtype=np.int32
    )
    for transition_weight in transition_weights:
        metrics = evaluate_scores(
            f"fpmc_weight_{transition_weight:g}",
            validation_candidates,
            user_ids,
            item_to_index,
            lambda user_index, candidate_indices, weight=transition_weight: (
                fpmc_validation_model.score(
                    user_index,
                    int(previous_train_items[user_index]),
                    candidate_indices,
                    weight,
                )
            ),
        )
        score = float(metrics["ndcg_10"].mean())
        tuning_rows.append(
            {
                "model_name": "fpmc",
                "hyperparameter": "transition_weight",
                "value": transition_weight,
                "validation_ndcg_10": score,
            }
        )
        if best_fpmc_weight is None or score > best_fpmc_weight[0]:
            best_fpmc_weight = (score, transition_weight)
    assert best_fpmc_weight is not None

    final_sequences = append_targets(
        train_sequences, split.validation, user_ids, item_to_index
    )
    final_matrix = build_interaction_matrix(final_sequences, len(movie_ids))
    final_popularity = np.asarray(final_matrix.sum(axis=0)).ravel()
    model_metrics = [
        evaluate_scores(
            "popularity",
            test_candidates,
            user_ids,
            item_to_index,
            lambda _user, candidate_indices: final_popularity[candidate_indices],
        )
    ]

    final_similarity = train_item_knn(final_matrix, best_knn[1])
    model_metrics.append(
        evaluate_scores(
            "item_knn",
            test_candidates,
            user_ids,
            item_to_index,
            lambda user_index, candidate_indices: (
                final_similarity[candidate_indices]
                .dot(final_matrix[user_index].T)
                .toarray()
                .ravel()
            ),
        )
    )

    final_svd = TruncatedSVD(
        n_components=best_svd[1], random_state=config.random_seed
    )
    final_embeddings = final_svd.fit_transform(final_matrix)
    model_metrics.append(
        evaluate_scores(
            "matrix_factorization",
            test_candidates,
            user_ids,
            item_to_index,
            lambda user_index, candidate_indices: (
                final_embeddings[user_index]
                @ final_svd.components_[:, candidate_indices]
            ),
        )
    )

    final_fpmc = FPMC(
        user_count=len(user_ids),
        item_count=len(movie_ids),
        factors=fpmc_factors,
        learning_rate=0.03,
        regularization=0.002,
        seed=config.random_seed,
    ).fit(final_sequences, epochs)
    previous_final_items = np.asarray(
        [sequence[-1] for sequence in final_sequences], dtype=np.int32
    )
    model_metrics.append(
        evaluate_scores(
            "fpmc_original",
            test_candidates,
            user_ids,
            item_to_index,
            lambda user_index, candidate_indices: final_fpmc.score(
                user_index,
                int(previous_final_items[user_index]),
                candidate_indices,
                best_fpmc_weight[1],
            ),
        )
    )

    shuffled_results: list[pd.DataFrame] = []
    shuffle_runs = 1 if config.fast_test else config.shuffle_runs
    for run_index in range(shuffle_runs):
        shuffle_seed = config.random_seed + 10_000 + run_index
        shuffled_train = shuffle_sequences(train_sequences, shuffle_seed)
        shuffled_final = append_targets(
            shuffled_train, split.validation, user_ids, item_to_index
        )
        shuffled_model = FPMC(
            user_count=len(user_ids),
            item_count=len(movie_ids),
            factors=fpmc_factors,
            learning_rate=0.03,
            regularization=0.002,
            seed=config.random_seed,
        ).fit(shuffled_final, epochs)
        shuffled_previous = np.asarray(
            [sequence[-1] for sequence in shuffled_final], dtype=np.int32
        )
        metrics = evaluate_scores(
            f"fpmc_shuffled_seed_{shuffle_seed}",
            test_candidates,
            user_ids,
            item_to_index,
            lambda user_index, candidate_indices, model=shuffled_model, previous=shuffled_previous: model.score(
                user_index,
                int(previous[user_index]),
                candidate_indices,
                best_fpmc_weight[1],
            ),
        )
        metrics["shuffle_seed"] = shuffle_seed
        shuffled_results.append(metrics)
        model_metrics.append(metrics.drop(columns="shuffle_seed"))

    selected_hyperparameters: dict[str, object] = {
        "popularity": {"score": "training positive-interaction count"},
        "item_knn": {"neighbors": best_knn[1], "similarity": "cosine"},
        "matrix_factorization": {
            "algorithm": "TruncatedSVD",
            "latent_dimensions": best_svd[1],
        },
        "fpmc": {
            "latent_dimensions": fpmc_factors,
            "learning_rate": 0.03,
            "regularization": 0.002,
            "epochs": epochs,
            "transition_weight": best_fpmc_weight[1],
            "shuffle_runs": shuffle_runs,
        },
        "validation_search": tuning_rows,
    }
    return (
        pd.concat(model_metrics, ignore_index=True),
        selected_hyperparameters,
        pd.concat(shuffled_results, ignore_index=True),
    )


def leakage_safe_features(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute temporal proxies using positive training interactions only."""
    timed = train.drop(
        columns=["_source_order", "positive_position", "positive_count"],
        errors="ignore",
    ).reset_index(drop=True)
    timed["RatingTime"] = pd.to_datetime(
        timed["Timestamp"], unit="s", utc=True, errors="raise"
    )
    temporal = build_user_temporal_features(timed)
    genre = build_genre_evolution_features(timed, movies)
    features = temporal.merge(genre, on="user_id", how="inner", validate="one_to_one")
    features["session_count"] = features["session_count_2h"]
    if len(features) != train["UserID"].nunique():
        raise RuntimeError("Leakage-safe features did not preserve all experiment users")
    return features


def _bootstrap_mean_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate_performance(
    per_user: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Aggregate per-user metrics with user-level bootstrap intervals."""
    rows: list[dict[str, object]] = []
    for model_name, frame in per_user.groupby("model_name", sort=False):
        for metric in ("hit_rate_10", "recall_10", "ndcg_10", "mrr_10"):
            values = frame[metric].to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_interval(
                values, bootstrap_samples, seed + len(rows)
            )
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": np.median(values),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "user_count": len(values),
                }
            )
    return pd.DataFrame(rows)


def build_advantage_table(
    per_user: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the user-model matrix and sequential advantage table."""
    matrix = per_user.pivot(
        index="user_id", columns="model_name", values="ndcg_10"
    ).reset_index()
    non_sequential = matrix[list(NON_SEQUENTIAL_MODELS)]
    matrix["best_non_sequential_ndcg_10"] = non_sequential.max(axis=1)
    matrix["best_non_sequential_model_deterministic"] = non_sequential.idxmax(axis=1)
    matrix["best_non_sequential_tie_count"] = non_sequential.eq(
        matrix["best_non_sequential_ndcg_10"], axis=0
    ).sum(axis=1)
    matrix["best_non_sequential_model"] = np.where(
        matrix["best_non_sequential_tie_count"].gt(1),
        "tie",
        matrix["best_non_sequential_model_deterministic"],
    )
    matrix["delta_ndcg"] = (
        matrix["fpmc_original"] - matrix["best_non_sequential_ndcg_10"]
    )
    core = matrix[list(CORE_MODELS)]
    matrix["best_model_ndcg_10"] = core.max(axis=1)
    matrix["best_model_deterministic"] = core.idxmax(axis=1)
    matrix["best_model_tie_count"] = core.eq(
        matrix["best_model_ndcg_10"], axis=0
    ).sum(axis=1)
    matrix["best_model"] = np.where(
        matrix["best_model_tie_count"].gt(1),
        "tie",
        matrix["best_model_deterministic"],
    )
    matrix["all_core_models_zero"] = core.eq(0).all(axis=1)
    matrix["sequential_model_best"] = matrix["best_model"].eq("fpmc_original")
    shuffled_columns = [
        column for column in matrix.columns if column.startswith("fpmc_shuffled_seed_")
    ]
    matrix["fpmc_shuffled_mean_ndcg_10"] = matrix[shuffled_columns].mean(axis=1)
    matrix["original_minus_shuffled_ndcg_10"] = (
        matrix["fpmc_original"] - matrix["fpmc_shuffled_mean_ndcg_10"]
    )
    advantage = matrix.merge(features, on="user_id", how="inner", validate="one_to_one")
    return matrix, advantage


def add_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Add distributional and interpretable user-group labels."""
    grouped = frame.copy()
    labels = ["low", "medium", "high"]
    for feature in (
        "largest_day_ratio",
        "history_span_days",
        "rating_count",
        "largest_session_ratio_2h",
        "genre_js_divergence",
    ):
        percentile_rank = grouped[feature].rank(method="average", pct=True)
        grouped[f"{feature}_quantile_group"] = pd.cut(
            percentile_rank,
            bins=[0, 1 / 3, 2 / 3, 1],
            labels=labels,
            include_lowest=True,
        )
    grouped["largest_day_ratio_threshold_group"] = pd.cut(
        grouped["largest_day_ratio"],
        bins=[-np.inf, 0.5, 0.8, np.inf],
        labels=["below_0.50", "0.50_to_below_0.80", "0.80_or_above"],
        right=False,
    )
    grouped["history_span_range"] = pd.cut(
        grouped["history_span_days"],
        bins=[-np.inf, 1, 30, 365, np.inf],
        labels=[
            "within_1_day",
            "over_1_to_30_days",
            "over_30_to_365_days",
            "over_365_days",
        ],
        right=True,
    )
    return grouped


def group_performance_summary(
    advantage: pd.DataFrame,
    per_user: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize metrics and model advantage within user groups."""
    metric_lookup = per_user.pivot(
        index="user_id", columns="model_name", values="ndcg_10"
    )
    model_columns = list(CORE_MODELS)
    group_variables = [
        "largest_day_ratio_quantile_group",
        "largest_day_ratio_threshold_group",
        "history_span_days_quantile_group",
        "history_span_range",
        "rating_count_quantile_group",
        "largest_session_ratio_2h_quantile_group",
        "genre_js_divergence_quantile_group",
    ]
    rows: list[dict[str, object]] = []
    for group_variable in group_variables:
        for group_label, group in advantage.groupby(
            group_variable, observed=True, sort=False
        ):
            user_ids = group["user_id"].to_numpy()
            delta_values = group["delta_ndcg"].to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_interval(
                delta_values, bootstrap_samples, seed + len(rows)
            )
            for model_name in model_columns:
                model_values = metric_lookup.loc[user_ids, model_name].to_numpy()
                rows.append(
                    {
                        "group_variable": group_variable,
                        "group_label": str(group_label),
                        "model_name": model_name,
                        "user_count": len(group),
                        "small_group_warning": len(group) < 30,
                        "mean_ndcg_10": model_values.mean(),
                        "median_ndcg_10": np.median(model_values),
                        "mean_delta_ndcg": delta_values.mean(),
                        "median_delta_ndcg": np.median(delta_values),
                        "delta_ci_95_lower": lower,
                        "delta_ci_95_upper": upper,
                        "percentage_sequential_best": (
                            group["sequential_model_best"].mean() * 100
                        ),
                        "mean_original_minus_shuffled_ndcg_10": group[
                            "original_minus_shuffled_ndcg_10"
                        ].mean(),
                    }
                )
    return pd.DataFrame(rows)


def association_analysis(
    advantage: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit standardized ridge associations and stratified comparisons."""
    transformed = pd.DataFrame(
        {
            "log1p_rating_count": np.log1p(advantage["rating_count"]),
            "log1p_history_span_days": np.log1p(advantage["history_span_days"]),
            "largest_day_ratio": advantage["largest_day_ratio"],
            "largest_session_ratio_2h": advantage["largest_session_ratio_2h"],
            "log1p_median_gap_minutes": np.log1p(advantage["median_gap_minutes"]),
            "genre_js_divergence": advantage["genre_js_divergence"],
        }
    )
    target = advantage["delta_ndcg"].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaled, target)

    rows: list[dict[str, object]] = [
        {
            "result_type": "ridge_coefficient",
            "term": name,
            "value": coefficient,
            "details": "standardized predictor; association only",
        }
        for name, coefficient in zip(
            transformed.columns, ridge.coef_, strict=True
        )
    ]
    rows.append(
        {
            "result_type": "ridge_intercept",
            "term": "intercept",
            "value": ridge.intercept_,
            "details": "mean-scale delta_ndcg",
        }
    )

    fold_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_indices, test_indices in folds.split(scaled):
        fold_model = Ridge(alpha=1.0).fit(scaled[train_indices], target[train_indices])
        fold_scores.append(
            r2_score(target[test_indices], fold_model.predict(scaled[test_indices]))
        )
    rows.extend(
        [
            {
                "result_type": "cross_validated_r2",
                "term": "mean",
                "value": float(np.mean(fold_scores)),
                "details": "5-fold user-level cross-validation",
            },
            {
                "result_type": "cross_validated_r2",
                "term": "standard_deviation",
                "value": float(np.std(fold_scores, ddof=1)),
                "details": "5-fold user-level cross-validation",
            },
        ]
    )

    feature_correlations = transformed.corr(method="spearman")
    upper = feature_correlations.where(
        np.triu(np.ones(feature_correlations.shape), k=1).astype(bool)
    ).stack()
    strongest_pair = upper.abs().idxmax()
    rows.append(
        {
            "result_type": "multicollinearity_check",
            "term": f"{strongest_pair[0]}__{strongest_pair[1]}",
            "value": float(feature_correlations.loc[strongest_pair]),
            "details": "strongest absolute predictor Spearman correlation",
        }
    )

    for (rating_group, concentration_group), group in advantage.groupby(
        ["rating_count_quantile_group", "largest_day_ratio_quantile_group"],
        observed=True,
    ):
        values = group["delta_ndcg"].to_numpy(dtype=float)
        lower, upper_bound = _bootstrap_mean_interval(
            values, bootstrap_samples, seed + len(rows)
        )
        rows.append(
            {
                "result_type": "rating_count_stratified_delta",
                "term": f"{rating_group}__{concentration_group}",
                "value": values.mean(),
                "details": (
                    f"n={len(values)}; median={np.median(values):.6f}; "
                    f"bootstrap95=[{lower:.6f},{upper_bound:.6f}]"
                ),
            }
        )
    return pd.DataFrame(rows), feature_correlations


def statistical_comparisons(
    advantage: pd.DataFrame,
    overall: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Create paired user-level effect-size and uncertainty summaries."""
    overall_ndcg = overall.loc[overall["metric"].eq("ndcg_10")].set_index("model_name")
    strongest_non_sequential = overall_ndcg.loc[
        list(NON_SEQUENTIAL_MODELS), "mean"
    ].idxmax()
    comparisons = {
        f"fpmc_original_minus_{strongest_non_sequential}": (
            advantage["fpmc_original"] - advantage[strongest_non_sequential]
        ).to_numpy(),
        "fpmc_original_minus_best_non_sequential_per_user": advantage[
            "delta_ndcg"
        ].to_numpy(),
        "fpmc_original_minus_shuffled_mean": advantage[
            "original_minus_shuffled_ndcg_10"
        ].to_numpy(),
    }
    rows = []
    for comparison, values in comparisons.items():
        lower, upper = _bootstrap_mean_interval(
            values, bootstrap_samples, seed + len(rows)
        )
        rows.append(
            {
                "comparison": comparison,
                "user_count": len(values),
                "mean_difference": values.mean(),
                "median_difference": np.median(values),
                "standardized_effect": (
                    values.mean() / values.std(ddof=1)
                    if values.std(ddof=1) > 0
                    else 0.0
                ),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "percentage_positive": float((values > 0).mean() * 100),
                "resampling_unit": "user",
            }
        )
    best_single = overall_ndcg.loc[list(CORE_MODELS), "mean"].idxmax()
    oracle = advantage[list(CORE_MODELS)].max(axis=1)
    rows.append(
        {
            "comparison": "oracle_selector_minus_best_single_model",
            "user_count": len(advantage),
            "mean_difference": oracle.mean() - overall_ndcg.loc[best_single, "mean"],
            "median_difference": np.nan,
            "standardized_effect": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
            "percentage_positive": float(
                (oracle > advantage[best_single]).mean() * 100
            ),
            "resampling_unit": "user",
        }
    )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    overall: pd.DataFrame,
    advantage: pd.DataFrame,
    group_summary: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Create the required model-feasibility plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    overall_ndcg = overall.loc[
        overall["metric"].eq("ndcg_10")
        & overall["model_name"].isin(CORE_MODELS)
    ].sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(overall_ndcg["model_name"], overall_ndcg["mean"])
    ax.set_ylim(bottom=0)
    ax.set_title("Overall NDCG@10 by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean NDCG@10")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, plots_dir / "01_overall_ndcg_by_model.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(advantage["delta_ndcg"], bins=40)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Distribution of Sequential Model Advantage")
    ax.set_xlabel("FPMC NDCG@10 − best non-sequential NDCG@10")
    ax.set_ylabel("Number of users")
    _save_figure(fig, plots_dir / "02_delta_ndcg_distribution.png")

    best_counts = advantage["best_model"].value_counts(normalize=True).mul(100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(best_counts.index, best_counts.values)
    ax.set_ylim(0, 100)
    ax.set_title("Unique Best Model and Tied Outcomes")
    ax.set_xlabel("Best-outcome category")
    ax.set_ylabel("Users (%)")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, plots_dir / "03_best_model_percentage.png")

    plot_groupings = [
        (
            "largest_day_ratio_threshold_group",
            "Sequential Advantage by Largest-Day Ratio",
            "04_delta_by_largest_day_group.png",
        ),
        (
            "history_span_range",
            "Sequential Advantage by Rating-History Span",
            "05_delta_by_history_span_group.png",
        ),
        (
            "rating_count_quantile_group",
            "Sequential Advantage by Rating-Count Group",
            "06_delta_by_rating_count_group.png",
        ),
    ]
    for group_column, title, filename in plot_groupings:
        grouped_values = [
            group["delta_ndcg"].to_numpy()
            for _, group in advantage.groupby(group_column, observed=True, sort=False)
        ]
        labels = [
            f"{label}\n(n={len(group)})"
            for label, group in advantage.groupby(
                group_column, observed=True, sort=False
            )
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(grouped_values, tick_labels=labels, showfliers=False)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("User group")
        ax.set_ylabel("Delta NDCG@10")
        ax.tick_params(axis="x", rotation=15)
        _save_figure(fig, plots_dir / filename)

    shuffled_columns = [
        column for column in advantage.columns if column.startswith("fpmc_shuffled_seed_")
    ]
    shuffled_means = [advantage[column].mean() for column in shuffled_columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["original"] + [column.rsplit("_", 1)[-1] for column in shuffled_columns]
    values = [advantage["fpmc_original"].mean()] + shuffled_means
    ax.bar(labels, values)
    ax.set_ylim(bottom=0)
    ax.set_title("Original vs Shuffled FPMC Order")
    ax.set_xlabel("Sequence condition / shuffle seed")
    ax.set_ylabel("Mean NDCG@10")
    _save_figure(fig, plots_dir / "07_original_vs_shuffled.png")

    scatter_specs = [
        (
            "largest_day_ratio",
            "Largest-day ratio",
            False,
            "08_advantage_vs_largest_day_ratio.png",
        ),
        (
            "history_span_days",
            "log1p(history span days)",
            True,
            "09_advantage_vs_history_span.png",
        ),
        (
            "rating_count",
            "log1p(rating count)",
            True,
            "10_advantage_vs_rating_count.png",
        ),
    ]
    for feature, label, log_transform, filename in scatter_specs:
        x_values = (
            np.log1p(advantage[feature])
            if log_transform
            else advantage[feature]
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_values, advantage["delta_ndcg"], s=10, alpha=0.25)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"Sequential Advantage vs {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Delta NDCG@10")
        _save_figure(fig, plots_dir / filename)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(
        feature_correlations.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm"
    )
    ax.set_title("Association-Feature Spearman Correlations")
    ax.set_xticks(
        range(len(feature_correlations.columns)),
        feature_correlations.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(feature_correlations.index)), feature_correlations.index)
    for row in range(len(feature_correlations)):
        for column in range(len(feature_correlations.columns)):
            ax.text(
                column,
                row,
                f"{feature_correlations.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    _save_figure(fig, plots_dir / "11_association_feature_heatmap.png")


def create_report(
    config: ExperimentConfig,
    split: SplitData,
    overall: pd.DataFrame,
    advantage: pd.DataFrame,
    group_summary: pd.DataFrame,
    association: pd.DataFrame,
    statistical: pd.DataFrame,
) -> str:
    """Create the actual-results model feasibility report."""
    overall_ndcg = overall.loc[
        overall["metric"].eq("ndcg_10")
        & overall["model_name"].isin(CORE_MODELS)
    ].set_index("model_name")
    best_model = overall_ndcg["mean"].idxmax()
    best_non_sequential = overall_ndcg.loc[
        list(NON_SEQUENTIAL_MODELS), "mean"
    ].idxmax()
    sequential_difference = (
        overall_ndcg.loc["fpmc_original", "mean"]
        - overall_ndcg.loc[best_non_sequential, "mean"]
    )
    best_percentages = advantage["best_model"].value_counts(normalize=True).mul(100)
    tied_percentage = float(advantage["best_model"].eq("tie").mean() * 100)
    all_zero_percentage = float(advantage["all_core_models_zero"].mean() * 100)
    oracle_row = statistical.loc[
        statistical["comparison"].eq("oracle_selector_minus_best_single_model")
    ].iloc[0]
    sequential_comparison = statistical.loc[
        statistical["comparison"].eq(
            f"fpmc_original_minus_{best_non_sequential}"
        )
    ].iloc[0]
    shuffle_comparison = statistical.loc[
        statistical["comparison"].eq("fpmc_original_minus_shuffled_mean")
    ].iloc[0]
    shuffle_difference = advantage["original_minus_shuffled_ndcg_10"].mean()
    regression_r2 = association.loc[
        association["result_type"].eq("cross_validated_r2")
        & association["term"].eq("mean"),
        "value",
    ].iloc[0]
    day_coefficient = association.loc[
        association["result_type"].eq("ridge_coefficient")
        & association["term"].eq("largest_day_ratio"),
        "value",
    ].iloc[0]

    day_groups = (
        group_summary.loc[
            group_summary["group_variable"].eq(
                "largest_day_ratio_threshold_group"
            )
            & group_summary["model_name"].eq("fpmc_original")
        ]
        .set_index("group_label")
    )
    low_day_delta = day_groups.loc["below_0.50", "mean_delta_ndcg"]
    high_day_delta = day_groups.loc["0.80_or_above", "mean_delta_ndcg"]
    history_groups = (
        group_summary.loc[
            group_summary["group_variable"].eq("history_span_range")
            & group_summary["model_name"].eq("fpmc_original")
        ]
        .set_index("group_label")
    )
    within_day_delta = history_groups.loc["within_1_day", "mean_delta_ndcg"]
    over_year_delta = history_groups.loc["over_365_days", "mean_delta_ndcg"]

    sequential_best_percentage = float(best_percentages.get("fpmc_original", 0.0))
    heterogeneous = best_percentages.max() < 90 and sequential_best_percentage >= 5
    meaningful_oracle = float(oracle_row["mean_difference"]) >= 0.01
    meaningful_shuffle = shuffle_difference >= 0.003
    temporal_signal = regression_r2 >= 0.01
    if (
        heterogeneous
        and sequential_best_percentage >= 15
        and sequential_difference >= -0.02
        and meaningful_oracle
        and meaningful_shuffle
        and temporal_signal
    ):
        decision = "Proceed"
    elif (
        heterogeneous
        and float(oracle_row["mean_difference"]) >= 0.005
        and (meaningful_shuffle or temporal_signal)
    ):
        decision = "Proceed with revisions"
    else:
        decision = "Do not proceed"

    model_lines = "\n".join(
        f"- `{model}`: mean NDCG@10 {row['mean']:.4f} "
        f"(95% user-bootstrap CI {row['ci_95_lower']:.4f}–{row['ci_95_upper']:.4f})"
        for model, row in overall_ndcg.iterrows()
    )
    best_lines = "\n".join(
        f"- `{model}`: {percentage:.2f}% of users"
        for model, percentage in best_percentages.items()
    )
    shuffle_runs = 1 if config.fast_test else config.shuffle_runs

    return f"""# MovieLens 1M Model Feasibility Report

## Experimental scope

The experiment treats ratings of {config.relevance_threshold:g} or higher as positive interactions and uses a chronological leave-last-two-out split with at least {config.minimum_training_interactions} positive training interactions. Each held-out target is ranked against the same {config.negative_samples} deterministic items not previously rated before that target. Metrics therefore describe sampled-candidate Top-K ranking, not full-catalog ranking.

Timestamps are rating-submission times, not movie-watching times. The FPMC sequence is a rating interaction sequence and must not be interpreted as a reconstructed watching history.

## A. Which model performs best overall?

{model_lines}

The best overall model is `{best_model}`.

## B. Does the sequential model outperform the strongest non-sequential model?

FPMC differs from the strongest non-sequential model (`{best_non_sequential}`) by {sequential_difference:+.4f} mean NDCG@10. The paired user-bootstrap 95% interval is {float(sequential_comparison['ci_95_lower']):+.4f} to {float(sequential_comparison['ci_95_upper']):+.4f}, with standardized effect {float(sequential_comparison['standardized_effect']):+.3f}. This separates statistical uncertainty from the practical magnitude.

## C. How much user-level heterogeneity exists?

Unique-winner and tied-outcome assignments are:

{best_lines}

Multiple models tie for best for {tied_percentage:.2f}% of users, including {all_zero_percentage:.2f}% for whom every core model misses the target in the top 10. FPMC is uniquely best for {sequential_best_percentage:.2f}% of users despite its weaker overall average. Ties are reported explicitly rather than assigned arbitrarily.

## D. Does sequence shuffling reduce performance?

Across {shuffle_runs} deterministic shuffled-order run(s), original-order FPMC exceeds shuffled FPMC by {shuffle_difference:+.4f} mean NDCG@10. The paired user-bootstrap 95% interval is {float(shuffle_comparison['ci_95_lower']):+.4f} to {float(shuffle_comparison['ci_95_upper']):+.4f}, with standardized effect {float(shuffle_comparison['standardized_effect']):+.3f}. The same hyperparameters, validation/test targets, candidates, and per-user item multisets are retained.

## E. Is FPMC relatively stronger for temporally distributed users?

Mean sequential-minus-best-non-sequential NDCG@10 is {low_day_delta:+.4f} for users with largest-day ratio below 0.50 and {high_day_delta:+.4f} for users at or above 0.80. By duration, the mean is {within_day_delta:+.4f} for histories within one day and {over_year_delta:+.4f} for histories longer than one year. Group results are associations and may reflect history length or other confounding.

## F. Does temporal concentration retain an association after rating count control?

The standardized ridge model has mean 5-fold cross-validated R² {regression_r2:.4f}. Its standardized largest-day-ratio coefficient is {day_coefficient:+.4f} after including log rating count, log history span, largest-session ratio at 2 hours, log median gap, and genre divergence. This is an exploratory conditional association, not a causal estimate.

## G. Is there oracle headroom for a later selector?

The oracle that chooses the best tested core model per user improves mean NDCG@10 over the best single model by {float(oracle_row['mean_difference']):.4f}, and improves outcomes for {float(oracle_row['percentage_positive']):.2f}% of users. This is descriptive oracle headroom, not achievable selector performance, and part of it may arise from choosing among non-sequential models rather than sequential suitability.

## H. Should the project proceed?

**{decision}.** The decision weighs model heterogeneity, oracle headroom, order sensitivity, and conditional temporal signal. Statistical uncertainty and practical effect sizes should be considered alongside average rankings; significance alone would not justify a selector.

## Limitations

- Rating timestamps are not watching timestamps.
- Many ratings may represent profile-building or historical backfilling.
- Genre divergence is only a proxy for observed change in rated-movie composition.
- There is no ground-truth timestamp-quality label.
- MovieLens 1M is an old, controlled recommender-system dataset.
- Results may not generalise to commercial streaming platforms.
- FPMC is a lightweight CPU implementation for feasibility testing, and evaluation uses sampled negatives rather than full ranking.
"""


def run_experiment(config: ExperimentConfig) -> None:
    """Run the complete staged model-feasibility experiment."""
    print("[1/10] Loading MovieLens 1M")
    ratings = load_ml1m_ratings(config.data_dir)
    movies = load_ml1m_movies(config.data_dir)

    print("[2/10] Building chronological positive-interaction splits")
    split = chronological_split(ratings, config)
    all_movie_ids = np.sort(movies["MovieID"].unique()).astype(np.int64)

    print("[3/10] Sampling common validation and test candidates")
    validation_candidates, validation_audit = sample_candidates(
        split,
        split.validation,
        all_movie_ids,
        config.negative_samples,
        config.random_seed,
        "validation",
    )
    test_candidates, test_audit = sample_candidates(
        split,
        split.test,
        all_movie_ids,
        config.negative_samples,
        config.random_seed,
        "test",
    )

    print("[4/10] Recomputing leakage-safe training-history features")
    features = leakage_safe_features(split.train, movies)

    print("[5/10] Tuning, fitting, and evaluating recommender models")
    per_user, hyperparameters, shuffled = tune_and_evaluate_models(
        split,
        movies,
        validation_candidates,
        test_candidates,
        config,
    )

    print("[6/10] Aggregating metrics and model advantage")
    bootstrap_samples = (
        min(config.bootstrap_samples, 100)
        if config.fast_test
        else config.bootstrap_samples
    )
    overall = aggregate_performance(
        per_user, bootstrap_samples, config.random_seed
    )
    matrix, advantage = build_advantage_table(per_user, features)
    advantage = add_groups(advantage)
    shuffled_original = per_user.loc[
        per_user["model_name"].eq("fpmc_original"),
        ["user_id", "ndcg_10", "hit_rate_10"],
    ].rename(
        columns={
            "ndcg_10": "original_ndcg_10",
            "hit_rate_10": "original_hit_rate_10",
        }
    )
    shuffled = shuffled.merge(
        shuffled_original, on="user_id", how="left", validate="many_to_one"
    )
    shuffled["ndcg_difference"] = (
        shuffled["original_ndcg_10"] - shuffled["ndcg_10"]
    )
    shuffled["hit_rate_difference"] = (
        shuffled["original_hit_rate_10"] - shuffled["hit_rate_10"]
    )

    print("[7/10] Running group, association, and uncertainty analyses")
    group_summary = group_performance_summary(
        advantage,
        per_user,
        bootstrap_samples,
        config.random_seed,
    )
    association, feature_correlations = association_analysis(
        advantage, bootstrap_samples, config.random_seed
    )
    statistical = statistical_comparisons(
        advantage,
        overall,
        bootstrap_samples,
        config.random_seed,
    )

    print("[8/10] Writing reproducible outputs")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(config.serializable(), indent=2), encoding="utf-8"
    )
    split.summary.to_csv(output_dir / "split_summary.csv", index=False)
    split.eligible_users.to_csv(output_dir / "eligible_users.csv", index=False)
    split.excluded_users.to_csv(output_dir / "excluded_users.csv", index=False)
    overall.to_csv(output_dir / "model_overall_metrics.csv", index=False)
    per_user.to_csv(output_dir / "model_per_user_metrics.csv", index=False)
    matrix.to_csv(output_dir / "user_model_performance_matrix.csv", index=False)
    advantage.to_csv(output_dir / "user_level_model_advantage.csv", index=False)
    shuffled.to_csv(output_dir / "shuffled_sequence_results.csv", index=False)
    group_summary.to_csv(output_dir / "group_performance_summary.csv", index=False)
    association.to_csv(
        output_dir / "regression_or_association_results.csv", index=False
    )
    Path(output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2), encoding="utf-8"
    )
    statistical.to_csv(output_dir / "statistical_comparisons.csv", index=False)
    pd.concat([validation_audit, test_audit], ignore_index=True).to_csv(
        output_dir / "evaluation_candidates.csv", index=False
    )

    print("[9/10] Creating figures and actual-results report")
    create_plots(overall, advantage, group_summary, feature_correlations, plots_dir)
    report = create_report(
        config,
        split,
        overall,
        advantage,
        group_summary,
        association,
        statistical,
    )
    (output_dir / "model_feasibility_report.md").write_text(
        report, encoding="utf-8"
    )

    print("[10/10] Verifying output user preservation")
    expected_users = set(split.eligible_users["user_id"])
    if set(advantage["user_id"]) != expected_users:
        raise RuntimeError("Final model-advantage output does not preserve all test users")
    print(
        f"Complete: {len(advantage):,} evaluated users; outputs written to {output_dir}"
    )


def parse_args() -> ExperimentConfig:
    """Parse command-line configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ML1M_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--minimum-training-interactions", type=int, default=5)
    parser.add_argument("--negative-samples", type=int, default=99)
    parser.add_argument("--random-seed", type=int, default=760)
    parser.add_argument("--selected-sequential-model", choices=["fpmc"], default="fpmc")
    parser.add_argument("--maximum-epochs", type=int, default=3)
    parser.add_argument("--shuffle-runs", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--fast-test", action="store_true")
    parser.add_argument("--fast-user-limit", type=int, default=400)
    arguments = parser.parse_args()
    return ExperimentConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        relevance_threshold=arguments.relevance_threshold,
        minimum_training_interactions=arguments.minimum_training_interactions,
        negative_samples=arguments.negative_samples,
        random_seed=arguments.random_seed,
        selected_sequential_model=arguments.selected_sequential_model,
        maximum_epochs=arguments.maximum_epochs,
        shuffle_runs=arguments.shuffle_runs,
        bootstrap_samples=arguments.bootstrap_samples,
        fast_test=arguments.fast_test,
        fast_user_limit=arguments.fast_user_limit,
    )


def main() -> None:
    """Run the command-line experiment."""
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
