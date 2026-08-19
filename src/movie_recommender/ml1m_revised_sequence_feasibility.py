"""Revised multi-target sequence feasibility experiment for MovieLens 1M."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
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
import psutil
import torch
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
from movie_recommender.ml1m_temporal_eda import (
    _jensen_shannon,
    build_genre_evolution_features,
    build_user_temporal_features,
)


DEFAULT_OUTPUT_DIR: Final = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "ml1m_revised_sequence_feasibility"
)
NON_SEQUENTIAL_MODELS: Final = ("popularity", "item_knn", "matrix_factorization")
CORE_MODELS: Final = (
    "popularity",
    "item_knn",
    "matrix_factorization",
    "fpmc",
    "sasrec_original",
)
STATIC_FEATURES: Final = (
    "rating_count",
    "positive_interaction_count",
    "rating_mean",
    "rating_std",
    "genre_entropy",
    "mean_item_popularity",
    "median_item_popularity",
)
TEMPORAL_FEATURES: Final = (
    "history_span_days",
    "median_gap_minutes",
    "largest_day_ratio",
    "largest_session_ratio_2h",
    "session_count_2h",
    "genre_js_divergence",
)
SEQUENCE_FEATURES: Final = (
    "mean_adjacent_genre_similarity",
    "median_adjacent_genre_similarity",
    "adjacent_genre_similarity_std",
    "genre_transition_entropy",
    "next_item_similarity_k3",
    "next_item_similarity_k5",
    "repeated_genre_run_ratio",
    "mean_item_popularity_change",
    "popularity_transition_std",
    "recent_vs_long_term_genre_distance",
    "mean_transition_log_probability",
)


@dataclass(frozen=True)
class RevisedConfig:
    """Configuration for the revised experiment."""

    data_dir: Path = DEFAULT_ML1M_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    relevance_threshold: float = 4.0
    minimum_training_interactions: int = 5
    minimum_validation_targets: int = 2
    minimum_test_targets: int = 2
    holdout_fraction: float = 0.10
    random_seed: int = 760
    maximum_epochs: int = 5
    patience: int = 2
    batch_size: int = 128
    shuffle_runs: int = 3
    bootstrap_samples: int = 500
    clear_win_epsilon: float = 0.01
    fast_test: bool = False
    fast_user_limit: int = 300
    equal_timestamp_ordering_rule: str = "timestamp_movie_id_source"

    def serializable(self) -> dict[str, object]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        values["ordering_rule"] = self.equal_timestamp_ordering_rule
        values["split_rule"] = (
            "test=max(2,floor(10%)); validation=max(2,floor(10%)); "
            "remainder training with minimum 5"
        )
        values["evaluation"] = "full ranking over unseen catalog items; set-based test block"
        values["timestamp_interpretation"] = "rating-submission time, not watching time"
        return values


@dataclass
class RevisedSplit:
    """Chronological blocks and audit tables."""

    train_positive: pd.DataFrame
    validation_positive: pd.DataFrame
    test_positive: pd.DataFrame
    training_history: pd.DataFrame
    eligible_users: pd.DataFrame
    excluded_users: pd.DataFrame
    summary: pd.DataFrame
    split_table: pd.DataFrame
    full_ratings: pd.DataFrame
    ordering_columns: tuple[str, ...]


@dataclass
class RankingSet:
    """Set-based full-ranking candidates and relevant targets."""

    user_id: int
    candidate_movie_ids: np.ndarray
    relevant_movie_ids: set[int]


@dataclass(frozen=True)
class SASRecParameters:
    embedding_dim: int
    max_sequence_length: int
    blocks: int
    heads: int
    dropout: float
    learning_rate: float


class SASRec(torch.nn.Module):
    """Compact unidirectional self-attention sequential recommender."""

    def __init__(
        self,
        item_count: int,
        parameters: SASRecParameters,
    ) -> None:
        super().__init__()
        self.parameters_config = parameters
        self.item_embedding = torch.nn.Embedding(item_count + 1, parameters.embedding_dim, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(
            parameters.max_sequence_length, parameters.embedding_dim
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=parameters.embedding_dim,
            nhead=parameters.heads,
            dim_feedforward=parameters.embedding_dim * 4,
            dropout=parameters.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=parameters.blocks
        )
        self.layer_norm = torch.nn.LayerNorm(parameters.embedding_dim)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, item_sequences: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            item_sequences.shape[1], device=item_sequences.device
        ).unsqueeze(0)
        hidden = self.item_embedding(item_sequences) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                item_sequences.shape[1],
                item_sequences.shape[1],
                dtype=torch.bool,
                device=item_sequences.device,
            ),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=item_sequences.eq(0),
        )
        return self.layer_norm(hidden)

    def score_all(self, item_sequences: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(item_sequences)[:, -1, :]
        return hidden @ self.item_embedding.weight[1:].T


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic random seeds for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def revised_chronological_split(
    ratings: pd.DataFrame,
    config: RevisedConfig,
) -> RevisedSplit:
    """Create deterministic hybrid 80/10/10 positive-interaction blocks."""
    ordering_rules = {
        "timestamp_movie_id_source": ("Timestamp", "MovieID", "_source_order"),
        "timestamp_source": ("Timestamp", "_source_order"),
    }
    try:
        ordering_columns = ordering_rules[config.equal_timestamp_ordering_rule]
    except KeyError as error:
        raise ValueError(
            "equal_timestamp_ordering_rule must be "
            "'timestamp_movie_id_source' or 'timestamp_source'"
        ) from error
    full = ratings.reset_index(names="_source_order").sort_values(
        ["UserID", *ordering_columns], kind="mergesort"
    )
    full["_user_order_rank"] = full.groupby("UserID").cumcount()
    positives = full.loc[full["Rating"].ge(config.relevance_threshold)].copy()
    positive_counts = positives.groupby("UserID").size()
    all_users = np.sort(full["UserID"].unique())

    eligible_records: list[dict[str, int]] = []
    excluded_records: list[dict[str, object]] = []
    split_frames: list[pd.DataFrame] = []
    for user_id in all_users:
        count = int(positive_counts.get(user_id, 0))
        validation_count = max(
            config.minimum_validation_targets,
            int(np.floor(count * config.holdout_fraction)),
        )
        test_count = max(
            config.minimum_test_targets,
            int(np.floor(count * config.holdout_fraction)),
        )
        training_count = count - validation_count - test_count
        if training_count < config.minimum_training_interactions:
            excluded_records.append(
                {
                    "user_id": int(user_id),
                    "positive_interaction_count": count,
                    "exclusion_reason": "insufficient_for_hybrid_multi_target_split",
                }
            )
            continue
        eligible_records.append(
            {
                "user_id": int(user_id),
                "positive_interaction_count": count,
                "training_interaction_count": training_count,
                "validation_target_count": validation_count,
                "test_target_count": test_count,
            }
        )

    eligible_columns = [
        "user_id",
        "positive_interaction_count",
        "training_interaction_count",
        "validation_target_count",
        "test_target_count",
    ]
    eligible = pd.DataFrame(eligible_records, columns=eligible_columns)
    selected_user_ids = eligible["user_id"].to_numpy(dtype=np.int64)
    if config.fast_test and len(selected_user_ids) > config.fast_user_limit:
        rng = np.random.default_rng(config.random_seed)
        selected_user_ids = np.sort(
            rng.choice(selected_user_ids, config.fast_user_limit, replace=False)
        )
        eligible = eligible.loc[eligible["user_id"].isin(selected_user_ids)].copy()

    selected = positives.loc[positives["UserID"].isin(selected_user_ids)].copy()
    counts = selected.groupby("UserID")["MovieID"].transform("size")
    validation_counts = selected["UserID"].map(
        eligible.set_index("user_id")["validation_target_count"]
    )
    test_counts = selected["UserID"].map(
        eligible.set_index("user_id")["test_target_count"]
    )
    position = selected.groupby("UserID").cumcount()
    train_mask = position < counts - validation_counts - test_counts
    validation_mask = (position >= counts - validation_counts - test_counts) & (
        position < counts - test_counts
    )
    test_mask = position >= counts - test_counts
    train = selected.loc[train_mask].copy()
    validation = selected.loc[validation_mask].copy()
    test = selected.loc[test_mask].copy()
    for label, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        labelled = frame.copy()
        labelled["split"] = label
        split_frames.append(labelled)

    cutoff = train.groupby("UserID", as_index=False)["_user_order_rank"].max().rename(
        columns={"_user_order_rank": "_cutoff_rank"}
    )
    history = full.loc[full["UserID"].isin(selected_user_ids)].merge(
        cutoff, on="UserID", how="inner", validate="many_to_one"
    )
    history = history.loc[history["_user_order_rank"].le(history["_cutoff_rank"])].copy()
    history = history.drop(columns=["_cutoff_rank"])

    test_target_counts = eligible["test_target_count"]
    summary = pd.DataFrame(
        [
            {"metric": "total_users", "value": len(all_users)},
            {"metric": "eligible_users_full_dataset", "value": len(eligible_records)},
            {"metric": "users_selected_for_run", "value": len(eligible)},
            {"metric": "excluded_users", "value": len(excluded_records)},
            {"metric": "training_positive_interactions", "value": len(train)},
            {"metric": "validation_positive_interactions", "value": len(validation)},
            {"metric": "test_positive_interactions", "value": len(test)},
            {
                "metric": "minimum_test_targets_per_user",
                "value": (
                    int(test_target_counts.min()) if not eligible.empty else np.nan
                ),
            },
            {
                "metric": "median_test_targets_per_user",
                "value": (
                    float(test_target_counts.median()) if not eligible.empty else np.nan
                ),
            },
            {
                "metric": "maximum_test_targets_per_user",
                "value": (
                    int(test_target_counts.max()) if not eligible.empty else np.nan
                ),
            },
        ]
    )
    return RevisedSplit(
        train_positive=train,
        validation_positive=validation,
        test_positive=test,
        training_history=history,
        eligible_users=eligible,
        excluded_users=pd.DataFrame(excluded_records),
        summary=summary,
        split_table=pd.concat(split_frames, ignore_index=True),
        full_ratings=full,
        ordering_columns=ordering_columns,
    )


def make_sequences(
    frame: pd.DataFrame,
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
) -> list[np.ndarray]:
    """Convert positive interactions to stable zero-based item sequences."""
    ordering_columns = (
        ["_user_order_rank"]
        if "_user_order_rank" in frame.columns
        else ["Timestamp", "MovieID", "_source_order"]
    )
    grouped = {
        int(user_id): group.sort_values(ordering_columns, kind="mergesort")
        for user_id, group in frame.groupby("UserID", sort=False)
    }
    return [
        np.asarray(
            [item_to_index[int(item)] for item in grouped[int(user_id)]["MovieID"]],
            dtype=np.int32,
        )
        for user_id in user_ids
    ]


def concatenate_blocks(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> pd.DataFrame:
    """Combine chronological interaction blocks."""
    combined = pd.concat([first, second], ignore_index=False)
    ordering_columns = (
        ["UserID", "_user_order_rank"]
        if "_user_order_rank" in combined.columns
        else ["UserID", "Timestamp", "MovieID", "_source_order"]
    )
    return combined.sort_values(ordering_columns, kind="mergesort")


def build_ranking_sets(
    split: RevisedSplit,
    targets: pd.DataFrame,
    all_movie_ids: np.ndarray,
) -> tuple[dict[int, RankingSet], pd.DataFrame]:
    """Build full-ranking unseen-item candidate sets for target blocks."""
    full_by_user = {
        int(user_id): frame
        for user_id, frame in split.full_ratings.groupby("UserID", sort=False)
    }
    sets: dict[int, RankingSet] = {}
    rows: list[dict[str, object]] = []
    for user_id, target_group in targets.groupby("UserID", sort=True):
        ordered_targets = target_group.sort_values("_user_order_rank", kind="mergesort")
        first = ordered_targets.iloc[0]
        history = full_by_user[int(user_id)]
        observed = history.loc[
            history["_user_order_rank"].lt(first["_user_order_rank"]), "MovieID"
        ]
        relevant = set(int(item) for item in ordered_targets["MovieID"])
        observed_items = set(int(value) for value in observed)
        candidates = np.asarray(
            [
                item
                for item in all_movie_ids
                if int(item) not in observed_items
                or int(item) in relevant
            ],
            dtype=np.int64,
        )
        if not relevant.issubset(set(int(item) for item in candidates)):
            raise RuntimeError(f"Target omission in candidate set for user {user_id}")
        sets[int(user_id)] = RankingSet(int(user_id), candidates, relevant)
        rows.append(
            {
                "user_id": int(user_id),
                "relevant_movie_ids": "|".join(str(item) for item in sorted(relevant)),
                "candidate_movie_ids": "|".join(str(item) for item in candidates),
                "number_of_test_targets": len(relevant),
                "number_of_candidates": len(candidates),
            }
        )
    return sets, pd.DataFrame(rows)


def set_ranking_metrics(
    scores: np.ndarray,
    candidate_movie_ids: np.ndarray,
    relevant_movie_ids: set[int],
    k: int = 10,
) -> tuple[dict[str, float], np.ndarray]:
    """Calculate set-based Top-K metrics without arbitrary score ties."""
    order = np.lexsort((candidate_movie_ids, -scores))
    ranked_items = candidate_movie_ids[order]
    top_items = ranked_items[:k]
    relevance = np.asarray(
        [int(item) in relevant_movie_ids for item in ranked_items], dtype=float
    )
    top_relevance = relevance[:k]
    hits = int(top_relevance.sum())
    precision = hits / k
    recall = hits / len(relevant_movie_ids)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((top_relevance * discounts).sum())
    ideal_hits = min(len(relevant_movie_ids), k)
    idcg = float(discounts[:ideal_hits].sum())
    ndcg = dcg / idcg if idcg else 0.0
    hit_rate = float(hits > 0)
    relevant_ranks = np.flatnonzero(relevance) + 1
    mrr = (
        float(1.0 / relevant_ranks[0])
        if len(relevant_ranks) and relevant_ranks[0] <= k
        else 0.0
    )
    cumulative_precision = np.cumsum(top_relevance) / np.arange(1, k + 1)
    average_precision = float(
        (cumulative_precision * top_relevance).sum() / ideal_hits
    )
    return (
        {
            "precision_10": precision,
            "recall_10": recall,
            "ndcg_10": ndcg,
            "hit_rate_10": hit_rate,
            "mrr_10": mrr,
            "map_10": average_precision,
        },
        top_items,
    )


def evaluate_full_ranker(
    model_name: str,
    ranking_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    scorer: object,
) -> pd.DataFrame:
    """Evaluate a callable scorer over each user's full unseen candidate set."""
    rows: list[dict[str, object]] = []
    for user_index, user_id in enumerate(user_ids):
        ranking_set = ranking_sets[int(user_id)]
        candidate_indices = np.asarray(
            [item_to_index[int(item)] for item in ranking_set.candidate_movie_ids],
            dtype=np.int32,
        )
        scores = np.asarray(scorer(user_index, candidate_indices), dtype=float)
        metrics, top_items = set_ranking_metrics(
            scores, ranking_set.candidate_movie_ids, ranking_set.relevant_movie_ids
        )
        rows.append(
            {
                "user_id": int(user_id),
                "model_name": model_name,
                **metrics,
                "number_of_test_targets": len(ranking_set.relevant_movie_ids),
                "number_of_candidates": len(ranking_set.candidate_movie_ids),
                "top_10_movie_ids": "|".join(str(item) for item in top_items),
            }
        )
    return pd.DataFrame(rows)


def pad_sequences(
    sequences: list[np.ndarray],
    max_length: int,
) -> np.ndarray:
    """Left-pad zero-based item sequences after shifting item IDs by one."""
    padded = np.zeros((len(sequences), max_length), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        shifted = sequence[-max_length:] + 1
        padded[index, -len(shifted) :] = shifted
    return padded


def training_arrays(
    sequences: list[np.ndarray],
    max_length: int,
    item_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create SASRec input, positive-next, and negative-next arrays."""
    inputs = np.zeros((len(sequences), max_length), dtype=np.int64)
    positives = np.zeros_like(inputs)
    negatives = np.zeros_like(inputs)
    rng = np.random.default_rng(seed)
    for user_index, sequence in enumerate(sequences):
        input_items = sequence[:-1][-max_length:] + 1
        positive_items = sequence[1:][-max_length:] + 1
        length = len(input_items)
        inputs[user_index, -length:] = input_items
        positives[user_index, -length:] = positive_items
        seen = set(int(item) + 1 for item in sequence)
        for position in range(max_length - length, max_length):
            negative = int(rng.integers(1, item_count + 1))
            while negative in seen:
                negative = int(rng.integers(1, item_count + 1))
            negatives[user_index, position] = negative
    return inputs, positives, negatives


def sasrec_score_matrix(
    model: SASRec,
    sequences: list[np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score every catalog item for every user in batches."""
    model.eval()
    inputs = pad_sequences(
        sequences, model.parameters_config.max_sequence_length
    )
    score_batches = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size], dtype=torch.long, device=device
            )
            score_batches.append(model.score_all(batch).cpu().numpy())
    return np.vstack(score_batches)


def train_sasrec(
    sequences: list[np.ndarray],
    validation_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_count: int,
    parameters: SASRecParameters,
    config: RevisedConfig,
    device: torch.device,
    run_label: str,
    validate_each_epoch: bool = True,
    fixed_epochs: int | None = None,
) -> tuple[SASRec, list[dict[str, object]], int]:
    """Train SASRec with validation NDCG early stopping."""
    set_deterministic_seed(config.random_seed)
    model = SASRec(item_count, parameters).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters.learning_rate, weight_decay=1e-6
    )
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
    maximum_epochs = fixed_epochs or (
        min(config.maximum_epochs, 2) if config.fast_test else config.maximum_epochs
    )
    history: list[dict[str, object]] = []
    best_score = -np.inf
    best_epoch = maximum_epochs
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, maximum_epochs + 1):
        started = time.perf_counter()
        inputs, positives, negatives = training_arrays(
            sequences,
            parameters.max_sequence_length,
            item_count,
            config.random_seed + epoch,
        )
        rng = np.random.default_rng(config.random_seed + epoch)
        order = rng.permutation(len(sequences))
        model.train()
        losses = []
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            input_batch = torch.as_tensor(
                inputs[indices], dtype=torch.long, device=device
            )
            positive_batch = torch.as_tensor(
                positives[indices], dtype=torch.long, device=device
            )
            negative_batch = torch.as_tensor(
                negatives[indices], dtype=torch.long, device=device
            )
            mask = positive_batch.gt(0)
            hidden = model(input_batch)
            positive_vectors = model.item_embedding(positive_batch)
            negative_vectors = model.item_embedding(negative_batch)
            positive_logits = (hidden * positive_vectors).sum(dim=-1)
            negative_logits = (hidden * negative_vectors).sum(dim=-1)
            positive_loss = loss_function(
                positive_logits, torch.ones_like(positive_logits)
            )
            negative_loss = loss_function(
                negative_logits, torch.zeros_like(negative_logits)
            )
            loss = ((positive_loss + negative_loss) * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_ndcg = np.nan
        if validate_each_epoch:
            score_matrix = sasrec_score_matrix(
                model, sequences, config.batch_size, device
            )
            validation_metrics = evaluate_full_ranker(
                "sasrec_validation",
                validation_sets,
                user_ids,
                item_to_index,
                lambda user_index, candidate_indices: score_matrix[
                    user_index, candidate_indices
                ],
            )
            validation_ndcg = float(validation_metrics["ndcg_10"].mean())
            if validation_ndcg > best_score + 1e-5:
                best_score = validation_ndcg
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(
            {
                "run_label": run_label,
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_ndcg_10": validation_ndcg,
                "epoch_runtime_seconds": time.perf_counter() - started,
            }
        )
        if validate_each_epoch and stale_epochs >= config.patience:
            break

    if validate_each_epoch:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum_epochs
    return model, history, best_epoch


def genre_assets(
    movies: pd.DataFrame,
) -> tuple[dict[int, set[str]], dict[int, np.ndarray], list[str]]:
    """Create genre sets and equal-weight genre vectors."""
    genres = sorted(
        movies["Genres"].str.split("|").explode().unique().tolist()
    )
    genre_index = {genre: index for index, genre in enumerate(genres)}
    sets: dict[int, set[str]] = {}
    vectors: dict[int, np.ndarray] = {}
    for row in movies.itertuples(index=False):
        movie_genres = set(str(row.Genres).split("|"))
        vector = np.zeros(len(genres), dtype=float)
        for genre in movie_genres:
            vector[genre_index[genre]] = 1.0 / len(movie_genres)
        sets[int(row.MovieID)] = movie_genres
        vectors[int(row.MovieID)] = vector
    return sets, vectors, genres


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else 0.0


def sequence_predictability_features(
    split: RevisedSplit,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe static, temporal, and sequence regularity features."""
    history = split.training_history.drop(
        columns=["_source_order"], errors="ignore"
    ).reset_index(drop=True)
    history["RatingTime"] = pd.to_datetime(
        history["Timestamp"], unit="s", utc=True, errors="raise"
    )
    temporal = build_user_temporal_features(history)
    genre_evolution = build_genre_evolution_features(history, movies)
    feature_frame = temporal[
        [
            "user_id",
            "history_span_days",
            "median_gap_minutes",
            "largest_day_ratio",
            "largest_session_ratio_2h",
            "session_count_2h",
        ]
    ].merge(
        genre_evolution[["user_id", "genre_js_divergence", "overall_genre_entropy"]],
        on="user_id",
        validate="one_to_one",
    )
    feature_frame = feature_frame.rename(
        columns={"overall_genre_entropy": "genre_entropy"}
    )

    static = history.groupby("UserID").agg(
        rating_count=("MovieID", "size"),
        rating_mean=("Rating", "mean"),
        rating_std=("Rating", "std"),
    )
    positive_counts = split.train_positive.groupby("UserID").size()
    static["positive_interaction_count"] = positive_counts
    static["rating_std"] = static["rating_std"].fillna(0)
    item_popularity = split.train_positive.groupby("MovieID").size()
    positive_with_popularity = split.train_positive.assign(
        item_popularity=split.train_positive["MovieID"].map(item_popularity)
    )
    popularity_features = positive_with_popularity.groupby("UserID")[
        "item_popularity"
    ].agg(
        mean_item_popularity="mean",
        median_item_popularity="median",
    )
    static = static.join(popularity_features)
    static.index.name = "user_id"
    feature_frame = feature_frame.merge(
        static.reset_index(), on="user_id", validate="one_to_one"
    )

    genre_sets, genre_vectors, genres = genre_assets(movies)
    global_transition = np.ones((len(genres), len(genres)), dtype=float)
    genre_index = {genre: index for index, genre in enumerate(genres)}
    sequence_by_user = {
        int(user_id): group.sort_values(
            ["Timestamp", "MovieID", "_source_order"], kind="mergesort"
        )["MovieID"].astype(int).tolist()
        for user_id, group in split.train_positive.groupby("UserID", sort=False)
    }
    for sequence in sequence_by_user.values():
        for previous, current in zip(sequence[:-1], sequence[1:], strict=True):
            previous_genres = genre_sets[previous]
            current_genres = genre_sets[current]
            weight = 1.0 / (len(previous_genres) * len(current_genres))
            for source in previous_genres:
                for target in current_genres:
                    global_transition[genre_index[source], genre_index[target]] += weight
    global_probabilities = global_transition / global_transition.sum(axis=1, keepdims=True)

    rows: list[dict[str, float | int]] = []
    for user_id, sequence in sequence_by_user.items():
        adjacent_similarities = []
        popularity_changes = []
        local_transition = np.zeros_like(global_transition)
        log_probabilities = []
        for previous, current in zip(sequence[:-1], sequence[1:], strict=True):
            previous_genres = genre_sets[previous]
            current_genres = genre_sets[current]
            union = previous_genres | current_genres
            adjacent_similarities.append(
                len(previous_genres & current_genres) / len(union)
            )
            popularity_changes.append(
                float(item_popularity.get(current, 0) - item_popularity.get(previous, 0))
            )
            weight = 1.0 / (len(previous_genres) * len(current_genres))
            transition_probability = 0.0
            for source in previous_genres:
                for target in current_genres:
                    source_index = genre_index[source]
                    target_index = genre_index[target]
                    local_transition[source_index, target_index] += weight
                    transition_probability += (
                        weight * global_probabilities[source_index, target_index]
                    )
            log_probabilities.append(np.log(max(transition_probability, 1e-12)))

        transition_distribution = local_transition.ravel()
        transition_distribution /= transition_distribution.sum()
        positive_transition = transition_distribution[transition_distribution > 0]
        transition_entropy = -float(
            (positive_transition * np.log2(positive_transition)).sum()
        ) / np.log2(len(genres) ** 2)

        coherence: dict[int, float] = {}
        for window in (3, 5):
            values = []
            for position in range(window, len(sequence)):
                previous_vector = np.mean(
                    [genre_vectors[item] for item in sequence[position - window : position]],
                    axis=0,
                )
                values.append(_cosine(previous_vector, genre_vectors[sequence[position]]))
            coherence[window] = float(np.mean(values)) if values else 0.0

        recent_count = min(5, max(1, len(sequence) // 5))
        recent = np.mean(
            [genre_vectors[item] for item in sequence[-recent_count:]], axis=0
        )
        long_term_items = sequence[:-recent_count] or sequence
        long_term = np.mean(
            [genre_vectors[item] for item in long_term_items], axis=0
        )
        recent /= recent.sum()
        long_term /= long_term.sum()
        recent_distance = float(
            _jensen_shannon(recent[None, :], long_term[None, :])[0]
        )
        rows.append(
            {
                "user_id": user_id,
                "mean_adjacent_genre_similarity": float(np.mean(adjacent_similarities)),
                "median_adjacent_genre_similarity": float(
                    np.median(adjacent_similarities)
                ),
                "adjacent_genre_similarity_std": float(
                    np.std(adjacent_similarities, ddof=1)
                    if len(adjacent_similarities) > 1
                    else 0.0
                ),
                "genre_transition_entropy": transition_entropy,
                "next_item_similarity_k3": coherence[3],
                "next_item_similarity_k5": coherence[5],
                "repeated_genre_run_ratio": float(
                    np.mean(np.asarray(adjacent_similarities) > 0)
                ),
                "mean_item_popularity_change": float(np.mean(popularity_changes)),
                "popularity_transition_std": float(
                    np.std(popularity_changes, ddof=1)
                    if len(popularity_changes) > 1
                    else 0.0
                ),
                "recent_vs_long_term_genre_distance": recent_distance,
                "mean_transition_log_probability": float(
                    np.mean(log_probabilities)
                ),
            }
        )
    feature_frame = feature_frame.merge(
        pd.DataFrame(rows), on="user_id", validate="one_to_one"
    )
    if feature_frame.isna().any().any():
        raise RuntimeError(
            f"Feature missingness detected: "
            f"{feature_frame.columns[feature_frame.isna().any()].tolist()}"
        )
    return feature_frame


def feature_quality(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report missingness, skewness, variance, and feature correlations."""
    selected = list(STATIC_FEATURES + TEMPORAL_FEATURES + SEQUENCE_FEATURES)
    rows = []
    for feature in selected:
        values = features[feature]
        rows.append(
            {
                "feature": feature,
                "feature_block": (
                    "static"
                    if feature in STATIC_FEATURES
                    else "temporal"
                    if feature in TEMPORAL_FEATURES
                    else "sequence_predictability"
                ),
                "missing_count": int(values.isna().sum()),
                "unique_values": int(values.nunique()),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "skewness": values.skew(),
                "near_zero_variance": bool(values.std() < 1e-8),
            }
        )
    return pd.DataFrame(rows), features[selected].corr(method="spearman")


def fit_baselines_and_fpmc(
    split: RevisedSplit,
    movies: pd.DataFrame,
    validation_sets: dict[int, RankingSet],
    test_sets: dict[int, RankingSet],
    config: RevisedConfig,
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]], list[np.ndarray], dict[int, int], np.ndarray]:
    """Tune and evaluate compatible non-sequential baselines and FPMC."""
    user_ids = split.eligible_users["user_id"].to_numpy(dtype=np.int64)
    movie_ids = np.sort(movies["MovieID"].unique()).astype(np.int64)
    item_to_index = {int(item): index for index, item in enumerate(movie_ids)}
    train_sequences = make_sequences(split.train_positive, user_ids, item_to_index)
    final_frame = concatenate_blocks(
        split.train_positive, split.validation_positive
    )
    final_sequences = make_sequences(final_frame, user_ids, item_to_index)
    train_matrix = build_interaction_matrix(train_sequences, len(movie_ids))
    final_matrix = build_interaction_matrix(final_sequences, len(movie_ids))
    validation_rows: list[dict[str, object]] = []

    knn_options = [30] if config.fast_test else [50, 100]
    best_knn = (-np.inf, knn_options[0])
    for neighbors in knn_options:
        similarity = train_item_knn(train_matrix, neighbors)
        metrics = evaluate_full_ranker(
            f"item_knn_k{neighbors}",
            validation_sets,
            user_ids,
            item_to_index,
            lambda user_index, candidates, sim=similarity: (
                sim[candidates].dot(train_matrix[user_index].T).toarray().ravel()
            ),
        )
        score = float(metrics["ndcg_10"].mean())
        validation_rows.append(
            {
                "model": "item_knn",
                "parameters": json.dumps({"neighbors": neighbors}),
                "validation_ndcg_10": score,
            }
        )
        if score > best_knn[0]:
            best_knn = (score, neighbors)

    svd_options = [16] if config.fast_test else [32, 64]
    best_svd = (-np.inf, svd_options[0])
    for dimensions in svd_options:
        model = TruncatedSVD(
            n_components=dimensions, random_state=config.random_seed
        )
        embeddings = model.fit_transform(train_matrix)
        metrics = evaluate_full_ranker(
            f"mf_d{dimensions}",
            validation_sets,
            user_ids,
            item_to_index,
            lambda user_index, candidates, emb=embeddings, components=model.components_: (
                emb[user_index] @ components[:, candidates]
            ),
        )
        score = float(metrics["ndcg_10"].mean())
        validation_rows.append(
            {
                "model": "matrix_factorization",
                "parameters": json.dumps({"dimensions": dimensions}),
                "validation_ndcg_10": score,
            }
        )
        if score > best_svd[0]:
            best_svd = (score, dimensions)

    fpmc_epochs = 1 if config.fast_test else min(3, config.maximum_epochs)
    fpmc_train = FPMC(
        len(user_ids), len(movie_ids), 16 if config.fast_test else 24, 0.03, 0.002, config.random_seed
    ).fit(train_sequences, fpmc_epochs)
    previous_train = np.asarray([sequence[-1] for sequence in train_sequences])
    best_weight = (-np.inf, 0.5)
    for weight in (0.5, 1.0):
        metrics = evaluate_full_ranker(
            f"fpmc_w{weight}",
            validation_sets,
            user_ids,
            item_to_index,
            lambda user_index, candidates, selected_weight=weight: fpmc_train.score(
                user_index, int(previous_train[user_index]), candidates, selected_weight
            ),
        )
        score = float(metrics["ndcg_10"].mean())
        validation_rows.append(
            {
                "model": "fpmc",
                "parameters": json.dumps({"transition_weight": weight}),
                "validation_ndcg_10": score,
            }
        )
        if score > best_weight[0]:
            best_weight = (score, weight)

    popularity = np.asarray(final_matrix.sum(axis=0)).ravel()
    popularity_metrics = evaluate_full_ranker(
        "popularity",
        test_sets,
        user_ids,
        item_to_index,
        lambda _user, candidates: popularity[candidates],
    )
    final_similarity = train_item_knn(final_matrix, best_knn[1])
    knn_metrics = evaluate_full_ranker(
        "item_knn",
        test_sets,
        user_ids,
        item_to_index,
        lambda user_index, candidates: (
            final_similarity[candidates]
            .dot(final_matrix[user_index].T)
            .toarray()
            .ravel()
        ),
    )
    final_svd = TruncatedSVD(
        n_components=best_svd[1], random_state=config.random_seed
    )
    final_embeddings = final_svd.fit_transform(final_matrix)
    mf_metrics = evaluate_full_ranker(
        "matrix_factorization",
        test_sets,
        user_ids,
        item_to_index,
        lambda user_index, candidates: (
            final_embeddings[user_index] @ final_svd.components_[:, candidates]
        ),
    )
    final_fpmc = FPMC(
        len(user_ids), len(movie_ids), 16 if config.fast_test else 24, 0.03, 0.002, config.random_seed
    ).fit(final_sequences, fpmc_epochs)
    previous_final = np.asarray([sequence[-1] for sequence in final_sequences])
    fpmc_metrics = evaluate_full_ranker(
        "fpmc",
        test_sets,
        user_ids,
        item_to_index,
        lambda user_index, candidates: final_fpmc.score(
            user_index, int(previous_final[user_index]), candidates, best_weight[1]
        ),
    )
    selected = {
        "item_knn": {"neighbors": best_knn[1], "similarity": "cosine"},
        "matrix_factorization": {
            "algorithm": "TruncatedSVD",
            "dimensions": best_svd[1],
        },
        "fpmc": {
            "transition_weight": best_weight[1],
            "epochs": fpmc_epochs,
            "dimensions": 16 if config.fast_test else 24,
        },
    }
    return (
        pd.concat(
            [popularity_metrics, knn_metrics, mf_metrics, fpmc_metrics],
            ignore_index=True,
        ),
        selected,
        validation_rows,
        final_sequences,
        item_to_index,
        movie_ids,
    )


def tune_and_evaluate_sasrec(
    split: RevisedSplit,
    validation_sets: dict[int, RankingSet],
    test_sets: dict[int, RankingSet],
    user_ids: np.ndarray,
    item_to_index: dict[int, int],
    movie_ids: np.ndarray,
    final_sequences: list[np.ndarray],
    config: RevisedConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Tune SASRec, evaluate original order, and run shuffled controls."""
    train_sequences = make_sequences(
        split.train_positive, user_ids, item_to_index
    )
    parameter_options = [
        SASRecParameters(32, 50, 1, 2, 0.1, 0.001),
        SASRecParameters(64, 100, 2, 4, 0.2, 0.0005),
    ]
    if config.fast_test:
        parameter_options = parameter_options[:1]
    training_rows: list[dict[str, object]] = []
    validation_results = []
    best = (-np.inf, parameter_options[0], 1)
    for option_index, parameters in enumerate(parameter_options):
        model, history, best_epoch = train_sasrec(
            train_sequences,
            validation_sets,
            user_ids,
            item_to_index,
            len(movie_ids),
            parameters,
            config,
            device,
            f"tuning_config_{option_index}",
        )
        training_rows.extend(history)
        best_score = max(row["validation_ndcg_10"] for row in history)
        validation_results.append(
            {
                "model": "sasrec",
                "parameters": json.dumps(asdict(parameters)),
                "best_epoch": best_epoch,
                "validation_ndcg_10": best_score,
            }
        )
        if best_score > best[0]:
            best = (best_score, parameters, best_epoch)
        del model

    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    original_model, history, _ = train_sasrec(
        final_sequences,
        validation_sets,
        user_ids,
        item_to_index,
        len(movie_ids),
        best[1],
        config,
        device,
        "sasrec_original_final",
        validate_each_epoch=False,
        fixed_epochs=best[2],
    )
    training_rows.extend(history)
    torch.save(original_model.state_dict(), checkpoints / "sasrec_original.pt")
    inference_started = time.perf_counter()
    original_scores = sasrec_score_matrix(
        original_model, final_sequences, config.batch_size, device
    )
    inference_seconds = time.perf_counter() - inference_started
    original_metrics = evaluate_full_ranker(
        "sasrec_original",
        test_sets,
        user_ids,
        item_to_index,
        lambda user_index, candidates: original_scores[user_index, candidates],
    )

    shuffle_metrics = []
    shuffle_runs = 1 if config.fast_test else config.shuffle_runs
    for run_index in range(shuffle_runs):
        seed = config.random_seed + 20_000 + run_index
        shuffled_train = []
        for user_index, sequence in enumerate(train_sequences):
            rng = np.random.default_rng(seed + user_index * 1_009)
            shuffled_train.append(rng.permutation(sequence).astype(np.int32))
        validation_sequences = make_sequences(
            split.validation_positive, user_ids, item_to_index
        )
        shuffled_final = [
            np.concatenate([train_sequence, validation_sequence])
            for train_sequence, validation_sequence in zip(
                shuffled_train, validation_sequences, strict=True
            )
        ]
        model, history, _ = train_sasrec(
            shuffled_final,
            validation_sets,
            user_ids,
            item_to_index,
            len(movie_ids),
            best[1],
            config,
            device,
            f"sasrec_shuffled_{seed}",
            validate_each_epoch=False,
            fixed_epochs=best[2],
        )
        training_rows.extend(history)
        torch.save(model.state_dict(), checkpoints / f"sasrec_shuffled_{seed}.pt")
        scores = sasrec_score_matrix(
            model, shuffled_final, config.batch_size, device
        )
        metrics = evaluate_full_ranker(
            f"sasrec_shuffled_seed_{seed}",
            test_sets,
            user_ids,
            item_to_index,
            lambda user_index, candidates, score_values=scores: score_values[
                user_index, candidates
            ],
        )
        metrics["shuffle_seed"] = seed
        shuffle_metrics.append(metrics)
        del model, scores

    selected = {
        "sasrec": {
            **asdict(best[1]),
            "best_epoch": best[2],
            "validation_ndcg_10": best[0],
            "device": str(device),
            "shuffle_runs": shuffle_runs,
            "inference_runtime_seconds": inference_seconds,
            "validation_results": validation_results,
        }
    }
    return (
        pd.concat(
            [original_metrics]
            + [frame.drop(columns="shuffle_seed") for frame in shuffle_metrics],
            ignore_index=True,
        ),
        pd.concat(shuffle_metrics, ignore_index=True),
        selected,
        pd.DataFrame(training_rows),
    )


def bootstrap_interval(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    )
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def aggregate_metrics(
    per_user: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    """Aggregate required metrics with user bootstrap intervals."""
    rows = []
    metric_names = (
        "precision_10",
        "recall_10",
        "ndcg_10",
        "hit_rate_10",
        "mrr_10",
        "map_10",
    )
    for model_name, frame in per_user.groupby("model_name", sort=False):
        for metric in metric_names:
            values = frame[metric].to_numpy(dtype=float)
            lower, upper = bootstrap_interval(values, samples, seed + len(rows))
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


def build_performance_matrix(
    per_user: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Create tie-aware model advantage and oracle columns."""
    matrix = per_user.pivot(
        index="user_id", columns="model_name", values="ndcg_10"
    ).reset_index()
    shuffled_columns = [
        column for column in matrix if column.startswith("sasrec_shuffled_seed_")
    ]
    matrix["sasrec_shuffled_mean"] = matrix[shuffled_columns].mean(axis=1)
    matrix["sequence_order_gain_user"] = (
        matrix["sasrec_original"] - matrix["sasrec_shuffled_mean"]
    )
    non_sequential = matrix[list(NON_SEQUENTIAL_MODELS)]
    matrix["best_non_sequential_ndcg"] = non_sequential.max(axis=1)
    matrix["best_non_sequential_model"] = non_sequential.idxmax(axis=1)
    matrix["best_sequential_ndcg"] = matrix[["fpmc", "sasrec_original"]].max(axis=1)
    matrix["best_sequential_model"] = matrix[
        ["fpmc", "sasrec_original"]
    ].idxmax(axis=1)
    matrix["delta_ndcg_strong_seq"] = (
        matrix["sasrec_original"] - matrix["best_non_sequential_ndcg"]
    )
    matrix["delta_ndcg_fpmc"] = matrix["fpmc"] - matrix["best_non_sequential_ndcg"]
    core = matrix[list(CORE_MODELS)]
    maximum = core.max(axis=1)
    matrix["best_model_tie_count"] = core.eq(maximum, axis=0).sum(axis=1)
    matrix["best_model"] = np.where(
        matrix["best_model_tie_count"].gt(1), "tie", core.idxmax(axis=1)
    )
    matrix["all_zero_outcome"] = core.eq(0).all(axis=1)
    matrix["oracle_ndcg"] = maximum
    matrix["restricted_oracle_ndcg"] = matrix[
        ["best_non_sequential_ndcg", "sasrec_original"]
    ].max(axis=1)
    return matrix.merge(features, on="user_id", validate="one_to_one")


def transformed_features(
    frame: pd.DataFrame, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    """Apply documented log transforms to skewed non-negative features."""
    transformed = frame[list(feature_names)].copy()
    log_features = {
        "rating_count",
        "positive_interaction_count",
        "history_span_days",
        "median_gap_minutes",
        "session_count_2h",
        "mean_item_popularity",
        "median_item_popularity",
        "popularity_transition_std",
    }
    for feature in feature_names:
        if feature in log_features:
            transformed[feature] = np.log1p(transformed[feature].clip(lower=0))
    return transformed


def association_models(
    advantage: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Compare static, temporal, and sequence feature blocks with ridge CV."""
    blocks = {
        "A_static": STATIC_FEATURES,
        "B_static_temporal": STATIC_FEATURES + TEMPORAL_FEATURES,
        "C_static_temporal_sequence": STATIC_FEATURES
        + TEMPORAL_FEATURES
        + SEQUENCE_FEATURES,
    }
    target = advantage["delta_ndcg_strong_seq"].to_numpy(dtype=float)
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for block_name, feature_names in blocks.items():
        values = transformed_features(advantage, feature_names).to_numpy()
        predictions = np.zeros(len(values))
        for train_indices, test_indices in folds.split(values):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(values[train_indices], target[train_indices])
            predictions[test_indices] = model.predict(values[test_indices])
        final_scaler = StandardScaler()
        scaled = final_scaler.fit_transform(values)
        final_model = Ridge(alpha=1.0).fit(scaled, target)
        rows.append(
            {
                "result_type": "cross_validation",
                "feature_block": block_name,
                "term": "overall",
                "value": r2_score(target, predictions),
                "metric": "r2",
            }
        )
        rows.append(
            {
                "result_type": "cross_validation",
                "feature_block": block_name,
                "term": "overall",
                "value": mean_absolute_error(target, predictions),
                "metric": "mae",
            }
        )
        rows.extend(
            {
                "result_type": "ridge_coefficient",
                "feature_block": block_name,
                "term": feature,
                "value": coefficient,
                "metric": "standardized_coefficient",
            }
            for feature, coefficient in zip(
                feature_names, final_model.coef_, strict=True
            )
        )
    return pd.DataFrame(rows), blocks


def classification_models(
    advantage: pd.DataFrame,
    blocks: dict[str, tuple[str, ...]],
    epsilon: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate tie-aware sequential-win classification targets."""
    delta = advantage["delta_ndcg_strong_seq"]
    target_definitions = {
        f"clear_win_epsilon_{epsilon:g}": pd.Series(
            np.where(delta.ge(epsilon), 1, np.where(delta.le(-epsilon), 0, np.nan)),
            index=advantage.index,
        ),
        f"clear_win_epsilon_{max(0.02, epsilon * 2):g}": pd.Series(
            np.where(
                delta.ge(max(0.02, epsilon * 2)),
                1,
                np.where(delta.le(-max(0.02, epsilon * 2)), 0, np.nan),
            ),
            index=advantage.index,
        ),
        "sequential_beneficial_strict": delta.gt(0).astype(float),
    }
    classifiers = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            ),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    rows = []
    for target_name, target_series in target_definitions.items():
        retained = target_series.notna()
        y = target_series.loc[retained].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        minimum_class_count = int(np.bincount(y).min())
        if minimum_class_count < 2:
            continue
        folds = StratifiedKFold(
            n_splits=min(5, minimum_class_count),
            shuffle=True,
            random_state=seed,
        )
        for block_name, feature_names in blocks.items():
            x = transformed_features(
                advantage.loc[retained], feature_names
            ).to_numpy()
            for model_name, estimator in classifiers.items():
                predictions = np.zeros(len(y), dtype=int)
                probabilities = np.zeros(len(y), dtype=float)
                for train_indices, test_indices in folds.split(x, y):
                    model = clone(estimator)
                    model.fit(x[train_indices], y[train_indices])
                    predictions[test_indices] = model.predict(x[test_indices])
                    probabilities[test_indices] = model.predict_proba(
                        x[test_indices]
                    )[:, 1]
                metrics = {
                    "balanced_accuracy": balanced_accuracy_score(y, predictions),
                    "macro_f1": f1_score(y, predictions, average="macro"),
                    "roc_auc": roc_auc_score(y, probabilities),
                    "sequential_win_precision": precision_score(
                        y, predictions, zero_division=0
                    ),
                    "sequential_win_recall": recall_score(
                        y, predictions, zero_division=0
                    ),
                    "brier_score": brier_score_loss(y, probabilities),
                }
                rows.extend(
                    {
                        "target": target_name,
                        "feature_block": block_name,
                        "model_name": model_name,
                        "metric": metric,
                        "value": value,
                        "users_retained": len(y),
                        "positive_users": int(y.sum()),
                        "ambiguous_users_excluded": int((~retained).sum()),
                    }
                    for metric, value in metrics.items()
                )
    return pd.DataFrame(rows)


def add_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Add quantile groups for required analyses."""
    grouped = frame.copy()
    labels = ["low", "medium", "high"]
    features = (
        "rating_count",
        "largest_day_ratio",
        "history_span_days",
        "mean_adjacent_genre_similarity",
        "genre_transition_entropy",
        "recent_vs_long_term_genre_distance",
        "sequence_order_gain_user",
    )
    for feature in features:
        ranks = grouped[feature].rank(method="average", pct=True)
        grouped[f"{feature}_group"] = pd.cut(
            ranks,
            bins=[0, 1 / 3, 2 / 3, 1],
            labels=labels,
            include_lowest=True,
        )
    return grouped


def group_summary(
    advantage: pd.DataFrame,
    per_user: pd.DataFrame,
    epsilon: float,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize model metrics and advantage by required user groups."""
    metric_matrix = per_user.pivot(
        index="user_id", columns="model_name", values="ndcg_10"
    )
    group_columns = [
        column for column in advantage if column.endswith("_group")
    ]
    rows = []
    for group_column in group_columns:
        for label, group in advantage.groupby(group_column, observed=True):
            delta = group["delta_ndcg_strong_seq"].to_numpy()
            lower, upper = bootstrap_interval(delta, samples, seed + len(rows))
            for model_name in CORE_MODELS:
                values = metric_matrix.loc[group["user_id"], model_name].to_numpy()
                rows.append(
                    {
                        "group_variable": group_column,
                        "group_label": str(label),
                        "model_name": model_name,
                        "user_count": len(group),
                        "small_group_warning": len(group) < 30,
                        "mean_ndcg_10": values.mean(),
                        "median_ndcg_10": np.median(values),
                        "mean_delta_ndcg_strong_seq": delta.mean(),
                        "median_delta_ndcg_strong_seq": np.median(delta),
                        "delta_ci_95_lower": lower,
                        "delta_ci_95_upper": upper,
                        "clear_sequential_win_percentage": (
                            group["delta_ndcg_strong_seq"].ge(epsilon).mean() * 100
                        ),
                        "tie_percentage": (
                            group["best_model_tie_count"].gt(1).mean() * 100
                        ),
                        "all_zero_percentage": (
                            group["all_zero_outcome"].mean() * 100
                        ),
                    }
                )
    return pd.DataFrame(rows)


def oracle_and_ties(
    advantage: pd.DataFrame,
    overall: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate unrestricted/restricted oracle headroom and tie reductions."""
    ndcg = overall.loc[overall["metric"].eq("ndcg_10")].set_index("model_name")[
        "mean"
    ]
    best_single = ndcg.loc[list(CORE_MODELS)].idxmax()
    best_single_value = float(ndcg[best_single])
    restricted_base = float(
        ndcg.loc[list(NON_SEQUENTIAL_MODELS)].max()
    )
    oracle_rows = [
        {
            "oracle": "all_core_models",
            "best_single_model": best_single,
            "best_single_ndcg_10": best_single_value,
            "oracle_ndcg_10": advantage["oracle_ndcg"].mean(),
            "absolute_headroom": advantage["oracle_ndcg"].mean() - best_single_value,
            "relative_headroom_percentage": (
                (advantage["oracle_ndcg"].mean() - best_single_value)
                / best_single_value
                * 100
            ),
            "users_benefiting_percentage": (
                advantage["oracle_ndcg"].gt(advantage[best_single]).mean() * 100
            ),
        },
        {
            "oracle": "strong_sequential_vs_best_non_sequential",
            "best_single_model": ndcg.loc[
                list(NON_SEQUENTIAL_MODELS) + ["sasrec_original"]
            ].idxmax(),
            "best_single_ndcg_10": max(
                restricted_base, float(ndcg["sasrec_original"])
            ),
            "oracle_ndcg_10": advantage["restricted_oracle_ndcg"].mean(),
            "absolute_headroom": advantage["restricted_oracle_ndcg"].mean()
            - max(restricted_base, float(ndcg["sasrec_original"])),
            "relative_headroom_percentage": (
                (
                    advantage["restricted_oracle_ndcg"].mean()
                    - max(restricted_base, float(ndcg["sasrec_original"]))
                )
                / max(restricted_base, float(ndcg["sasrec_original"]))
                * 100
            ),
            "users_benefiting_percentage": (
                advantage["restricted_oracle_ndcg"]
                .gt(advantage["best_non_sequential_ndcg"])
                .mean()
                * 100
            ),
        },
    ]
    tie_rows = [
        {
            "experiment": "previous_leave_one_out",
            "tie_rate_percentage": 51.03,
            "all_zero_rate_percentage": 18.52,
        },
        {
            "experiment": "revised_multi_target",
            "tie_rate_percentage": (
                advantage["best_model_tie_count"].gt(1).mean() * 100
            ),
            "all_zero_rate_percentage": advantage["all_zero_outcome"].mean() * 100,
        },
    ]
    return pd.DataFrame(oracle_rows), pd.DataFrame(tie_rows)


def statistical_comparisons(
    advantage: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Perform paired user-level bootstrap and Wilcoxon comparisons."""
    comparisons = {
        "sasrec_minus_fpmc": advantage["sasrec_original"] - advantage["fpmc"],
        "sasrec_minus_best_non_sequential": advantage["delta_ndcg_strong_seq"],
        "sasrec_original_minus_shuffled": advantage["sequence_order_gain_user"],
    }
    rows = []
    for name, values_series in comparisons.items():
        values = values_series.to_numpy(dtype=float)
        lower, upper = bootstrap_interval(values, samples, seed + len(rows))
        nonzero = values[values != 0]
        wilcoxon_result = (
            wilcoxon(nonzero, alternative="two-sided")
            if len(nonzero)
            else None
        )
        rows.append(
            {
                "comparison": name,
                "user_count": len(values),
                "mean_difference": values.mean(),
                "median_difference": np.median(values),
                "standardized_effect": values.mean() / values.std(ddof=1),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "wilcoxon_statistic": (
                    float(wilcoxon_result.statistic) if wilcoxon_result else np.nan
                ),
                "wilcoxon_p_value": (
                    float(wilcoxon_result.pvalue) if wilcoxon_result else np.nan
                ),
                "multiple_comparison_caution": True,
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    overall: pd.DataFrame,
    advantage: pd.DataFrame,
    oracle: pd.DataFrame,
    ties: pd.DataFrame,
    association: pd.DataFrame,
    classification: pd.DataFrame,
    correlations: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Create the 18 required revised-feasibility plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    ndcg = overall.loc[
        overall["metric"].eq("ndcg_10")
        & overall["model_name"].isin(CORE_MODELS)
    ]
    recall = overall.loc[
        overall["metric"].eq("recall_10")
        & overall["model_name"].isin(CORE_MODELS)
    ]
    for data, metric, title, filename in (
        (ndcg, "mean", "Overall NDCG@10 by Model", "01_overall_ndcg.png"),
        (recall, "mean", "Overall Recall@10 by Model", "02_overall_recall.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(data["model_name"], data[metric])
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel(title.split("Overall ")[1])
        ax.tick_params(axis="x", rotation=20)
        _save_figure(fig, plots_dir / filename)

    fig, ax = plt.subplots(figsize=(7, 5))
    means = ndcg.set_index("model_name")["mean"]
    ax.bar(["FPMC", "SASRec"], [means["fpmc"], means["sasrec_original"]])
    ax.set_ylim(bottom=0)
    ax.set_title("FPMC versus SASRec")
    ax.set_xlabel("Sequential model")
    ax.set_ylabel("Mean NDCG@10")
    _save_figure(fig, plots_dir / "03_fpmc_vs_sasrec.png")

    shuffled = [
        column for column in advantage if column.startswith("sasrec_shuffled_seed_")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ["original"] + [column.rsplit("_", 1)[-1] for column in shuffled],
        [advantage["sasrec_original"].mean()]
        + [advantage[column].mean() for column in shuffled],
    )
    ax.set_ylim(bottom=0)
    ax.set_title("Original versus Shuffled SASRec")
    ax.set_xlabel("Order condition")
    ax.set_ylabel("Mean NDCG@10")
    _save_figure(fig, plots_dir / "04_original_vs_shuffled.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(advantage["delta_ndcg_strong_seq"], bins=40)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Distribution of SASRec Advantage")
    ax.set_xlabel("SASRec NDCG@10 − best non-sequential NDCG@10")
    ax.set_ylabel("Users")
    _save_figure(fig, plots_dir / "05_delta_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    restricted = oracle.loc[
        oracle["oracle"].eq("strong_sequential_vs_best_non_sequential")
    ].iloc[0]
    ax.bar(
        ["Best single", "Restricted oracle"],
        [restricted["best_single_ndcg_10"], restricted["oracle_ndcg_10"]],
    )
    ax.set_ylim(bottom=0)
    ax.set_title("Restricted Sequential/Non-Sequential Oracle")
    ax.set_ylabel("Mean NDCG@10")
    _save_figure(fig, plots_dir / "06_restricted_oracle.png")

    best = advantage["best_model"].value_counts(normalize=True).mul(100)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(best.index, best.values)
    ax.set_ylim(0, 100)
    ax.set_title("Unique Best Model and Tied Outcomes")
    ax.set_xlabel("Best-outcome category")
    ax.set_ylabel("Users (%)")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, plots_dir / "07_best_model_percentage.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ties))
    width = 0.35
    ax.bar(x - width / 2, ties["tie_rate_percentage"], width, label="Tie rate")
    ax.bar(
        x + width / 2,
        ties["all_zero_rate_percentage"],
        width,
        label="All-zero rate",
    )
    ax.set_xticks(x, ties["experiment"], rotation=15)
    ax.set_ylim(0, 100)
    ax.set_title("Tie and All-Zero Rates Before and After Revision")
    ax.set_ylabel("Users (%)")
    ax.legend()
    _save_figure(fig, plots_dir / "08_tie_zero_comparison.png")

    group_specs = [
        ("rating_count_group", "Rating-count group", "09_delta_rating_count.png"),
        ("largest_day_ratio_group", "Largest-day-ratio group", "10_delta_day_ratio.png"),
        ("history_span_days_group", "History-span group", "11_delta_history.png"),
        (
            "mean_adjacent_genre_similarity_group",
            "Adjacent-genre-similarity group",
            "12_delta_adjacent_genre.png",
        ),
        (
            "genre_transition_entropy_group",
            "Genre-transition-entropy group",
            "13_delta_transition_entropy.png",
        ),
    ]
    for column, label, filename in group_specs:
        grouped = list(advantage.groupby(column, observed=True))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(
            [group["delta_ndcg_strong_seq"] for _, group in grouped],
            tick_labels=[f"{name}\n(n={len(group)})" for name, group in grouped],
            showfliers=False,
        )
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"SASRec Advantage by {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Delta NDCG@10")
        _save_figure(fig, plots_dir / filename)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        advantage["sequence_order_gain_user"],
        advantage["delta_ndcg_strong_seq"],
        s=10,
        alpha=0.25,
    )
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title("SASRec Advantage versus Sequence-Order Gain")
    ax.set_xlabel("Original minus shuffled SASRec NDCG@10")
    ax.set_ylabel("SASRec advantage NDCG@10")
    _save_figure(fig, plots_dir / "14_delta_vs_order_gain.png")

    r2 = association.loc[
        association["result_type"].eq("cross_validation")
        & association["metric"].eq("r2")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(r2["feature_block"], r2["value"])
    ax.set_title("Cross-Validated Ridge R² by Feature Block")
    ax.set_xlabel("Feature block")
    ax.set_ylabel("Cross-validated R²")
    ax.tick_params(axis="x", rotation=15)
    _save_figure(fig, plots_dir / "15_feature_block_r2.png")

    macro = classification.loc[
        classification["target"].eq(
            classification["target"].iloc[0]
        )
        & classification["metric"].eq("macro_f1")
        & classification["model_name"].eq("logistic_regression")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(macro["feature_block"], macro["value"])
    ax.set_title("Classification Macro-F1 by Feature Block")
    ax.set_xlabel("Feature block")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=15)
    _save_figure(fig, plots_dir / "16_classification_macro_f1.png")

    coefficients = association.loc[
        association["result_type"].eq("ridge_coefficient")
        & association["feature_block"].eq("C_static_temporal_sequence")
    ].copy()
    coefficients["absolute"] = coefficients["value"].abs()
    coefficients = coefficients.nlargest(12, "absolute").sort_values("value")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coefficients["term"], coefficients["value"])
    ax.set_title("Selected Standardized Ridge Coefficients")
    ax.set_xlabel("Coefficient for SASRec advantage")
    ax.set_ylabel("Feature")
    _save_figure(fig, plots_dir / "17_selected_feature_importance.png")

    reduced = correlations.loc[
        list(STATIC_FEATURES + TEMPORAL_FEATURES + SEQUENCE_FEATURES),
        list(STATIC_FEATURES + TEMPORAL_FEATURES + SEQUENCE_FEATURES),
    ]
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(reduced.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Reduced Feature Spearman Correlations")
    ax.set_xticks(range(len(reduced)), reduced.columns, rotation=90)
    ax.set_yticks(range(len(reduced)), reduced.index)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    _save_figure(fig, plots_dir / "18_reduced_feature_correlation.png")


def create_report(
    config: RevisedConfig,
    overall: pd.DataFrame,
    advantage: pd.DataFrame,
    oracle: pd.DataFrame,
    ties: pd.DataFrame,
    association: pd.DataFrame,
    classification: pd.DataFrame,
    statistical: pd.DataFrame,
) -> str:
    """Create the actual-results revised feasibility report."""
    ndcg = overall.loc[
        overall["metric"].eq("ndcg_10")
    ].set_index("model_name")["mean"]
    sasrec_gain_fpmc = ndcg["sasrec_original"] - ndcg["fpmc"]
    best_nonseq = ndcg.loc[list(NON_SEQUENTIAL_MODELS)].idxmax()
    sasrec_gap = ndcg["sasrec_original"] - ndcg[best_nonseq]
    revised_ties = ties.loc[ties["experiment"].eq("revised_multi_target")].iloc[0]
    order_gain = advantage["sequence_order_gain_user"].mean()
    clear_win = advantage["delta_ndcg_strong_seq"].ge(
        config.clear_win_epsilon
    ).mean() * 100
    r2 = association.loc[
        association["result_type"].eq("cross_validation")
        & association["metric"].eq("r2")
    ].set_index("feature_block")["value"]
    classification_main = classification.loc[
        classification["target"].eq(f"clear_win_epsilon_{config.clear_win_epsilon:g}")
        & classification["metric"].eq("macro_f1")
    ]
    best_classifier = classification_main.loc[
        classification_main["value"].idxmax()
    ]
    restricted = oracle.loc[
        oracle["oracle"].eq("strong_sequential_vs_best_non_sequential")
    ].iloc[0]
    if (
        clear_win >= 15
        and revised_ties["tie_rate_percentage"] < 25
        and restricted["absolute_headroom"] >= 0.02
        and best_classifier["value"] >= 0.60
        and r2["C_static_temporal_sequence"] >= 0.05
    ):
        decision = "Proceed to selector"
    elif (
        clear_win >= 5
        and restricted["absolute_headroom"] >= 0.01
        and best_classifier["value"] >= 0.55
    ):
        decision = "Proceed with a narrower selector"
    elif order_gain > 0.005:
        decision = "Revise the research question"
    else:
        decision = "Stop the selector direction"

    return f"""# Revised MovieLens 1M Sequence Feasibility Report

## Scope

This experiment uses rating-submission order, not watching order. Ratings ≥ {config.relevance_threshold:g} are positive. The chronological hybrid split assigns at least two validation and two test positives per user while preserving at least five training positives. Test evaluation ranks the complete held-out test set against all unseen catalog items.

## A. Does SASRec improve over FPMC?

SASRec mean NDCG@10 is {ndcg['sasrec_original']:.4f}, compared with FPMC {ndcg['fpmc']:.4f}; the improvement is {sasrec_gain_fpmc:+.4f}.

## B. Is SASRec competitive with non-sequential models?

The strongest non-sequential model is `{best_nonseq}` at {ndcg[best_nonseq]:.4f}. SASRec differs by {sasrec_gap:+.4f}.

## C. Does multi-target evaluation reduce ties and all-zero outcomes?

Tie rate changes from 51.03% to {revised_ties['tie_rate_percentage']:.2f}%, and all-zero rate changes from 18.52% to {revised_ties['all_zero_rate_percentage']:.2f}%.

## D. Does original order outperform shuffled order?

Original SASRec exceeds the mean of shuffled-order SASRec runs by {order_gain:+.4f} NDCG@10. Shuffled training preserves item multisets, validation/test blocks, item frequencies, and sequence lengths.

## E. Is there a meaningful clear-win subgroup?

Using epsilon {config.clear_win_epsilon:.2f}, {clear_win:.2f}% of users are clear SASRec wins over the best non-sequential model. Ambiguous users are excluded from margin-based classification rather than assigned arbitrary labels.

## F. Do sequence-predictability features add explanatory value?

Cross-validated ridge R² is {r2['A_static']:.4f} for static features, {r2['B_static_temporal']:.4f} after basic temporal features, and {r2['C_static_temporal_sequence']:.4f} after sequence-predictability features. The best clear-win classifier is `{best_classifier['model_name']}` with `{best_classifier['feature_block']}` features and Macro-F1 {best_classifier['value']:.4f}.

## G. Do associations remain after rating-count control?

Every ridge block includes rating count, and Models B/C add temporal and sequence predictors. These are conditional predictive associations, not causal effects; correlated predictors and multiple comparisons require caution.

## H. Is restricted oracle headroom material?

The restricted SASRec-versus-best-non-sequential oracle headroom is {restricted['absolute_headroom']:.4f} NDCG@10 ({restricted['relative_headroom_percentage']:.2f}% relative), benefiting {restricted['users_benefiting_percentage']:.2f}% of users.

## I. Decision

**{decision}.**

The decision jointly considers SASRec competitiveness, clear-win prevalence, tie reduction, order sensitivity, restricted oracle headroom, and out-of-fold feature predictability.

## Limitations

- Timestamps are rating-submission times, not watching times.
- Many histories may reflect profile-building or historical backfilling.
- Sequence-predictability features are proxies.
- No ground-truth sequence-quality label exists.
- MovieLens 1M is old and may not generalise to modern streaming systems.
- SASRec ran on {next(iter(set(str(value) for value in ['CPU-only PyTorch'])))} with compact course-project hyperparameters.
"""


def run_experiment(config: RevisedConfig) -> None:
    """Run the complete revised feasibility experiment."""
    started = time.perf_counter()
    set_deterministic_seed(config.random_seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_times: dict[str, float] = {}

    print("[1/11] Loading data and constructing revised split", flush=True)
    stage = time.perf_counter()
    ratings = load_ml1m_ratings(config.data_dir)
    movies = load_ml1m_movies(config.data_dir)
    split = revised_chronological_split(ratings, config)
    stage_times["split_seconds"] = time.perf_counter() - stage
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[2/11] Building full-ranking validation/test sets", flush=True)
    movie_ids = np.sort(movies["MovieID"].unique()).astype(np.int64)
    validation_sets, validation_candidates = build_ranking_sets(
        split, split.validation_positive, movie_ids
    )
    test_sets, test_candidates = build_ranking_sets(
        split, split.test_positive, movie_ids
    )

    print("[3/11] Extracting leakage-safe sequence features", flush=True)
    stage = time.perf_counter()
    features = sequence_predictability_features(split, movies)
    quality, correlations = feature_quality(features)
    stage_times["feature_seconds"] = time.perf_counter() - stage

    print("[4/11] Training compatible baselines and FPMC", flush=True)
    stage = time.perf_counter()
    (
        baseline_metrics,
        selected_hyperparameters,
        baseline_validation,
        final_sequences,
        item_to_index,
        mapped_movie_ids,
    ) = fit_baselines_and_fpmc(
        split, movies, validation_sets, test_sets, config
    )
    stage_times["baseline_seconds"] = time.perf_counter() - stage

    print(f"[5/11] Tuning and training SASRec on {device}", flush=True)
    stage = time.perf_counter()
    user_ids = split.eligible_users["user_id"].to_numpy(dtype=np.int64)
    sasrec_metrics, shuffle_results, sasrec_parameters, training_history = (
        tune_and_evaluate_sasrec(
            split,
            validation_sets,
            test_sets,
            user_ids,
            item_to_index,
            mapped_movie_ids,
            final_sequences,
            config,
            output_dir,
            device,
        )
    )
    stage_times["sasrec_seconds"] = time.perf_counter() - stage
    selected_hyperparameters.update(sasrec_parameters)
    selected_hyperparameters["validation_results"] = baseline_validation

    print("[6/11] Aggregating metrics and advantages", flush=True)
    stage = time.perf_counter()
    per_user = pd.concat([baseline_metrics, sasrec_metrics], ignore_index=True)
    bootstrap_samples = min(config.bootstrap_samples, 100) if config.fast_test else config.bootstrap_samples
    overall = aggregate_metrics(per_user, bootstrap_samples, config.random_seed)
    advantage = build_performance_matrix(per_user, features)
    advantage = add_groups(advantage)
    stage_times["aggregation_seconds"] = time.perf_counter() - stage

    print("[7/11] Running feature-block association/classification", flush=True)
    stage = time.perf_counter()
    association, blocks = association_models(advantage, config.random_seed)
    classification = classification_models(
        advantage, blocks, config.clear_win_epsilon, config.random_seed
    )
    groups = group_summary(
        advantage,
        per_user,
        config.clear_win_epsilon,
        bootstrap_samples,
        config.random_seed,
    )
    oracle, ties = oracle_and_ties(advantage, overall)
    statistical = statistical_comparisons(
        advantage, bootstrap_samples, config.random_seed
    )
    stage_times["analysis_seconds"] = time.perf_counter() - stage

    print("[8/11] Saving tables, splits, candidates, and predictions", flush=True)
    stage = time.perf_counter()
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(config.serializable(), indent=2), encoding="utf-8"
    )
    runtime = {
        "python_device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads_used": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "ram_total_gib": psutil.virtual_memory().total / 1024**3,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "stage_runtime_seconds": stage_times,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "hardware_and_runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    split.summary.to_csv(output_dir / "split_summary.csv", index=False)
    split.eligible_users.to_csv(output_dir / "eligible_users.csv", index=False)
    split.excluded_users.to_csv(output_dir / "excluded_users.csv", index=False)
    split.split_table.to_csv(
        output_dir / "train_validation_test_splits.csv", index=False
    )
    split.training_history.to_csv(output_dir / "training_history.csv", index=False)
    validation_candidates.assign(split="validation").to_csv(
        output_dir / "validation_candidates.csv", index=False
    )
    test_candidates.assign(split="test").to_csv(
        output_dir / "test_candidates.csv", index=False
    )
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected_hyperparameters, indent=2), encoding="utf-8"
    )
    training_history.to_csv(output_dir / "sasrec_training_curves.csv", index=False)
    overall.to_csv(output_dir / "model_overall_metrics.csv", index=False)
    per_user.to_csv(output_dir / "model_per_user_metrics.csv", index=False)
    advantage.to_csv(output_dir / "user_model_performance_matrix.csv", index=False)
    oracle.to_csv(output_dir / "oracle_headroom.csv", index=False)
    ties.to_csv(output_dir / "tie_and_zero_summary.csv", index=False)
    shuffle_results.to_csv(output_dir / "sequence_shuffle_results.csv", index=False)
    features.to_csv(
        output_dir / "user_sequence_predictability_features.csv", index=False
    )
    quality.to_csv(output_dir / "feature_quality_report.csv", index=False)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=True)
    association.to_csv(output_dir / "association_model_results.csv", index=False)
    classification.to_csv(output_dir / "classification_results.csv", index=False)
    groups.to_csv(output_dir / "group_performance_summary.csv", index=False)
    statistical.to_csv(output_dir / "statistical_comparisons.csv", index=False)
    stage_times["table_output_seconds"] = time.perf_counter() - stage

    print("[9/11] Creating required figures", flush=True)
    stage = time.perf_counter()
    create_plots(
        overall,
        advantage,
        oracle,
        ties,
        association,
        classification,
        correlations,
        output_dir / "plots",
    )
    stage_times["plot_seconds"] = time.perf_counter() - stage

    print("[10/11] Writing actual-results report", flush=True)
    stage = time.perf_counter()
    report = create_report(
        config,
        overall,
        advantage,
        oracle,
        ties,
        association,
        classification,
        statistical,
    )
    (output_dir / "revised_model_feasibility_report.md").write_text(
        report, encoding="utf-8"
    )
    stage_times["report_seconds"] = time.perf_counter() - stage

    print("[11/11] Verifying user preservation and outputs", flush=True)
    if len(advantage) != len(split.eligible_users):
        raise RuntimeError("Final user output does not preserve all eligible users")
    if advantage.isna().any().any():
        raise RuntimeError("Missing values found in final performance/features table")
    runtime["stage_runtime_seconds"] = stage_times
    runtime["total_runtime_seconds"] = time.perf_counter() - started
    (output_dir / "hardware_and_runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    print(
        f"Complete: {len(advantage):,} users on {device}; outputs at {output_dir}",
        flush=True,
    )


def parse_args() -> RevisedConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ML1M_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--minimum-training-interactions", type=int, default=5)
    parser.add_argument("--minimum-validation-targets", type=int, default=2)
    parser.add_argument("--minimum-test-targets", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=760)
    parser.add_argument("--maximum-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shuffle-runs", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--clear-win-epsilon", type=float, default=0.01)
    parser.add_argument("--fast-test", action="store_true")
    parser.add_argument("--fast-user-limit", type=int, default=300)
    arguments = parser.parse_args()
    return RevisedConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        relevance_threshold=arguments.relevance_threshold,
        minimum_training_interactions=arguments.minimum_training_interactions,
        minimum_validation_targets=arguments.minimum_validation_targets,
        minimum_test_targets=arguments.minimum_test_targets,
        holdout_fraction=arguments.holdout_fraction,
        random_seed=arguments.random_seed,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        shuffle_runs=arguments.shuffle_runs,
        bootstrap_samples=arguments.bootstrap_samples,
        clear_win_epsilon=arguments.clear_win_epsilon,
        fast_test=arguments.fast_test,
        fast_user_limit=arguments.fast_user_limit,
    )


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
