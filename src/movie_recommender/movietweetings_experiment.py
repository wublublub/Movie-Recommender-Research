"""Isolated, memory-bounded MovieTweetings replication of the order experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from movie_recommender.lstm_model import LSTMRecommender
from movie_recommender.ml1m_lstm_experiment import make_history_batch, make_padded_batch
from movie_recommender.ml1m_model_feasibility import FPMC, build_interaction_matrix
from movie_recommender.ml1m_order_experiment import (
    FULL_SHUFFLE, ORIGINAL, RECENT_FIXED, METRICS, build_order_histories,
)
from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig, SASRec, SASRecParameters, pad_sequences,
    revised_chronological_split, set_deterministic_seed, training_arrays,
)
from movie_recommender.movietweetings_audit import (
    COMMIT, DATA_DIR, ROOT, file_provenance, load_movies, load_ratings,
)


@dataclass
class CompactSets:
    movie_ids: np.ndarray
    user_ids: np.ndarray
    excluded: list[np.ndarray]
    targets: list[np.ndarray]


def compact_sets(split, targets: pd.DataFrame, movie_ids: np.ndarray) -> CompactSets:
    user_ids = split.eligible_users["user_id"].to_numpy(dtype=np.int64)
    grouped = {int(u): g for u, g in targets.groupby("UserID", sort=False)}
    histories = {
        int(u): g for u, g in split.full_ratings.loc[
            split.full_ratings["UserID"].isin(user_ids)
        ].groupby("UserID", sort=False)
    }
    excluded, relevant = [], []
    for user in user_ids:
        target = grouped[int(user)]
        boundary = target["_user_order_rank"].min()
        history = histories[int(user)]
        seen = history.loc[history["_user_order_rank"].lt(boundary), "MovieID"]
        target_ids = np.sort(target["MovieID"].unique())
        blocked = np.setdiff1d(seen, target_ids)
        if not np.isin(target_ids, movie_ids).all() or not np.isin(blocked, movie_ids).all():
            raise AssertionError("Catalog omitted a target or a historical item")
        excluded.append(np.searchsorted(movie_ids, blocked).astype(np.int32))
        relevant.append(np.searchsorted(movie_ids, target_ids).astype(np.int32))
    return CompactSets(movie_ids, user_ids, excluded, relevant)


def top_metrics(scores: np.ndarray, excluded: np.ndarray, targets: np.ndarray):
    """Exact Top-10 with the existing ascending MovieID score-tie rule."""
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite prediction scores")
    if len(scores) - len(excluded) < 10 or not len(targets):
        raise ValueError("Need at least 10 candidates and one target")
    scores = np.asarray(scores, dtype=np.float64).copy()
    scores[excluded] = -np.inf
    cutoff = np.partition(scores, len(scores) - 10)[-10]
    higher = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:10 - len(higher)]
    selected = np.concatenate([higher, tied])
    selected = selected[np.lexsort((selected, -scores[selected]))]
    hits = np.isin(selected, targets).astype(float)
    ideal = min(10, len(targets))
    discounts = 1 / np.log2(np.arange(2, 12))
    hit_ranks = np.flatnonzero(hits)
    return {
        "precision_10": float(hits.sum() / 10),
        "recall_10": float(hits.sum() / len(targets)),
        "ndcg_10": float(hits @ discounts / discounts[:ideal].sum()),
        "hit_rate_10": float(hits.any()),
        "mrr_10": float(1 / (hit_ranks[0] + 1)) if len(hit_ranks) else 0.0,
        "map_10": float((hits.cumsum() / np.arange(1, 11) * hits).sum() / ideal),
    }, selected


def evaluate(sets: CompactSets, scorer, batch_size=64) -> pd.DataFrame:
    rows = []
    for start in range(0, len(sets.user_ids), batch_size):
        end = min(start + batch_size, len(sets.user_ids))
        scores = np.asarray(scorer(start, end))
        if scores.shape != (end - start, len(sets.movie_ids)):
            raise AssertionError("Scorer returned an incomplete catalog or user batch")
        for offset, row in enumerate(range(start, end)):
            metrics, top = top_metrics(scores[offset], sets.excluded[row], sets.targets[row])
            rows.append({
                "user_id": int(sets.user_ids[row]), **metrics,
                "number_of_test_targets": len(sets.targets[row]),
                "number_of_candidates": len(sets.movie_ids) - len(sets.excluded[row]),
                "top_10_movie_ids": "|".join(map(str, sets.movie_ids[top])),
            })
    return pd.DataFrame(rows)


def fpmc_rank_diagnostics(sets: CompactSets, scorer, training_movie_ids: set[int]) -> pd.DataFrame:
    rows = []
    for user_index, user_id in enumerate(sets.user_ids):
        scores = np.asarray(scorer(user_index, user_index + 1))[0]
        candidate_mask = np.ones(len(sets.movie_ids), dtype=bool)
        candidate_mask[sets.excluded[user_index]] = False
        candidate_scores = scores[candidate_mask]
        finite = bool(np.isfinite(candidate_scores).all())
        if not finite:
            raise ValueError("Nonfinite FPMC candidate scores")
        for target_index in sets.targets[user_index]:
            target_score = scores[target_index]
            rank = 1 + int(np.count_nonzero(candidate_scores > target_score))
            rank += int(np.count_nonzero((scores == target_score) & candidate_mask &
                                         (np.arange(len(scores)) < target_index)))
            rows.append({
                "user_id": int(user_id), "target_movie_id": int(sets.movie_ids[target_index]),
                "warm_in_global_training": int(sets.movie_ids[target_index]) in training_movie_ids,
                "rank": rank, "candidate_count": len(candidate_scores),
                "all_candidate_scores_finite": finite,
                "candidate_score_min": float(candidate_scores.min()),
                "candidate_score_max": float(candidate_scores.max()),
                "candidate_score_std": float(candidate_scores.std()),
            })
    return pd.DataFrame(rows)


def block_item_knn(matrix, neighbors=50, block_size=128):
    item_user = matrix.T.tocsr()
    model = NearestNeighbors(
        n_neighbors=min(neighbors + 1, item_user.shape[0]),
        metric="cosine", algorithm="brute", n_jobs=1,
    ).fit(item_user)
    rows, columns, values = [], [], []
    for start in range(0, item_user.shape[0], block_size):
        distances, indices = model.kneighbors(item_user[start:start + block_size])
        for offset in range(len(indices)):
            row = start + offset
            keep = (indices[offset] != row) & (distances[offset] < 1)
            selected = indices[offset][keep]
            rows.extend([row] * len(selected))
            columns.extend(selected.tolist())
            values.extend((1 - distances[offset][keep]).tolist())
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(matrix.shape[1], matrix.shape[1]), dtype=np.float32
    )


def lstm_valid_logits(model, inputs, targets):
    hidden, _ = model.lstm(model.item_embedding(inputs))
    valid = targets.ne(-100)
    return model.output_layer(hidden[valid]), targets[valid]


def neural_scorer(model, sequences, kind, max_length):
    def score(start, end):
        model.eval()
        fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            # The fused CPU encoder propagates NaNs from fully masked left-padding
            # queries into later layers. The SDPA path handles these queries safely.
            if kind == "sasrec":
                torch.backends.mha.set_fastpath_enabled(False)
            with torch.no_grad():
                if kind == "sasrec":
                    inputs = torch.from_numpy(pad_sequences(sequences[start:end], max_length))
                    return model.score_all(inputs).numpy()
                inputs, lengths = make_history_batch(sequences[start:end], max_length)
                # Project only the last valid hidden state, not every padded timestep.
                hidden, _ = model.lstm(model.item_embedding(inputs[:, :int(lengths.max())]))
                return model.output_layer(hidden[torch.arange(end - start), lengths - 1]).numpy()
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
    return score


def train_neural(kind, sequences, item_count, parameters, epochs, label, validation=None,
                 initialization_seed=760):
    set_deterministic_seed(initialization_seed)
    if kind == "sasrec":
        params = SASRecParameters(**parameters)
        model = SASRec(item_count, params)
        max_length, batch_size = params.max_sequence_length, 128
    else:
        model = LSTMRecommender(item_count, 32, 32)
        max_length, batch_size = 64, 32
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    best_score, best_epoch, stale, best_state = -np.inf, 1, 0, None
    curve = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        arrays = training_arrays(sequences, max_length, item_count, 17000 + initialization_seed + epoch) if kind == "sasrec" else None
        order = np.random.default_rng(initialization_seed + epoch).permutation(len(sequences))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if kind == "sasrec":
                inputs, positives, negatives = [torch.from_numpy(a[indices]) for a in arrays]
                hidden = model(inputs)
                positive_logits = (hidden * model.item_embedding(positives)).sum(-1)
                negative_logits = (hidden * model.item_embedding(negatives)).sum(-1)
                loss_values = (
                    torch.nn.functional.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits), reduction="none")
                    + torch.nn.functional.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits), reduction="none")
                )
                mask = positives.gt(0)
                loss = (loss_values * mask).sum() / mask.sum()
            else:
                inputs, targets, lengths = make_padded_batch([sequences[int(i)] for i in indices], max_length)
                width = int(lengths.max())
                logits, targets = lstm_valid_logits(model, inputs[:, :width], targets[:, :width])
                loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss in {label} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_score = None
        if validation is not None:
            val_score = float(evaluate(validation, neural_scorer(model, sequences, kind, max_length))["ndcg_10"].mean())
            tolerance = 1e-5 if kind == "sasrec" else 0
            if val_score > best_score + tolerance:
                best_score, best_epoch, stale = val_score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        curve.append({"run": label, "epoch": epoch, "training_loss": float(np.mean(losses)),
                      "validation_ndcg_10": val_score, "seconds": time.perf_counter() - started})
        print(json.dumps(curve[-1]), flush=True)
        if validation is not None and stale >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve, best_epoch, best_score


def fpmc_scorer(model, sequences, weight):
    previous = np.asarray([s[-1] for s in sequences])
    def score(start, end):
        return (model.user_factors[start:end] @ model.item_preference_factors.T
                + weight * (model.previous_item_factors[previous[start:end]] @ model.next_item_factors.T))
    return score


def sha256(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_snapshot():
    paths = list((ROOT / "data" / "raw" / "ml-1m").rglob("*"))
    paths += list((ROOT / "src" / "movie_recommender").glob("ml1m*.py"))
    paths += [ROOT / "src" / "movie_recommender" / "lstm_model.py"]
    for folder in (ROOT / "outputs").iterdir():
        if folder.is_dir() and not folder.name.startswith("movietweetings"):
            paths += list(folder.rglob("*"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths) if p.is_file()}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def current_git_branch() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "branch", "--show-current"],
        cwd=ROOT, text=True,
    ).strip()


def parse_initialization_seeds(value: str | None) -> list[int]:
    if value is None:
        return [760]
    try:
        seeds = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--initialization-seeds requires comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("--initialization-seeds requires distinct nonnegative integers")
    return seeds


def condition_grid(initialization_seeds, shuffle_seeds, pilot):
    return [(initialization_seed, ORIGINAL, None) for initialization_seed in initialization_seeds] + [
        (initialization_seed, condition, shuffle_seed)
        for initialization_seed in initialization_seeds
        for shuffle_seed in shuffle_seeds
        for condition in ((FULL_SHUFFLE,) if pilot else (FULL_SHUFFLE, RECENT_FIXED))
    ]


def seed_metadata(initialization_seeds, shuffle_seeds):
    return {"initialization_seeds": list(initialization_seeds),
            "shuffle_seeds": list(shuffle_seeds)}


def summarize(per_user, output, smoke, interventions=(FULL_SHUFFLE, RECENT_FIXED)):
    aggregate = per_user.groupby(["model_name", "order_condition"])[list(METRICS)].mean().reset_index()
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    per_initialization = per_user.groupby(
        ["model_initialization_seed", "model_name", "order_condition"]
    )[list(METRICS)].mean().reset_index()
    per_initialization.to_csv(output / "aggregate_metrics_by_initialization.csv", index=False)
    effects, seed_effects = [], []
    for model_name in ("fpmc", "sasrec", "lstm"):
        frame = per_user.loc[per_user.model_name.eq(model_name)]
        original = frame.loc[frame.order_condition.eq(ORIGINAL)].set_index(
            ["model_initialization_seed", "user_id"]
        )["ndcg_10"]
        for condition in interventions:
            other = frame.loc[frame.order_condition.eq(condition)]
            by_user = other.groupby(["model_initialization_seed", "user_id"])["ndcg_10"].mean().reindex(original.index)
            values = (original - by_user).to_numpy()
            if not np.isfinite(values).all():
                raise AssertionError("Incomplete paired results")
            rng = np.random.default_rng(30760 + len(effects))
            initialization_count = original.index.get_level_values(0).nunique()
            samples = 200 if smoke else 5000
            if initialization_count == 1:
                means = np.array([rng.choice(values, len(values), replace=True).mean()
                                  for _ in range(samples)])
            else:
                paired = (original - by_user).unstack("user_id").to_numpy()
                means = np.array([
                    paired[np.ix_(rng.integers(initialization_count, size=initialization_count),
                                  rng.integers(paired.shape[1], size=paired.shape[1]))].mean()
                    for _ in range(samples)
                ])
            user_effects = (original - by_user).groupby("user_id").mean().to_numpy()
            nonzero = user_effects[user_effects != 0]
            p = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            low, high = np.quantile(means, [.025, .975])
            adjusted_low, adjusted_high = np.quantile(means, [.05 / 6, 1 - .05 / 6])
            effects.append({
                "model_name": model_name, "intervention": condition,
                "original_ndcg_10": float(original.mean()),
                "intervention_ndcg_10": float(by_user.mean()),
                "mean_paired_difference": float(values.mean()),
                "ci95_low": float(low), "ci95_high": float(high),
                "ci_bonferroni_three_models_low": float(adjusted_low),
                "ci_bonferroni_three_models_high": float(adjusted_high),
                "standardized_effect": float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) else 0.0,
                "wilcoxon_p": p, "users": len(user_effects),
                "initialization_count": initialization_count,
                "uncertainty": (
                    "hierarchical paired initialization/user bootstrap, configured shuffles"
                    if initialization_count > 1 else
                    "user bootstrap conditional on one initialization and configured shuffles"
                ),
            })
            for (initialization_seed, seed), group in other.groupby(
                ["model_initialization_seed", "sequence_shuffle_seed"]
            ):
                matching_original = original.loc[initialization_seed]
                delta = matching_original - group.set_index("user_id")["ndcg_10"].reindex(matching_original.index)
                seed_effects.append({"model_name": model_name, "intervention": condition,
                                     "initialization_seed": int(initialization_seed),
                                     "shuffle_seed": int(seed), "mean_paired_difference": float(delta.mean())})
    statistics = pd.DataFrame(effects)
    statistics.to_csv(output / "order_effects.csv", index=False)
    pd.DataFrame(seed_effects).to_csv(output / "shuffle_seed_effects.csv", index=False)
    return aggregate, statistics


def run(smoke=False, name=None, pilot_users=None, initialization_seeds=None):
    started = time.perf_counter()
    if pilot_users is not None and (smoke or not 1 <= pilot_users <= 1500):
        raise ValueError("Pilot requires 1-1500 users and cannot use --smoke")
    initialization_seeds = [760] if initialization_seeds is None else list(initialization_seeds)
    if not initialization_seeds or any(seed < 0 for seed in initialization_seeds) or len(set(initialization_seeds)) != len(initialization_seeds):
        raise ValueError("Initialization seeds must be distinct nonnegative integers")
    pilot = pilot_users is not None
    name = name or (f"medium_pilot_u{pilot_users}" if pilot else "smoke" if smoke else "formal_rating8")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    output = ROOT / "outputs" / "movietweetings_experiment" / name
    if pilot and output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protected = protected_snapshot()
    protection_path = output / "protected_movielens_before.json"
    if protection_path.exists():
        if json.loads(protection_path.read_text()) != protected:
            raise ValueError("Protected MovieLens files changed since this run began")
    else:
        write_json(protection_path, protected)
    provenance = file_provenance(DATA_DIR)
    if provenance["ratings.dat"]["git_blob_sha1"] != "d1e392168f487faca0fb781d647b97f57a778494":
        raise ValueError("Wrong dataset version")
    protocol = {
        "dataset": "MovieTweetings", "upstream_commit": COMMIT,
        "source_files": provenance, "smoke_test": smoke, "pilot": pilot,
        "pilot_user_limit": pilot_users, "threshold": 8,
        **seed_metadata(initialization_seeds, [20760] if smoke or pilot else [20760, 20761, 20762]),
        "maximum_neural_epochs": 2 if smoke else 5, "patience": 2,
        "negative_sampling_seed": 17760, "bootstrap_seed": 30760,
        "split": "per-user chronological positive events; 10% validation and test, min 5/2/2",
        "metadata_policy": "unique numeric MovieID; titles and genres are unused, raw aliases retained",
        "primary_threshold_rationale": "rating >=8 is the prespecified high-positive category on the 0-10 scale; 7 and 9 are audit-only sensitivity thresholds",
        "catalog_policy": "all normalized catalog items, including cold items; no sampled candidates",
        "scope": "source movie-rating catalog including shorts; not independently feature-film-only verified",
        "ranking": "same observed-item exclusion and target inclusion as MovieLens, computed in batches",
        "sasrec_inference": "fused MHA fastpath disabled locally to prevent left-padding NaNs",
        "losses": {"fpmc": "BPR", "sasrec": "sampled binary cross entropy", "lstm": "full softmax cross entropy"},
        "runner_sha256": sha256(Path(__file__)),
        "protected_code_sha256": {k: v for k, v in protected.items() if k.startswith("src/")},
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Run directory has a different protocol; choose a new --name")
    write_json(protocol_path, protocol)
    torch.set_num_threads(4)
    ratings, movies = load_ratings(DATA_DIR / "ratings.dat"), load_movies(DATA_DIR / "movies.dat")
    print(f"Loaded data in {time.perf_counter() - started:.1f}s", flush=True)
    if ratings.duplicated(["UserID", "MovieID"]).any():
        raise ValueError("Repeated user-movie ratings need an explicit policy")
    config = RevisedConfig(relevance_threshold=8, fast_test=smoke, fast_user_limit=40,
                           user_limit=pilot_users, random_seed=40760)
    split = revised_chronological_split(ratings, config)
    print(f"Split complete at {time.perf_counter() - started:.1f}s", flush=True)
    movie_ids = np.sort(movies["MovieID"].unique())
    mapping = {int(movie): i for i, movie in enumerate(movie_ids)}
    users = split.eligible_users["user_id"].to_numpy()
    if not len(users):
        raise ValueError("No eligible users")
    histories = build_order_histories(split, users, mapping, protocol["shuffle_seeds"])
    print(f"Order histories complete at {time.perf_counter() - started:.1f}s", flush=True)
    validation, test = compact_sets(split, split.validation_positive, movie_ids), compact_sets(split, split.test_positive, movie_ids)
    shuffled = histories.fully_shuffled[protocol["shuffle_seeds"][0]]
    multiset_mismatches = sum(
        not np.array_equal(np.sort(original), np.sort(shuffled_row))
        for original, shuffled_row in zip(histories.original, shuffled, strict=True)
    )
    candidate_target_mismatches = sum(
        bool(np.intersect1d(excluded, targets).size)
        for excluded, targets in zip(test.excluded, test.targets, strict=True)
    )
    if multiset_mismatches or candidate_target_mismatches:
        raise AssertionError("Pilot order or candidate/target invariant failed")
    write_json(output / "order_audit.json", {
        "original_users": len(histories.original),
        "fully_shuffled_users": len(shuffled),
        "interaction_multiset_mismatches": multiset_mismatches,
        "candidate_target_mismatches": candidate_target_mismatches,
        "condition_specific_target_or_candidate_sets": False,
        "validation_users": len(validation.user_ids),
        "test_users": len(test.user_ids),
    })
    split.summary.to_csv(output / "split_summary.csv", index=False)
    split.split_table[["UserID", "MovieID", "Rating", "Timestamp", "_source_order", "split"]].to_csv(output / "split_assignments.csv", index=False)
    pd.DataFrame({"MovieID": movie_ids, "item_index": range(len(movie_ids))}).to_csv(output / "item_mapping.csv", index=False)
    write_json(output / "candidate_policy.json", {"catalog_count": len(movie_ids), "users": len(users),
        "test_candidate_min": min(len(movie_ids) - len(x) for x in test.excluded),
        "test_candidate_max": max(len(movie_ids) - len(x) for x in test.excluded),
        "test_targets": int(sum(map(len, test.targets))), "invariants_passed": True})
    print(f"Dataset ready: {len(users)} users, {len(movie_ids)} items; smoke={smoke}", flush=True)
    all_frames = []
    model_runtime = {}
    training_movie_ids = set(split.train_positive["MovieID"].astype(int)) if pilot else set()
    def store(frame, model_name, condition, seed, path, initialization_seed=760):
        frame = frame.assign(model_name=model_name, order_condition=condition,
                             model_initialization_seed=initialization_seed, sequence_shuffle_seed=seed)
        frame.to_csv(path, index=False)
        print(f"Completed {model_name} {condition} {seed}: NDCG@10={frame.ndcg_10.mean():.6f}", flush=True)
        return frame

    baseline_path = output / "baselines.csv"
    if not pilot and baseline_path.exists():
        all_frames.append(pd.read_csv(baseline_path))
    elif not pilot:
        print("Fitting non-sequential controls", flush=True)
        matrix = build_interaction_matrix(histories.original, len(movie_ids))
        for groups in (histories.fully_shuffled, histories.recent_context_fixed):
            for sequences in groups.values():
                if (matrix != build_interaction_matrix(sequences, len(movie_ids))).nnz:
                    raise AssertionError("Order-invariant input changed")
        popularity = np.asarray(matrix.sum(axis=0)).ravel()
        frames = [store(evaluate(test, lambda a, b: np.broadcast_to(popularity, (b-a, len(movie_ids)))),
                        "popularity", "order_invariant", None, output / "popularity.csv")]
        neighbors = block_item_knn(matrix, 50)
        frames.append(store(evaluate(test, lambda a, b: matrix[a:b].dot(neighbors.T).toarray()),
                            "item_knn", "order_invariant", None, output / "item_knn.csv"))
        svd = TruncatedSVD(n_components=min(32, min(matrix.shape)-1), random_state=760)
        embeddings = svd.fit_transform(matrix)
        frames.append(store(evaluate(test, lambda a, b: embeddings[a:b] @ svd.components_),
                            "matrix_factorization", "order_invariant", None, output / "matrix_factorization.csv"))
        baselines = pd.concat(frames, ignore_index=True)
        baselines.to_csv(baseline_path, index=False)
        all_frames.append(baselines)
        del matrix, neighbors, svd, embeddings

    multiple_initializations = len(initialization_seeds) > 1
    condition_sequences = {
        ORIGINAL: {None: histories.original},
        FULL_SHUFFLE: histories.fully_shuffled,
        RECENT_FIXED: histories.recent_context_fixed,
    }
    conditions = [
        (condition, shuffle_seed, condition_sequences[condition][shuffle_seed])
        for _, condition, shuffle_seed in condition_grid([760], protocol["shuffle_seeds"], pilot)
    ]
    for initialization_seed in initialization_seeds:
        suffix = f"_init{initialization_seed}" if multiple_initializations else ""
        tuning_path = output / f"selected_parameters{suffix}.json"
        selected = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
        for kind in ("fpmc", "sasrec", "lstm"):
            runtime_key = (kind, initialization_seed)
            if kind in selected:
                model_runtime[runtime_key] = {"tuning_seconds": 0.0, "condition_seconds": 0.0}
                continue
            tuning_started = time.perf_counter()
            print(f"Validation tuning: {kind}, initialization {initialization_seed}", flush=True)
            if kind == "fpmc":
                epochs = 1 if smoke else 3
                model = FPMC(len(users), len(movie_ids), 24, .03, .002,
                             initialization_seed, 17000 + initialization_seed).fit(histories.training, epochs)
                candidates = []
                for weight in (.5, 1.):
                    score = evaluate(validation, fpmc_scorer(model, histories.training, weight)).ndcg_10.mean()
                    candidates.append({"transition_weight": weight, "epochs": epochs, "factors": 24,
                                       "validation_ndcg_10": float(score)})
                selected[kind] = max(candidates, key=lambda p: p["validation_ndcg_10"])
                if pilot:
                    diagnostic = fpmc_rank_diagnostics(
                        validation, fpmc_scorer(model, histories.training,
                                                selected[kind]["transition_weight"]),
                        training_movie_ids,
                    )
                    diagnostic.to_csv(output / f"fpmc_validation_rank_diagnostics{suffix}.csv", index=False)
            else:
                options = [asdict(SASRecParameters(32, 50, 1, 2, .1, .001)),
                           asdict(SASRecParameters(64, 100, 2, 4, .2, .0005))] if kind == "sasrec" else [{"learning_rate": .001}]
                if smoke:
                    options = options[:1]
                candidates = []
                for index, parameters in enumerate(options):
                    label = f"{kind}_tuning_{index}{suffix}"
                    model, curve, epoch, score = train_neural(
                        kind, histories.training, len(movie_ids), parameters,
                        protocol["maximum_neural_epochs"], label, validation,
                        initialization_seed=initialization_seed,
                    )
                    pd.DataFrame(curve).to_csv(output / f"{label}_curve.csv", index=False)
                    candidates.append({"parameters": parameters, "epochs": epoch, "validation_ndcg_10": score})
                selected[kind] = max(candidates, key=lambda p: p["validation_ndcg_10"])
            write_json(output / f"{kind}_validation_candidates{suffix}.json", candidates)
            write_json(tuning_path, selected)
            model_runtime[runtime_key] = {"tuning_seconds": time.perf_counter() - tuning_started,
                                          "condition_seconds": 0.0}
            del model
        for kind in ("fpmc", "sasrec", "lstm"):
            choice = selected[kind]
            for condition, shuffle_seed, sequences in conditions:
                label = f"{kind}_{condition}_{shuffle_seed}{suffix}"
                path = output / f"{label}_metrics.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    if (frame.user_id.tolist() != users.tolist()
                            or not np.isfinite(frame[list(METRICS)]).all().all()
                            or not frame.model_initialization_seed.eq(initialization_seed).all()
                            or not frame.model_name.eq(kind).all()
                            or not frame.order_condition.eq(condition).all()
                            or not (frame.sequence_shuffle_seed.isna().all() if shuffle_seed is None
                                    else frame.sequence_shuffle_seed.eq(shuffle_seed).all())):
                        raise ValueError(f"Invalid cached result: {path}")
                    all_frames.append(frame)
                    continue
                print(f"Training {label} for {choice['epochs']} epochs", flush=True)
                run_started = time.perf_counter()
                if kind == "fpmc":
                    model = FPMC(len(users), len(movie_ids), choice["factors"], .03, .002,
                                 initialization_seed, 17000 + initialization_seed).fit(sequences, choice["epochs"])
                    scorer = fpmc_scorer(model, sequences, choice["transition_weight"])
                else:
                    model, curve, _, _ = train_neural(
                        kind, sequences, len(movie_ids), choice["parameters"], choice["epochs"],
                        label, initialization_seed=initialization_seed,
                    )
                    pd.DataFrame(curve).to_csv(output / f"{label}_curve.csv", index=False)
                    length = choice["parameters"].get("max_sequence_length", 64)
                    scorer = neural_scorer(model, sequences, kind, length)
                frame = store(evaluate(test, scorer), kind, condition, shuffle_seed, path,
                              initialization_seed)
                all_frames.append(frame)
                if pilot and kind == "fpmc":
                    diagnostic = fpmc_rank_diagnostics(test, scorer, training_movie_ids)
                    diagnostic.to_csv(output / f"{label}_rank_diagnostics.csv", index=False)
                condition_seconds = time.perf_counter() - run_started
                model_runtime[(kind, initialization_seed)]["condition_seconds"] += condition_seconds
                write_json(output / f"{label}_runtime.json", {"seconds": condition_seconds})
                if kind != "fpmc":
                    torch.save(model.state_dict(), output / f"{label}.pt")
                del model, scorer
    per_user = pd.concat(all_frames, ignore_index=True)
    expected = len(users) * ((0 if pilot else 3) + 3 * len(conditions) * len(initialization_seeds))
    if len(per_user) != expected or per_user.duplicated(
        ["user_id", "model_name", "order_condition", "model_initialization_seed", "sequence_shuffle_seed"]
    ).any():
        raise AssertionError("Incomplete or duplicate result grid")
    expected_cells = {
        (initialization_seed, kind, condition, shuffle_seed)
        for initialization_seed, condition, shuffle_seed in condition_grid(
            initialization_seeds, protocol["shuffle_seeds"], pilot
        )
        for kind in ("fpmc", "sasrec", "lstm")
    }
    if not pilot:
        expected_cells.update((760, kind, "order_invariant", None)
                              for kind in ("popularity", "item_knn", "matrix_factorization"))
    actual_cells = {
        (int(initialization_seed), kind, condition,
         None if pd.isna(shuffle_seed) else int(shuffle_seed))
        for initialization_seed, kind, condition, shuffle_seed in per_user[
            ["model_initialization_seed", "model_name", "order_condition", "sequence_shuffle_seed"]
        ].itertuples(index=False, name=None)
    }
    if actual_cells != expected_cells:
        raise AssertionError("Missing or unexpected initialization/shuffle cells")
    per_user.to_csv(output / "per_user_metrics.csv", index=False)
    aggregate, effects = summarize(per_user, output, smoke,
                                   (FULL_SHUFFLE,) if pilot else (FULL_SHUFFLE, RECENT_FIXED))
    pd.DataFrame([
        {"model_name": kind, "model_initialization_seed": initialization_seed, **times,
         "total_seconds": times["tuning_seconds"] + times["condition_seconds"]}
        for (kind, initialization_seed), times in model_runtime.items()
    ]).to_csv(output / "model_runtime.csv", index=False)
    if protected_snapshot() != protected:
        raise AssertionError("Protected MovieLens file hashes changed")
    write_json(output / "completion.json", {
        "complete": True, "smoke_test": smoke, "pilot": pilot, "users": len(users),
        **seed_metadata(initialization_seeds, protocol["shuffle_seeds"]),
        "catalog_items": len(movie_ids), "rows": len(per_user),
        "this_invocation_seconds": time.perf_counter() - started,
        "movielens_files_unchanged": True, "protected_files": len(protected),
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "rss_gib_at_completion": psutil.Process().memory_info().rss / 1024**3,
        "peak_process_rss_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset",
                                               psutil.Process().memory_info().rss)),
        "git_branch": current_git_branch(),
    })
    lines = ["# MovieTweetings order experiment", "",
             "MEDIUM PILOT - NON-SCIENTIFIC" if pilot else "SMOKE TEST - NOT FINAL" if smoke else "Completed full eligible-user evaluation", "",
             f"Users: {len(users):,}; catalog: {len(movie_ids):,}; positive threshold: >=8.",
             f"Initializations: {initialization_seeds}. Shuffle seeds: {protocol['shuffle_seeds']}. No MovieLens files changed.", "",
             "## NDCG@10", "", "| Model | Condition | NDCG@10 |", "|---|---|---:|"]
    lines.extend(f"| {r.model_name} | {r.order_condition} | {r.ndcg_10:.6f} |" for r in aggregate.itertuples())
    lines += ["", "## Original minus intervention", "", "| Model | Intervention | Difference | 95% CI |", "|---|---|---:|---|"]
    lines.extend(f"| {r.model_name} | {r.intervention} | {r.mean_paired_difference:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] |" for r in effects.itertuples())
    lines += ["", "## Interpretation limits", "",
              "- These are measured results, not literature values; no claim of globally optimized models.",
              "- Bootstrap intervals resample users after averaging shuffles; conditional on one initialization.",
              "- order_effects.csv also gives wider intervals for three model comparisons per intervention family.",
              "- Wilcoxon tests rank differences, not the same estimand as a confidence interval for the mean.",
              "- Rating timestamps are not verified viewing order. Equal timestamps use MovieID then source row.",
              "- 84.28% of all users are excluded at threshold 8; findings concern eligible active users.",
              "- All catalog items are retained, including cold items, matching the existing protocol.",
              "- Models use different losses; model performance differences do not isolate architecture alone.",
              "- SASRec inference disables the fused CPU attention fastpath to avoid left-padding NaNs.",
              "- Full shuffling before sequence truncation can change visible items as well as positions.",
              "- Metadata aliases were normalized numerically; genre/title fields are not used.",
              "- This is a source movie-rating catalog, not independently verified as feature-films-only.",
              "- Compare within-dataset order effects; raw MovieLens and MovieTweetings scores are not directly equivalent.",
              "- Five neural epochs and the existing small hyperparameter grid do not establish full convergence.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)
    print(effects.to_string(index=False), flush=True)
    print(f"COMPLETE: {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--pilot-users", type=int)
    parser.add_argument("--initialization-seeds", type=parse_initialization_seeds)
    return parser


def main():
    args = build_parser().parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke, args.name, args.pilot_users, args.initialization_seeds)


if __name__ == "__main__":
    main()
