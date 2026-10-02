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


def train_neural(kind, sequences, item_count, parameters, epochs, label, validation=None):
    set_deterministic_seed(760)
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
        arrays = training_arrays(sequences, max_length, item_count, 17760 + epoch) if kind == "sasrec" else None
        order = np.random.default_rng(760 + epoch).permutation(len(sequences))
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


def summarize(per_user, output, smoke):
    aggregate = per_user.groupby(["model_name", "order_condition"])[list(METRICS)].mean().reset_index()
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    effects, seed_effects = [], []
    for model_name in ("fpmc", "sasrec", "lstm"):
        frame = per_user.loc[per_user.model_name.eq(model_name)]
        original = frame.loc[frame.order_condition.eq(ORIGINAL)].set_index("user_id")["ndcg_10"]
        for condition in (FULL_SHUFFLE, RECENT_FIXED):
            other = frame.loc[frame.order_condition.eq(condition)]
            by_user = other.groupby("user_id")["ndcg_10"].mean().reindex(original.index)
            values = (original - by_user).to_numpy()
            if not np.isfinite(values).all():
                raise AssertionError("Incomplete paired results")
            rng = np.random.default_rng(30760 + len(effects))
            means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(200 if smoke else 5000)])
            nonzero = values[values != 0]
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
                "wilcoxon_p": p, "users": len(values),
                "uncertainty": "user bootstrap conditional on one initialization and configured shuffles",
            })
            for seed, group in other.groupby("sequence_shuffle_seed"):
                delta = original - group.set_index("user_id")["ndcg_10"].reindex(original.index)
                seed_effects.append({"model_name": model_name, "intervention": condition,
                                     "shuffle_seed": int(seed), "mean_paired_difference": float(delta.mean())})
    statistics = pd.DataFrame(effects)
    statistics.to_csv(output / "order_effects.csv", index=False)
    pd.DataFrame(seed_effects).to_csv(output / "shuffle_seed_effects.csv", index=False)
    return aggregate, statistics


def run(smoke=False, name=None):
    started = time.perf_counter()
    name = name or ("smoke" if smoke else "formal_rating8")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    output = ROOT / "outputs" / "movietweetings_experiment" / name
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
        "source_files": provenance, "smoke_test": smoke, "threshold": 8,
        "initialization_seeds": [760], "shuffle_seeds": [20760] if smoke else [20760, 20761, 20762],
        "maximum_neural_epochs": 2 if smoke else 5, "patience": 2,
        "negative_sampling_seed": 17760, "bootstrap_seed": 30760,
        "split": "per-user chronological positive events; 10% validation and test, min 5/2/2",
        "metadata_policy": "unique numeric MovieID; titles and genres are unused, raw aliases retained",
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
    config = RevisedConfig(relevance_threshold=8, fast_test=smoke, fast_user_limit=40, random_seed=40760)
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
    split.summary.to_csv(output / "split_summary.csv", index=False)
    split.split_table[["UserID", "MovieID", "Rating", "Timestamp", "_source_order", "split"]].to_csv(output / "split_assignments.csv", index=False)
    pd.DataFrame({"MovieID": movie_ids, "item_index": range(len(movie_ids))}).to_csv(output / "item_mapping.csv", index=False)
    write_json(output / "candidate_policy.json", {"catalog_count": len(movie_ids), "users": len(users),
        "test_candidate_min": min(len(movie_ids) - len(x) for x in test.excluded),
        "test_candidate_max": max(len(movie_ids) - len(x) for x in test.excluded),
        "test_targets": int(sum(map(len, test.targets))), "invariants_passed": True})
    print(f"Dataset ready: {len(users)} users, {len(movie_ids)} items; smoke={smoke}", flush=True)
    all_frames = []
    def store(frame, model_name, condition, seed, path):
        frame = frame.assign(model_name=model_name, order_condition=condition,
                             model_initialization_seed=760, sequence_shuffle_seed=seed)
        frame.to_csv(path, index=False)
        print(f"Completed {model_name} {condition} {seed}: NDCG@10={frame.ndcg_10.mean():.6f}", flush=True)
        return frame

    baseline_path = output / "baselines.csv"
    if baseline_path.exists():
        all_frames.append(pd.read_csv(baseline_path))
    else:
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

    tuning_path = output / "selected_parameters.json"
    selected = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    for kind in ("fpmc", "sasrec", "lstm"):
        if kind in selected:
            continue
        print(f"Validation tuning: {kind}", flush=True)
        if kind == "fpmc":
            epochs = 1 if smoke else 3
            model = FPMC(len(users), len(movie_ids), 24, .03, .002, 760, 17760).fit(histories.training, epochs)
            candidates = []
            for weight in (.5, 1.):
                score = evaluate(validation, fpmc_scorer(model, histories.training, weight)).ndcg_10.mean()
                candidates.append({"transition_weight": weight, "epochs": epochs, "factors": 24,
                                   "validation_ndcg_10": float(score)})
            selected[kind] = max(candidates, key=lambda p: p["validation_ndcg_10"])
        else:
            options = [asdict(SASRecParameters(32, 50, 1, 2, .1, .001)),
                       asdict(SASRecParameters(64, 100, 2, 4, .2, .0005))] if kind == "sasrec" else [{"learning_rate": .001}]
            if smoke:
                options = options[:1]
            candidates = []
            for index, parameters in enumerate(options):
                model, curve, epoch, score = train_neural(kind, histories.training, len(movie_ids), parameters,
                    protocol["maximum_neural_epochs"], f"{kind}_tuning_{index}", validation)
                pd.DataFrame(curve).to_csv(output / f"{kind}_tuning_{index}_curve.csv", index=False)
                candidates.append({"parameters": parameters, "epochs": epoch, "validation_ndcg_10": score})
            selected[kind] = max(candidates, key=lambda p: p["validation_ndcg_10"])
        write_json(output / f"{kind}_validation_candidates.json", candidates)
        write_json(tuning_path, selected)
        del model

    conditions = [(ORIGINAL, None, histories.original)]
    for seed in protocol["shuffle_seeds"]:
        conditions.extend([(FULL_SHUFFLE, seed, histories.fully_shuffled[seed]),
                           (RECENT_FIXED, seed, histories.recent_context_fixed[seed])])
    for kind in ("fpmc", "sasrec", "lstm"):
        choice = selected[kind]
        for condition, seed, sequences in conditions:
            label = f"{kind}_{condition}_{seed}"
            path = output / f"{label}_metrics.csv"
            if path.exists():
                frame = pd.read_csv(path)
                if frame.user_id.tolist() != users.tolist() or not np.isfinite(frame[list(METRICS)]).all().all():
                    raise ValueError(f"Invalid cached result: {path}")
                all_frames.append(frame)
                continue
            print(f"Training {label} for {choice['epochs']} epochs", flush=True)
            run_started = time.perf_counter()
            if kind == "fpmc":
                model = FPMC(len(users), len(movie_ids), choice["factors"], .03, .002, 760, 17760).fit(sequences, choice["epochs"])
                scorer = fpmc_scorer(model, sequences, choice["transition_weight"])
            else:
                model, curve, _, _ = train_neural(kind, sequences, len(movie_ids), choice["parameters"], choice["epochs"], label)
                pd.DataFrame(curve).to_csv(output / f"{label}_curve.csv", index=False)
                length = choice["parameters"].get("max_sequence_length", 64)
                scorer = neural_scorer(model, sequences, kind, length)
            frame = store(evaluate(test, scorer), kind, condition, seed, path)
            all_frames.append(frame)
            write_json(output / f"{label}_runtime.json", {"seconds": time.perf_counter() - run_started})
            if kind != "fpmc":
                torch.save(model.state_dict(), output / f"{label}.pt")
            del model, scorer
    per_user = pd.concat(all_frames, ignore_index=True)
    expected = len(users) * (3 + 3 * len(conditions))
    if len(per_user) != expected or per_user.duplicated(["user_id", "model_name", "order_condition", "sequence_shuffle_seed"]).any():
        raise AssertionError("Incomplete or duplicate result grid")
    per_user.to_csv(output / "per_user_metrics.csv", index=False)
    aggregate, effects = summarize(per_user, output, smoke)
    if protected_snapshot() != protected:
        raise AssertionError("Protected MovieLens file hashes changed")
    write_json(output / "completion.json", {
        "complete": True, "smoke_test": smoke, "users": len(users),
        "catalog_items": len(movie_ids), "rows": len(per_user),
        "this_invocation_seconds": time.perf_counter() - started,
        "movielens_files_unchanged": True, "protected_files": len(protected),
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "rss_gib_at_completion": psutil.Process().memory_info().rss / 1024**3,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
    })
    lines = ["# MovieTweetings order experiment", "",
             "SMOKE TEST - NOT FINAL" if smoke else "Completed full eligible-user evaluation", "",
             f"Users: {len(users):,}; catalog: {len(movie_ids):,}; positive threshold: >=8.",
             "Initialization: 760. Three shuffle seeds in formal runs. No MovieLens files changed.", "",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--name")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.smoke, args.name)


if __name__ == "__main__":
    main()
