import torch
from movie_recommender.lstm_model import LSTMRecommender
from movie_recommender.ml1m_data_loader import (
    load_ml1m_ratings,
    load_ml1m_movies,
)
from movie_recommender.ml1m_revised_sequence_feasibility import (
    RevisedConfig,
    revised_chronological_split,
    make_sequences,
)

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.ml1m_order_experiment import (
    FULL_SHUFFLE,
    ORIGINAL,
    RECENT_FIXED,
    assert_split_and_candidates,
    build_order_histories,
)
from movie_recommender.ml1m_revised_sequence_feasibility import (
    build_ranking_sets,
    evaluate_full_ranker,
)

#import dataset
def inspect_ratings():
    ratings = load_ml1m_ratings()

    print(ratings.head())
    print("Rows:", len(ratings))
    print("Users:", ratings["UserID"].nunique())
    print("Rated movies:", ratings["MovieID"].nunique())
    print("Rating counts:")
    print(ratings["Rating"].value_counts().sort_index())

def prepare_split():
    ratings = load_ml1m_ratings()
    config = RevisedConfig(relevance_threshold=4.0, fast_test=False)

    split = revised_chronological_split(ratings, config)

    print(split.summary.to_string(index=False))
    print(split.eligible_users.head())

    return split

def prepare_training_sequences():
    split = prepare_split()
    movies = load_ml1m_movies()
    movie_ids = movies["MovieID"].sort_values().unique()
    item_to_index = {}
    for index, movie_id in enumerate(movie_ids):
        item_to_index[int(movie_id)] = index

    user_ids = split.eligible_users["user_id"].to_numpy()
    sequences = make_sequences(
        split.train_positive, user_ids, item_to_index
    )
    print("Movies:", len(item_to_index))
    print("Training sequences:", len(sequences))
    print("First user's first 10 indices:", sequences[0][:10])
    return split, user_ids, item_to_index, sequences


def make_training_example(sequence):
    items = torch.tensor(sequence, dtype=torch.long)

    inputs = (items[:-1] + 1).unsqueeze(0)
    targets = items[1:].unsqueeze(0)

    return inputs, targets

def inspect_training_example():
    split, user_ids, item_to_index, sequences = prepare_training_sequences()
    inputs, targets = make_training_example(sequences[0])

    print("User:", user_ids[0])
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    print("Inputs:", inputs[0, :10])
    print("Targets:", targets[0, :10])

    assert inputs.shape == targets.shape
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def train_one_user():
    split, user_ids, item_to_index, sequences = prepare_training_sequences()
    inputs, targets = make_training_example(sequences[0])
    num_items = len(item_to_index)

    torch.manual_seed(760)
    model = LSTMRecommender(
        num_items=num_items,
        embedding_dim=32,
        hidden_dim=32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss()

    model.train()
    for step in range(1, 101):
        optimizer.zero_grad()
        scores = model(inputs)
        loss = loss_function(
            scores.reshape(-1, num_items),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(f"Step {step:3d} | loss = {loss.item():.4f}")



def make_padded_batch(sequences, max_length):
    batch_size = len(sequences)

    inputs = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
    )
    targets = torch.full(
        (batch_size, max_length),
        fill_value=-100,
        dtype=torch.long,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for row, sequence in enumerate(sequences):
        recent = sequence[-(max_length + 1):]

        input_items = torch.tensor(
            recent[:-1],
            dtype=torch.long,
        ) + 1

        target_items = torch.tensor(
            recent[1:],
            dtype=torch.long,
        )

        length = len(input_items)

        inputs[row, :length] = input_items
        targets[row, :length] = target_items
        lengths[row] = length

    return inputs, targets, lengths

def inspect_padded_batch():
    _, user_ids, _, sequences = prepare_training_sequences()

    inputs, targets, lengths = make_padded_batch(
        sequences[:3],
        max_length=64,
    )

    print("Users:", user_ids[:3])
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    print("Lengths:", lengths)

    for row, length in enumerate(lengths.tolist()):
        assert torch.all(inputs[row, length:] == 0)
        assert torch.all(targets[row, length:] == -100)

    print("Padding checks passed")


def inspect_model_with_padding():
    _, _, item_to_index, sequences = prepare_training_sequences()
    num_items = len(item_to_index)

    inputs, targets, lengths = make_padded_batch(
        sequences[:3],
        max_length=64,
    )

    model = LSTMRecommender(
        num_items=num_items,
        embedding_dim=32,
        hidden_dim=32,
    )

    scores = model(inputs)

    print("Input shape:", inputs.shape)
    print("Score shape:", scores.shape)
    print("Real target positions:", (targets != -100).sum().item())
    print("Padding vector:", model.item_embedding.weight[0])

    assert scores.shape == (3, 64, num_items)
    assert (targets != -100).sum().item() == lengths.sum().item()
    assert torch.all(model.item_embedding.weight[0] == 0)

    print("Model padding checks passed")

def train_one_batch():
    _, _, item_to_index, sequences = prepare_training_sequences()
    num_items = len(item_to_index)

    inputs, targets, lengths = make_padded_batch(
        sequences[:8],
        max_length=64,
    )

    torch.manual_seed(760)
    model = LSTMRecommender(
        num_items=num_items,
        embedding_dim=32,
        hidden_dim=32,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )
    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=-100,
    )

    print("Batch shape:", inputs.shape)
    print("Sequence lengths:", lengths)

    model.train()

    for step in range(1, 51):
        optimizer.zero_grad()

        scores = model(inputs)
        loss = loss_function(
            scores.reshape(-1, num_items),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(f"Step {step:3d} | loss = {loss.item():.4f}")

def make_history_batch(sequences, max_length):
    inputs = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
    )
    lengths = torch.zeros(len(sequences), dtype=torch.long)

    for row, sequence in enumerate(sequences):
        recent = sequence[-max_length:]
        length = len(recent)

        inputs[row, :length] = torch.tensor(
            recent + 1,
            dtype=torch.long,
        )
        lengths[row] = length

    return inputs, lengths

def score_all_users(model, sequences, batch_size, max_length):
    model.eval()
    score_batches = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start:start + batch_size]
            inputs, lengths = make_history_batch(
                batch_sequences,
                max_length,
            )
            score_batches.append(
                model.score_last(inputs, lengths).numpy()
            )

    return np.vstack(score_batches)

def train_lstm(
    sequences,
    num_items,
    max_length,
    batch_size,
    maximum_epochs,
    learning_rate,
    seed,
    validation_sets=None,
    user_ids=None,
    item_to_index=None,
    fixed_epochs=None,
):
    torch.manual_seed(seed)

    model = LSTMRecommender(
        num_items=num_items,
        embedding_dim=32,
        hidden_dim=32,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=-100,
    )

    epochs = fixed_epochs or maximum_epochs
    best_score = float("-inf")
    best_epoch = epochs
    best_state = None
    stale_epochs = 0
    training_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        order = np.random.default_rng(seed + epoch).permutation(
            len(sequences)
        )

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_sequences = [
                sequences[int(index)] for index in indices
            ]
            inputs, targets, _ = make_padded_batch(
                batch_sequences,
                max_length,
            )

            optimizer.zero_grad()
            scores = model(inputs)
            loss = loss_function(
                scores.reshape(-1, num_items),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()
            losses.append(loss.item())

        validation_ndcg = float("nan")

        if validation_sets is not None:
            score_matrix = score_all_users(
                model,
                sequences,
                batch_size,
                max_length,
            )
            metrics = evaluate_full_ranker(
                "lstm_validation",
                validation_sets,
                user_ids,
                item_to_index,
                lambda user, candidates: score_matrix[user, candidates],
            )
            validation_ndcg = metrics["ndcg_10"].mean()

            if validation_ndcg > best_score:
                best_score = validation_ndcg
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

        training_rows.append(
            {
                "epoch": epoch,
                "training_loss": np.mean(losses),
                "validation_ndcg_10": validation_ndcg,
            }
        )

        print(
            f"Epoch {epoch} | "
            f"loss={np.mean(losses):.4f} | "
            f"validation_ndcg={validation_ndcg:.4f}"
        )

        if validation_sets is not None and stale_epochs >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(training_rows), best_epoch

def run_lstm_experiment(fast_test=False, output_dir=None):
    output_dir = output_dir or Path("outputs/lstm_order_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ml1m_ratings()
    movies = load_ml1m_movies()

    split_config = RevisedConfig(
        relevance_threshold=4.0,
        fast_test=fast_test,
        fast_user_limit=120,
    )
    split = revised_chronological_split(ratings, split_config)

    movie_ids = np.sort(movies["MovieID"].unique())
    item_to_index = {
        int(movie_id): index
        for index, movie_id in enumerate(movie_ids)
    }
    user_ids = split.eligible_users["user_id"].to_numpy()

    shuffle_seeds = (20760,) if fast_test else (20760, 20761, 20762)

    histories = build_order_histories(
        split,
        user_ids,
        item_to_index,
        shuffle_seeds,
    )
    validation_sets, _ = build_ranking_sets(
        split,
        split.validation_positive,
        movie_ids,
    )
    test_sets, _ = build_ranking_sets(
        split,
        split.test_positive,
        movie_ids,
    )
    assert_split_and_candidates(
        split,
        validation_sets,
        test_sets,
    )

    maximum_epochs = 2 if fast_test else 5
    batch_size = 32
    max_length = 64
    num_items = len(movie_ids)

    _, tuning_curve, best_epoch = train_lstm(
        histories.training,
        num_items,
        max_length,
        batch_size,
        maximum_epochs,
        learning_rate=0.001,
        seed=760,
        validation_sets=validation_sets,
        user_ids=user_ids,
        item_to_index=item_to_index,
    )

    conditions = [(ORIGINAL, None, histories.original)]

    for shuffle_seed in shuffle_seeds:
        conditions.append(
            (
                FULL_SHUFFLE,
                shuffle_seed,
                histories.fully_shuffled[shuffle_seed],
            )
        )
        conditions.append(
            (
                RECENT_FIXED,
                shuffle_seed,
                histories.recent_context_fixed[shuffle_seed],
            )
        )

    result_frames = []
    curve_frames = [tuning_curve.assign(run="validation_tuning")]

    for condition, shuffle_seed, sequences in conditions:
        model, curve, _ = train_lstm(
            sequences,
            num_items,
            max_length,
            batch_size,
            maximum_epochs,
            learning_rate=0.001,
            seed=760,
            fixed_epochs=best_epoch,
        )

        score_matrix = score_all_users(
            model,
            sequences,
            batch_size,
            max_length,
        )
        metrics = evaluate_full_ranker(
            f"lstm_{condition}",
            test_sets,
            user_ids,
            item_to_index,
            lambda user, candidates: score_matrix[user, candidates],
        )
        metrics["model_name"] = "lstm"
        metrics["order_condition"] = condition
        metrics["model_initialization_seed"] = 760
        metrics["sequence_shuffle_seed"] = shuffle_seed

        result_frames.append(metrics)
        curve_frames.append(
            curve.assign(
                run=condition,
                sequence_shuffle_seed=shuffle_seed,
            )
        )

    per_user = pd.concat(result_frames, ignore_index=True)
    training_curves = pd.concat(curve_frames, ignore_index=True)

    aggregate = (
        per_user.groupby(["model_name", "order_condition"])[
            [
                "precision_10",
                "recall_10",
                "ndcg_10",
                "hit_rate_10",
                "mrr_10",
                "map_10",
            ]
        ]
        .mean()
        .reset_index()
    )

    original = per_user.loc[
        per_user["order_condition"].eq(ORIGINAL),
        ["user_id", "ndcg_10"],
    ].rename(columns={"ndcg_10": "original_ndcg_10"})

    paired = per_user.loc[
        ~per_user["order_condition"].eq(ORIGINAL)
    ].merge(original, on="user_id")

    paired["original_minus_intervention_ndcg_10"] = (
        paired["original_ndcg_10"] - paired["ndcg_10"]
    )

    per_user.to_csv(
        output_dir / "lstm_per_user_metrics.csv",
        index=False,
    )
    aggregate.to_csv(
        output_dir / "lstm_aggregate_metrics.csv",
        index=False,
    )
    paired.to_csv(
        output_dir / "lstm_order_differences.csv",
        index=False,
    )
    training_curves.to_csv(
        output_dir / "lstm_training_curves.csv",
        index=False,
    )

    print(aggregate.to_string(index=False))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-test", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lstm_order_experiment"),
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    run_lstm_experiment(
        fast_test=arguments.fast_test,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()

