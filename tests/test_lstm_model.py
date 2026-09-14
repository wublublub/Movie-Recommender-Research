import numpy as np
import torch

from movie_recommender.lstm_model import LSTMRecommender
from movie_recommender.ml1m_lstm_experiment import (
    make_history_batch,
    make_padded_batch,
)


def test_model_shapes_and_padding():
    model = LSTMRecommender(
        num_items=10,
        embedding_dim=4,
        hidden_dim=8,
    )

    inputs = torch.tensor(
        [[1, 2, 3, 0],
         [4, 5, 0, 0]],
    )
    lengths = torch.tensor([3, 2])

    scores = model(inputs)
    last_scores = model.score_last(inputs, lengths)

    assert scores.shape == (2, 4, 10)
    assert last_scores.shape == (2, 10)
    assert torch.all(model.item_embedding.weight[0] == 0)


def test_padded_batch_alignment():
    sequences = [
        np.asarray([0, 2, 4]),
        np.asarray([1, 3, 5, 7]),
    ]

    inputs, targets, lengths = make_padded_batch(
        sequences,
        max_length=4,
    )

    assert inputs.tolist() == [
        [1, 3, 0, 0],
        [2, 4, 6, 0],
    ]
    assert targets.tolist() == [
        [2, 4, -100, -100],
        [3, 5, 7, -100],
    ]
    assert lengths.tolist() == [2, 3]


def test_last_position_selection():
    model = LSTMRecommender(10, 4, 8)
    inputs = torch.tensor(
        [[1, 2, 3, 0],
         [4, 5, 0, 0]],
    )
    lengths = torch.tensor([3, 2])

    all_scores = model(inputs)
    last_scores = model.score_last(inputs, lengths)

    assert torch.allclose(last_scores[0], all_scores[0, 2])
    assert torch.allclose(last_scores[1], all_scores[1, 1])