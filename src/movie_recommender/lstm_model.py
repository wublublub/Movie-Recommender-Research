"""LSTM model and tensor utilities for sequential recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LSTMParameters:
    """Hyperparameters for the recurrent recommendation model."""

    embedding_dimension: int = 32
    hidden_dimension: int = 32
    maximum_sequence_length: int = 64
    learning_rate: float = 0.001


class LSTMRecommender(nn.Module):
    """Predict the next catalog item from an ordered interaction sequence."""

    def __init__(
        self, num_items: int, embedding_dim: int, hidden_dim: int
    ) -> None:
        """Initialize the recurrent next-item model."""
        super().__init__()
        self.item_embedding = nn.Embedding(
            num_items + 1,
            embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, num_items)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return next-item logits for every input position."""
        embeddings = self.item_embedding(inputs)
        outputs, _ = self.lstm(embeddings)
        return self.output_layer(outputs)

    def score_last(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return catalog scores at each sequence's last real position."""
        scores = self.forward(inputs)
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        last_positions = lengths.to(inputs.device) - 1
        return scores[rows, last_positions]


def make_padded_batch(
    sequences: Sequence[np.ndarray], max_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create shifted next-item training tensors with right padding."""
    inputs = torch.zeros(
        (len(sequences), max_length), dtype=torch.long
    )
    targets = torch.full(
        (len(sequences), max_length),
        fill_value=-100,
        dtype=torch.long,
    )
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        recent = np.asarray(sequence)[-(max_length + 1) :]
        input_items = torch.as_tensor(recent[:-1], dtype=torch.long) + 1
        target_items = torch.as_tensor(recent[1:], dtype=torch.long)
        length = len(input_items)
        inputs[row, :length] = input_items
        targets[row, :length] = target_items
        lengths[row] = length
    return inputs, targets, lengths


def make_history_batch(
    sequences: Sequence[np.ndarray], max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create padded inference tensors from complete pre-test histories."""
    inputs = torch.zeros(
        (len(sequences), max_length), dtype=torch.long
    )
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        recent = np.asarray(sequence)[-max_length:]
        length = len(recent)
        if length == 0:
            raise ValueError("LSTM inference requires non-empty user histories")
        inputs[row, :length] = torch.as_tensor(recent, dtype=torch.long) + 1
        lengths[row] = length
    return inputs, lengths


def lstm_score_matrix(
    model: LSTMRecommender,
    sequences: Sequence[np.ndarray],
    batch_size: int,
    maximum_sequence_length: int,
    device: torch.device,
) -> np.ndarray:
    """Score every catalog item for every supplied user sequence."""
    model.eval()
    score_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            inputs, lengths = make_history_batch(
                sequences[start : start + batch_size], maximum_sequence_length
            )
            scores = model.score_last(inputs.to(device), lengths.to(device))
            score_batches.append(scores.detach().cpu().numpy())
    return np.vstack(score_batches)
