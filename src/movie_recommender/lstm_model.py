import torch
from torch import nn


class LSTMRecommender(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim):
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

    def forward(self, inputs):
        embeddings = self.item_embedding(inputs)
        outputs, _ = self.lstm(embeddings)
        return self.output_layer(outputs)

    def score_last(self, inputs, lengths):
        scores = self.forward(inputs)
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        return scores[rows, lengths - 1]