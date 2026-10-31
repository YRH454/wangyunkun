"""An independent ESIM-style BiLSTM pair classifier, not an author reproduction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def batch_encode(pairs: list[tuple[list[int], list[int]]], device: torch.device) -> tuple:
    left = [torch.tensor(a, dtype=torch.long) for a, _ in pairs]
    right = [torch.tensor(b, dtype=torch.long) for _, b in pairs]
    return (pad_sequence(left, batch_first=True).to(device), torch.tensor([len(a) for a in left]),
            pad_sequence(right, batch_first=True).to(device), torch.tensor([len(b) for b in right]))


class PairESIM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 300, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.context = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.project = nn.Sequential(nn.Linear(8 * hidden, 2 * hidden), nn.ReLU())
        self.composition = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(8 * hidden, 2 * hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(2 * hidden, 2))

    @staticmethod
    def run_lstm(layer: nn.LSTM, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = layer(packed)
        return pad_packed_sequence(encoded, batch_first=True, total_length=inputs.shape[1])[0]

    @staticmethod
    def pool(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        maximum = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min).max(1).values
        return mean, maximum

    def forward(self, left: torch.Tensor, left_len: torch.Tensor, right: torch.Tensor, right_len: torch.Tensor) -> tuple:
        left_mask, right_mask = left.ne(0), right.ne(0)
        a = self.run_lstm(self.context, self.dropout(self.embedding(left)), left_len)
        b = self.run_lstm(self.context, self.dropout(self.embedding(right)), right_len)
        scores = torch.bmm(a, b.transpose(1, 2))
        a_align = torch.bmm(scores.masked_fill(~right_mask[:, None, :], -1e4).softmax(-1), b)
        b_align = torch.bmm(scores.transpose(1, 2).masked_fill(~left_mask[:, None, :], -1e4).softmax(-1), a)
        a_features = self.project(torch.cat((a, a_align, a - a_align, a * a_align), dim=-1))
        b_features = self.project(torch.cat((b, b_align, b - b_align, b * b_align), dim=-1))
        a_comp = self.run_lstm(self.composition, self.dropout(a_features), left_len)
        b_comp = self.run_lstm(self.composition, self.dropout(b_features), right_len)
        pair = torch.cat((*self.pool(a_comp, left_mask), *self.pool(b_comp, right_mask)), dim=-1)
        return self.classifier(self.dropout(pair)), pair
