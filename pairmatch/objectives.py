"""Microbatch pair-level contrastive objectives with explicit invalid-case handling."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_contrastive(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> tuple[torch.Tensor, int]:
    n = len(labels)
    if n < 2:
        return embeddings.sum() * 0, 0
    z = F.normalize(embeddings, dim=1)
    similarity = z @ z.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    counts = positives.sum(1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0, 0
    logits = similarity.masked_fill(self_mask, float("-inf"))
    log_probability = similarity - torch.logsumexp(logits, dim=1, keepdim=True)
    losses = -(log_probability.masked_fill(~positives, 0).sum(1) / counts.clamp_min(1))
    return losses[valid].mean(), int(valid.sum().item())


def unsupervised_contrastive(original: torch.Tensor, augmented: torch.Tensor, ids: list[str], temperature: float) -> tuple[torch.Tensor, bool]:
    if len(ids) < 2 or len(set(ids)) < 2:
        return original.sum() * 0, False
    logits = F.normalize(original, dim=1) @ F.normalize(augmented, dim=1).T / temperature
    duplicate = torch.tensor([[i != j and x == y for j, y in enumerate(ids)] for i, x in enumerate(ids)], device=logits.device)
    logits = logits.masked_fill(duplicate, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(ids), device=logits.device)), True
