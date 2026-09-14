"""Evaluation helpers shared by the phase scripts."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from curupira.dataset import get_batch


@dataclass(frozen=True)
class LossPoint:
    """One evaluation: the loss on train and on validation at a given step."""

    step: int
    train: float
    val: float


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    splits: Mapping[str, torch.Tensor],
    batch_size: int,
    block_size: int,
    iters: int,
    device: str,
) -> dict[str, float]:
    """Average loss over `iters` random batches of each split.

    One batch is a noisy estimate; averaging several gives a number that does
    not jump around between evaluations.
    """
    model.eval()  # no effect yet, but dropout/batchnorm would care
    out: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)  # (iters,)
        for k in range(iters):
            x, y = get_batch(data, batch_size, block_size, device)  # (B, T), (B, T)
            _, loss = model(x, y)  # scalar
            losses[k] = loss.item()
        out[name] = float(losses.mean().item())
    model.train()
    return out
