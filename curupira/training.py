"""Evaluation and schedule helpers shared by the phase scripts."""

import math
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


def lr_at(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Learning rate for `step`: linear warmup, then a cosine decay to min_lr.

    Warmup: the first updates happen when the weights are still random and
    Adam's running averages are unreliable; a big step there can wreck the
    model (or produce NaN). So the rate ramps up from ~0.

    Cosine decay: big steps early to cross the landscape fast, then smaller and
    smaller steps to settle into a minimum instead of bouncing around it.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # 0.0 -> 1.0
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


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
