"""Training loop, evaluation and schedule helpers shared by the phase scripts."""

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from curupira.dataset import get_batch
from curupira.ops import AdamW


@dataclass(frozen=True)
class LossPoint:
    """One evaluation: the loss on train and on validation at a given step."""

    step: int
    train: float
    val: float


@dataclass(frozen=True)
class TrainConfig:
    """The phase 5 recipe: AdamW with warmup + cosine decay."""

    steps: int = 4000
    batch_size: int = 32
    block_size: int = 128
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.1
    eval_every: int = 250
    eval_iters: int = 40


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


def train_model(
    model: nn.Module,
    train: torch.Tensor,
    val: torch.Tensor,
    cfg: TrainConfig,
    device: str,
    chars_per_token: float = 1.0,
    on_eval: Callable[[LossPoint], None] | None = None,
) -> list[LossPoint]:
    """Train `model` with the phase 5 recipe and return the evaluation history.

    `chars_per_token` only affects what is printed: with a subword tokenizer the
    loss per token is also shown per character, which is what can be compared
    across tokenizers. `on_eval` runs after every evaluation (e.g. to save the
    best checkpoint).
    """
    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    min_lr = cfg.lr * cfg.min_lr_frac
    splits: dict[str, torch.Tensor] = {"train": train, "val": val}
    history: list[LossPoint] = []
    t0 = time.perf_counter()

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            losses = estimate_loss(model, splits, cfg.batch_size, cfg.block_size, cfg.eval_iters, device)
            point = LossPoint(step=step, train=losses["train"], val=losses["val"])
            history.append(point)
            per_char = "" if chars_per_token == 1.0 else f" = {point.val / chars_per_token:.4f}/char"
            print(f"  step {step:>5} | lr {lr_at(step, cfg.lr, min_lr, cfg.warmup, cfg.steps):.2e} "
                  f"| train {point.train:.4f} | val {point.val:.4f}{per_char} "
                  f"| {time.perf_counter() - t0:6.1f}s")
            if on_eval is not None:
                on_eval(point)
        if step == cfg.steps:
            break

        x, y = get_batch(train, cfg.batch_size, cfg.block_size, device)  # (B, T), (B, T)
        _, loss = model(x, y)  # scalar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lr=lr_at(step, cfg.lr, min_lr, cfg.warmup, cfg.steps))

    return history
