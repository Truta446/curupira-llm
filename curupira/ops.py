"""Hand-written building blocks shared by the models."""

from collections.abc import Iterable

import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy between predicted scores and the correct tokens.

    For each position: loss = -log(probability given to the correct token),
    where probability = softmax(logits). Written without F.cross_entropy.
    """
    # logits: (B, T, V) raw scores, any real number
    # targets: (B, T) correct token ids
    B, T, V = logits.shape
    flat_logits = logits.view(B * T, V)  # (B, T, V) -> (B*T, V)
    flat_targets = targets.view(B * T)   # (B, T) -> (B*T,)

    # log(softmax(z))_i = z_i - log(sum_j exp(z_j)).
    # logsumexp avoids overflow from exp() of large scores.
    log_norm = torch.logsumexp(flat_logits, dim=-1)  # (B*T,)
    correct = flat_logits[torch.arange(B * T, device=logits.device), flat_targets]  # (B*T,) score of the right token
    log_prob = correct - log_norm  # (B*T,) log-probability of the right token, <= 0

    return -log_prob.mean()  # scalar


class LayerNorm(nn.Module):
    """Normalize each token's vector to mean 0 and std 1, then rescale.

    Written by hand instead of nn.LayerNorm. Note it normalizes ACROSS THE
    CHANNELS of one token: every token is normalized on its own, so nothing
    leaks from one position to another (which would break causality).
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # (C,) learned scale
        self.beta = nn.Parameter(torch.zeros(dim))  # (C,) learned shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        mean = x.mean(dim=-1, keepdim=True)                    # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)      # (..., 1)
        normalized = (x - mean) / torch.sqrt(var + self.eps)   # (..., C), mean 0 / std 1
        return self.gamma * normalized + self.beta             # (..., C)


class SGD:
    """Stochastic gradient descent with optional momentum, written by hand.

    Without momentum each step follows only the current batch's gradient, which
    zig-zags. Momentum keeps a running average of past gradients (the "velocity"),
    so consistent directions build up speed and noisy ones cancel out.
    """

    params: list[torch.Tensor]
    velocity: list[torch.Tensor]

    def __init__(self, params: Iterable[torch.Tensor], lr: float, momentum: float = 0.0) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]  # same shape as each parameter

    @torch.no_grad()
    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            v.mul_(self.momentum).add_(p.grad)  # v = momentum * v + grad
            p -= self.lr * v                    # walk downhill along the smoothed direction

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
