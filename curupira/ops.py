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


def rope_tables(head_size: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosines and sines of the rotation angle for every (position, channel pair).

    RoPE splits a head's H channels into H/2 pairs and treats each pair as a
    point on a plane. At position p, pair i is rotated by the angle p * theta_i.
    theta_i goes from 1 (pair 0, spins fast) down to ~1/base (last pair, spins
    slowly), like the hands of a clock: fast pairs tell nearby positions apart,
    slow pairs still distinguish positions that are far apart.
    """
    assert head_size % 2 == 0, "RoPE needs an even head size: channels are rotated in pairs"
    theta = base ** (-torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)  # (H/2,)
    positions = torch.arange(max_len, dtype=torch.float32)  # (T,)
    angles = torch.outer(positions, theta)  # (T,) x (H/2,) -> (T, H/2)
    return angles.cos(), angles.sin()  # (T, H/2), (T, H/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each channel pair of x by its position's angle (a 2-D rotation per pair)."""
    # x: (..., T, H); cos, sin: (T, H/2), already cropped to the T positions of x
    x1 = x[..., 0::2]  # (..., T, H/2) first coordinate of every pair
    x2 = x[..., 1::2]  # (..., T, H/2) second coordinate
    rotated1 = x1 * cos - x2 * sin  # (..., T, H/2)
    rotated2 = x1 * sin + x2 * cos  # (..., T, H/2)
    # Interleave back so every pair returns to its original channels.
    return torch.stack((rotated1, rotated2), dim=-1).flatten(-2)  # (..., T, H/2, 2) -> (..., T, H)


class AdamW:
    """Adam with decoupled weight decay, written by hand.

    Three ideas stacked on plain SGD:
      1. momentum (m): a running average of the gradient, as in SGD+momentum;
      2. per-parameter scaling (v): a running average of the gradient SQUARED.
         Dividing by its square root gives every parameter its own step size,
         so rare parameters (a rare character's embedding) still move;
      3. decoupled weight decay: pull every weight slightly toward zero, as a
         separate step. "Decoupled" (the W in AdamW) means it is NOT added to
         the gradient, so the v-scaling does not distort it.

    Bias correction: m and v start at zero, so early on they underestimate.
    Dividing by (1 - beta**t) fixes exactly that bias.
    """

    params: list[torch.Tensor]
    m: list[torch.Tensor]
    v: list[torch.Tensor]

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # step counter, for the bias correction
        self.m = [torch.zeros_like(p) for p in self.params]  # 1st moment, same shape as p
        self.v = [torch.zeros_like(p) for p in self.params]  # 2nd moment, same shape as p
        # Decay matrices only. Biases and LayerNorm gains (1-D tensors) are not
        # "weights" in the usual sense; shrinking them just hurts.
        self.decay_mask = [p.dim() >= 2 for p in self.params]

    @torch.no_grad()
    def step(self, lr: float | None = None) -> None:
        """One update. `lr` overrides the base rate, so a schedule can drive it."""
        step_lr = self.lr if lr is None else lr
        self.t += 1
        bias1 = 1 - self.beta1**self.t
        bias2 = 1 - self.beta2**self.t
        for p, m, v, decay in zip(self.params, self.m, self.v, self.decay_mask):
            if p.grad is None:
                continue
            g = p.grad  # same shape as p
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)          # m = b1*m + (1-b1)*g
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)   # v = b2*v + (1-b2)*g^2

            m_hat = m / bias1  # same shape as p
            v_hat = v / bias2  # same shape as p

            if decay and self.weight_decay > 0:
                p.mul_(1 - step_lr * self.weight_decay)  # decoupled: applied straight to p
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-step_lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


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
