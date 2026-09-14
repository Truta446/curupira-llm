"""Turning next-token scores into an actual choice: temperature and top-k.

The model never "writes". At every step it returns V scores (logits), one per
character, and something outside the model has to pick one. How we pick
changes the text a lot, without touching a single weight.
"""

import math

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide the scores by the temperature before the softmax.

    temperature < 1 stretches the gaps between scores: the favorite gets even
    more probability (safer, more repetitive). temperature > 1 squeezes them:
    the probabilities flatten out (more varied, more invented words).
    """
    # logits: (B, V) -> (B, V)
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest scores; everything else gets -inf (probability 0).

    It cuts the long tail: hundreds of unlikely characters that, together, still
    get picked now and then and derail the text.
    """
    # logits: (B, V)
    k = min(k, logits.size(-1))
    kth_best = torch.topk(logits, k, dim=-1).values[:, -1:]  # (B, k) -> (B, 1), the k-th highest score
    return logits.masked_fill(logits < kth_best, float("-inf"))  # (B, V)


def next_token_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """Scores -> probabilities after top-k and temperature."""
    # logits: (B, V)
    if top_k is not None:
        logits = apply_top_k(logits, top_k)  # (B, V)
    return torch.softmax(apply_temperature(logits, temperature), dim=-1)  # (B, V), rows sum to 1


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """Pick the next token for each row. temperature <= 0 means greedy (always the favorite)."""
    # logits: (B, V)
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)  # (B, 1)
    probs = next_token_probs(logits, temperature, top_k)  # (B, V)
    return torch.multinomial(probs, num_samples=1)  # (B, 1), a weighted random draw


def effective_choices(probs: torch.Tensor) -> float:
    """How many characters the model is really hesitating between.

    exp(entropy): a uniform choice among N options gives exactly N. A
    distribution with one clear favorite gives a number close to 1.
    """
    # probs: (V,)
    nonzero = probs[probs > 0]  # (n,) log(0) is undefined, and 0 * log 0 counts as 0
    entropy = float(-(nonzero * nonzero.log()).sum().item())
    return math.exp(entropy)
