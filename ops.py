"""Hand-written building blocks shared by the models."""

import torch


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
