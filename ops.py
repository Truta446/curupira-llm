"""Hand-written building blocks shared by the models."""

import torch


def cross_entropy(logits, targets):
    """Average cross-entropy between predicted scores and the correct tokens.

    For each position: loss = -log(probability given to the correct token),
    where probability = softmax(logits). Written without F.cross_entropy.
    """
    # logits: (B, T, V) raw scores, any real number
    # targets: (B, T) correct token ids
    B, T, V = logits.shape
    logits = logits.view(B * T, V)  # (B, T, V) -> (B*T, V)
    targets = targets.view(B * T)   # (B, T) -> (B*T,)

    # log(softmax(z))_i = z_i - log(sum_j exp(z_j)).
    # logsumexp avoids overflow from exp() of large scores.
    log_norm = torch.logsumexp(logits, dim=-1)  # (B*T,)
    correct = logits[torch.arange(B * T), targets]  # (B*T,) score of the right token
    log_prob = correct - log_norm  # (B*T,) log-probability of the right token, <= 0

    return -log_prob.mean()  # scalar
