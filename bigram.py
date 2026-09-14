"""Bigram language model: the next token depends ONLY on the current token.

The whole model is one (V, V) table. Row i holds the scores (logits) for
"which token comes after token i". No context beyond one character.
"""

import torch
import torch.nn as nn

from ops import cross_entropy


class BigramLM(nn.Module):
    table: nn.Parameter

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # Zeros = every next token equally likely = the uniform guess, loss ln(V).
        self.table = nn.Parameter(torch.zeros(vocab_size, vocab_size))  # (V, V)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) token ids
        logits = self.table[idx]  # row lookup: (B, T) -> (B, T, V)
        loss = None if targets is None else cross_entropy(logits, targets)  # scalar
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx: (B, T) starting context
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, V)
            last = logits[:, -1, :]  # (B, V) only the last position predicts what's next
            probs = torch.softmax(last, dim=-1)  # (B, V), each row sums to 1
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1) draw one token
            idx = torch.cat([idx, nxt], dim=1)  # (B, T) -> (B, T+1)
        return idx
