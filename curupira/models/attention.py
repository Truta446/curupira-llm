"""One head of causal self-attention, written step by step.

Only nn.Linear is used (a plain matrix multiply). The attention itself -
scores, causal mask, 1/sqrt(d) scale, softmax, weighted sum - is written by
hand: no nn.MultiheadAttention, no F.scaled_dot_product_attention.

Intuition for the three projections of a token:
    query = "what am I looking for?"
    key   = "what do I offer?"
    value = "what do I pass along if someone looks at me?"
"""

import torch
import torch.nn as nn

from curupira.ops import cross_entropy


class Head(nn.Module):
    """A single attention head."""

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)    # (C,) -> (H,)
        self.query = nn.Linear(n_embd, head_size, bias=False)  # (C,) -> (H,)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # (C,) -> (H,)
        # Lower-triangular ones: position t may only read positions <= t.
        # A buffer moves with .to(device) but is not a trainable parameter.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.tril: torch.Tensor

    def forward(self, x: torch.Tensor, scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attention), the attention matrix included for inspection."""
        # x: (B, T, C)
        B, T, C = x.shape

        q = self.query(x)  # (B, T, C) -> (B, T, H)
        k = self.key(x)    # (B, T, C) -> (B, T, H)
        v = self.value(x)  # (B, T, C) -> (B, T, H)

        # How much each position wants to read from each other position:
        # dot product of my query with every key.
        scores = q @ k.transpose(-2, -1)  # (B, T, H) @ (B, H, T) -> (B, T, T)

        # Divide by sqrt(head_size): the dot product of H random numbers grows
        # like sqrt(H), and without this the softmax saturates into a hard
        # argmax, killing the gradient. See phase3.py section 2.
        if scale:
            scores = scores * self.head_size**-0.5  # (B, T, T)

        # Causal mask: -inf above the diagonal, so softmax gives it probability 0.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)

        attention = torch.softmax(scores, dim=-1)  # (B, T, T), each row sums to 1

        out = attention @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out, attention


class AttentionLM(nn.Module):
    """Token + position embeddings -> one attention head -> next-token scores."""

    def __init__(self, vocab_size: int, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # (V, C) lookup table
        self.position_embedding = nn.Embedding(block_size, n_embd)  # (T, C): "where am I?"
        self.head = Head(n_embd, head_size, block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)             # (H,) -> (V,)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) token ids
        B, T = idx.shape
        tok = self.token_embedding(idx)  # (B, T) -> (B, T, C) "which character am I?"
        pos_ids = torch.arange(T, device=idx.device)  # (T,)
        pos = self.position_embedding(pos_ids)  # (T,) -> (T, C) "which slot am I in?"

        x = tok + pos  # (B, T, C) + (T, C) broadcast -> (B, T, C)
        x, _ = self.head(x)  # (B, T, C) -> (B, T, H)
        logits = self.lm_head(x)  # (B, T, H) -> (B, T, V)

        loss = None if targets is None else cross_entropy(logits, targets)  # scalar
        return logits, loss

    @torch.no_grad()
    def attention_for(self, idx: torch.Tensor) -> torch.Tensor:
        """Attention matrix for one sequence, for inspection."""
        # idx: (1, T)
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        _, attention = self.head(x)  # (B, T, C) -> attention: (B, T, T)
        return attention[0]  # (T, T)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx: (B, T) starting context
        for _ in range(max_new_tokens):
            cropped = idx[:, -self.block_size :]  # (B, min(T, block_size)); positions beyond have no embedding
            logits, _ = self(cropped)  # (B, T, V)
            last = logits[:, -1, :]  # (B, V) only the last position predicts what comes next
            probs = torch.softmax(last, dim=-1)  # (B, V)
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)  # (B, T) -> (B, T+1)
        return idx
