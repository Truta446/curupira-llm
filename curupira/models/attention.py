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

from curupira.ops import apply_rope, cross_entropy, rope_tables

KVCache = tuple[torch.Tensor, torch.Tensor]
"""Keys and values already computed for the past tokens of one head: (B, T_past, H) each."""


class Head(nn.Module):
    """A single attention head, optionally with rotary position embeddings (RoPE)."""

    tril: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, n_embd: int, head_size: int, block_size: int, rope: bool = False) -> None:
        super().__init__()
        self.head_size = head_size
        self.rope = rope
        self.key = nn.Linear(n_embd, head_size, bias=False)    # (C,) -> (H,)
        self.query = nn.Linear(n_embd, head_size, bias=False)  # (C,) -> (H,)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # (C,) -> (H,)
        # Lower-triangular ones: position t may only read positions <= t.
        # A buffer moves with .to(device) but is not a trainable parameter.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        if rope:
            cos, sin = rope_tables(head_size, block_size)  # (T, H/2) each
            # Not persistent: the tables are recomputed from the config, never learned.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention itself, shared by the full and the cached paths."""
        # q: (B, T_q, H); k, v: (B, T_k, H); mask: (T_q, T_k), 1 where reading is allowed
        # How much each query position wants to read from each key position.
        scores = q @ k.transpose(-2, -1)  # (B, T_q, H) @ (B, H, T_k) -> (B, T_q, T_k)

        # Divide by sqrt(head_size): the dot product of H random numbers grows
        # like sqrt(H), and without this the softmax saturates into a hard
        # argmax, killing the gradient. See phase3.py section 2.
        if scale:
            scores = scores * self.head_size**-0.5  # (B, T_q, T_k)

        # Causal mask: -inf where reading is not allowed, so softmax gives it probability 0.
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, T_q, T_k)

        attention = torch.softmax(scores, dim=-1)  # (B, T_q, T_k), each row sums to 1
        out = attention @ v  # (B, T_q, T_k) @ (B, T_k, H) -> (B, T_q, H)
        return out, attention

    def forward(self, x: torch.Tensor, scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Every position at once (training). Returns (output, attention matrix)."""
        # x: (B, T, C)
        B, T, C = x.shape

        q = self.query(x)  # (B, T, C) -> (B, T, H)
        k = self.key(x)    # (B, T, C) -> (B, T, H)
        v = self.value(x)  # (B, T, C) -> (B, T, H)

        if self.rope:
            # Rotate queries and keys by their position. Values are NOT rotated:
            # position should decide WHO to look at, not WHAT gets passed along.
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])  # (B, T, H)
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])  # (B, T, H)

        return self._attend(q, k, v, self.tril[:T, :T], scale)  # (B, T, H), (B, T, T)

    def forward_cached(self, x: torch.Tensor, past: KVCache | None) -> tuple[torch.Tensor, KVCache]:
        """Only the NEW tokens, reusing the keys and values of the past ones (generation).

        The keys and values of a token depend only on that token and on what came
        before it, never on what comes after. So once computed they never change,
        and there is no reason to compute them again at every generated token.
        """
        # x: (B, T_new, C), the embeddings of the tokens not seen yet
        B, T_new, C = x.shape
        start = 0 if past is None else past[0].shape[1]  # absolute position of the first new token
        end = start + T_new
        if end > self.tril.shape[0]:
            raise ValueError(f"KV cache full: {end} positions, but block_size is {self.tril.shape[0]}")

        q = self.query(x)  # (B, T_new, C) -> (B, T_new, H)
        k = self.key(x)    # (B, T_new, C) -> (B, T_new, H)
        v = self.value(x)  # (B, T_new, C) -> (B, T_new, H)

        if self.rope:
            # The new tokens sit at positions start..end-1, not 0..T_new-1.
            q = apply_rope(q, self.rope_cos[start:end], self.rope_sin[start:end])  # (B, T_new, H)
            k = apply_rope(k, self.rope_cos[start:end], self.rope_sin[start:end])  # (B, T_new, H)

        if past is not None:
            k = torch.cat([past[0], k], dim=1)  # (B, T_past, H) + (B, T_new, H) -> (B, end, H)
            v = torch.cat([past[1], v], dim=1)  # (B, T_past, H) + (B, T_new, H) -> (B, end, H)

        # Rows: the new queries (positions start..end-1). Columns: every key so far (0..end-1).
        out, _ = self._attend(q, k, v, self.tril[start:end, :end], scale=True)  # (B, T_new, H)
        return out, (k, v)


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
