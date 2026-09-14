"""A Transformer block = multi-head attention + MLP + residuals + LayerNorm.

Each piece answers one limitation of the single head from phase 3:
    multi-head  -> several kinds of "what am I looking for?" at the same time
    MLP         -> time to *think* about what was gathered, position by position
    residual    -> a highway that lets gradients reach the early layers
    LayerNorm   -> keeps the numbers in a sane range, so deep stacks train
"""

import torch
import torch.nn as nn

from curupira.models.attention import Head
from curupira.ops import LayerNorm, cross_entropy
from curupira.sampling import sample_next


class MultiHeadAttention(nn.Module):
    """`n_head` heads running in parallel, their outputs concatenated."""

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide evenly among the heads"
        head_size = n_embd // n_head  # H: each head gets a slice of the channels
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)  # mixes what the heads found

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        outs = [head(x)[0] for head in self.heads]  # n_head tensors of (B, T, H)
        out = torch.cat(outs, dim=-1)  # (B, T, H * n_head) = (B, T, C)
        return self.proj(out)  # (B, T, C) -> (B, T, C)


class FeedForward(nn.Module):
    """Per-position MLP: the same small network applied to every token."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        # 4x wider inside, as in the original Transformer: room to compute,
        # then squeezed back to C so the residual stream keeps its width.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # (B, T, C) -> (B, T, 4C)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # (B, T, 4C) -> (B, T, C)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, C)
        return self.net(x)


class Block(nn.Module):
    """Attention (talk to the past) then MLP (think), each around a residual."""

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). "Pre-norm": normalize on the way INTO each sub-layer and
        # add the result back, so the residual stream itself is never touched.
        x = x + self.attn(self.ln1(x))  # (B, T, C) + (B, T, C) -> (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C) + (B, T, C) -> (B, T, C)
        return x


class GPT(nn.Module):
    """Token + position embeddings -> N blocks -> next-token scores."""

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # (V, C)
        self.position_embedding = nn.Embedding(block_size, n_embd)  # (T, C)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)                # final normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # (C,) -> (V,)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) token ids
        B, T = idx.shape
        tok = self.token_embedding(idx)  # (B, T) -> (B, T, C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T,) -> (T, C)

        x = tok + pos          # (B, T, C)
        x = self.blocks(x)     # (B, T, C) -> (B, T, C), N blocks in sequence
        x = self.ln_f(x)       # (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) -> (B, T, V)

        loss = None if targets is None else cross_entropy(logits, targets)  # scalar
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        # idx: (B, T) starting context
        for _ in range(max_new_tokens):
            cropped = idx[:, -self.block_size :]  # (B, min(T, block_size)): positions beyond have no embedding
            logits, _ = self(cropped)  # (B, T, V)
            last = logits[:, -1, :]  # (B, V) only the last position predicts what comes next
            nxt = sample_next(last, temperature, top_k)  # (B, V) -> (B, 1)
            idx = torch.cat([idx, nxt], dim=1)  # (B, T) -> (B, T+1)
        return idx
