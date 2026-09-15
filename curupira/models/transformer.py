"""A Transformer block = multi-head attention + MLP + residuals + normalization.

Each piece answers one limitation of the single head from phase 3:
    multi-head    -> several kinds of "what am I looking for?" at the same time
    MLP           -> time to *think* about what was gathered, position by position
    residual      -> a highway that lets gradients reach the early layers
    normalization -> keeps the numbers in a sane range, so deep stacks train

Phase 7 upgrades are switched on through GPT's keyword arguments, one at a
time, so every variant is the same code with a single piece swapped.
"""

import math
from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn as nn

from curupira.models.attention import Head, KVCache
from curupira.ops import LayerNorm, RMSNorm, cross_entropy, swish
from curupira.sampling import sample_next

PositionKind = Literal["learned", "rope"]
NormKind = Literal["layernorm", "rmsnorm"]
MlpKind = Literal["relu", "swiglu"]

LayerCache = list[KVCache]   # one (keys, values) pair per head
ModelCache = list[LayerCache]  # one LayerCache per block


def make_norm(kind: NormKind, dim: int) -> nn.Module:
    return RMSNorm(dim) if kind == "rmsnorm" else LayerNorm(dim)


class MultiHeadAttention(nn.Module):
    """`n_head` heads running in parallel, their outputs concatenated."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, rope: bool = False) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide evenly among the heads"
        head_size = n_embd // n_head  # H: each head gets a slice of the channels
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, rope) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)  # mixes what the heads found

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        outs = [head(x)[0] for head in self.heads]  # n_head tensors of (B, T, H)
        out = torch.cat(outs, dim=-1)  # (B, T, H * n_head) = (B, T, C)
        return self.proj(out)  # (B, T, C) -> (B, T, C)

    def forward_cached(self, x: torch.Tensor, past: LayerCache | None) -> tuple[torch.Tensor, LayerCache]:
        # x: (B, T_new, C)
        outs: list[torch.Tensor] = []
        cache: LayerCache = []
        for i, head in enumerate(self.heads):
            assert isinstance(head, Head)
            out, head_cache = head.forward_cached(x, None if past is None else past[i])  # (B, T_new, H)
            outs.append(out)
            cache.append(head_cache)
        return self.proj(torch.cat(outs, dim=-1)), cache  # (B, T_new, C)


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


def swiglu_hidden(n_embd: int) -> int:
    """Hidden width that gives SwiGLU about the same parameters as the ReLU MLP.

    ReLU MLP: two matrices of C x 4C = 8C^2 weights. SwiGLU: three matrices of
    C x h = 3Ch. They match when h = 8C/3; rounded up to a multiple of 8.
    """
    return 8 * math.ceil(8 * n_embd / 3 / 8)


class SwiGLU(nn.Module):
    """Gated MLP: one projection decides, per token and per unit, how much of the other passes.

    ReLU applies the same fixed cut (negative -> 0) to every unit. Here the cut
    itself is computed from the token: gate = swish(W_gate x) scales, unit by
    unit, the candidate values W_up x. No biases, as in the modern models.
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        hidden = swiglu_hidden(n_embd)  # h
        self.gate = nn.Linear(n_embd, hidden, bias=False)  # (C,) -> (h,) "how much lets through"
        self.up = nn.Linear(n_embd, hidden, bias=False)    # (C,) -> (h,) "what could go through"
        self.down = nn.Linear(hidden, n_embd, bias=False)  # (h,) -> (C,) back to the residual width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        gate = swish(self.gate(x))  # (B, T, C) -> (B, T, h), smooth, computed from this very token
        candidates = self.up(x)     # (B, T, C) -> (B, T, h)
        return self.down(gate * candidates)  # (B, T, h) * (B, T, h) elementwise -> (B, T, C)


def make_mlp(kind: MlpKind, n_embd: int) -> nn.Module:
    return SwiGLU(n_embd) if kind == "swiglu" else FeedForward(n_embd)


class Block(nn.Module):
    """Attention (talk to the past) then MLP (think), each around a residual."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        rope: bool = False,
        norm: NormKind = "layernorm",
        mlp: MlpKind = "relu",
    ) -> None:
        super().__init__()
        self.ln1 = make_norm(norm, n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, rope)
        self.ln2 = make_norm(norm, n_embd)
        self.ffwd = make_mlp(mlp, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). "Pre-norm": normalize on the way INTO each sub-layer and
        # add the result back, so the residual stream itself is never touched.
        x = x + self.attn(self.ln1(x))  # (B, T, C) + (B, T, C) -> (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C) + (B, T, C) -> (B, T, C)
        return x

    def forward_cached(self, x: torch.Tensor, past: LayerCache | None) -> tuple[torch.Tensor, LayerCache]:
        # x: (B, T_new, C). Only attention looks at other positions, so only it needs the cache;
        # the norms and the MLP work token by token anyway.
        attended, cache = self.attn.forward_cached(self.ln1(x), past)  # (B, T_new, C)
        x = x + attended                # (B, T_new, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T_new, C)
        return x, cache


class GPT(nn.Module):
    """Token embeddings (+ position information) -> N blocks -> next-token scores.

    position="learned": a trainable vector per slot is added to each token (phase 4).
    position="rope": no position vector at all; every attention head rotates its
    queries and keys by their position instead (phase 7b).
    norm="layernorm" (phase 4) or "rmsnorm" (phase 7c), used everywhere a
    normalization appears: twice per block and once before the output.
    mlp="relu" (phase 4) or "swiglu" (phase 7d), the MLP inside every block.
    """

    position_embedding: nn.Embedding | None

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        position: PositionKind = "learned",
        norm: NormKind = "layernorm",
        mlp: MlpKind = "relu",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.position = position
        self.norm = norm
        self.mlp = mlp
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # (V, C)
        self.position_embedding = nn.Embedding(block_size, n_embd) if position == "learned" else None  # (T, C)
        rope = position == "rope"
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, rope, norm, mlp) for _ in range(n_layer)]
        )
        self.ln_f = make_norm(norm, n_embd)           # final normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # (C,) -> (V,)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: (B, T) token ids
        B, T = idx.shape
        x = self.token_embedding(idx)  # (B, T) -> (B, T, C)
        if self.position_embedding is not None:
            pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T,) -> (T, C)
            x = x + pos  # (B, T, C) + (T, C) broadcast -> (B, T, C)

        x = self.blocks(x)     # (B, T, C) -> (B, T, C), N blocks in sequence
        x = self.ln_f(x)       # (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) -> (B, T, V)

        loss = None if targets is None else cross_entropy(logits, targets)  # scalar
        return logits, loss

    def forward_cached(self, idx: torch.Tensor, past: ModelCache | None) -> tuple[torch.Tensor, ModelCache]:
        """Next-token scores for the NEW tokens only, extending the KV cache of every block."""
        # idx: (B, T_new) token ids not seen yet
        B, T_new = idx.shape
        start = 0 if past is None else past[0][0][0].shape[1]  # how many tokens the cache already holds
        x = self.token_embedding(idx)  # (B, T_new) -> (B, T_new, C)
        if self.position_embedding is not None:
            positions = torch.arange(start, start + T_new, device=idx.device)  # (T_new,)
            x = x + self.position_embedding(positions)  # (B, T_new, C) + (T_new, C)

        cache: ModelCache = []
        for i, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x, layer_cache = block.forward_cached(x, None if past is None else past[i])  # (B, T_new, C)
            cache.append(layer_cache)
        logits = self.lm_head(self.ln_f(x))  # (B, T_new, C) -> (B, T_new, V)
        return logits, cache

    @torch.no_grad()
    def generate_cached(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Same result as `generate`, computing each token's keys and values only once.

        Limited to block_size tokens in total: past that, `generate` slides its
        window and every position changes, so there is nothing left to reuse.
        """
        # idx: (B, T) starting context
        if idx.shape[1] + max_new_tokens > self.block_size:
            raise ValueError(f"prompt + new tokens = {idx.shape[1] + max_new_tokens} > block_size {self.block_size}")
        logits, cache = self.forward_cached(idx, None)  # "prefill": the whole prompt in one pass
        for step in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_k)  # (B, V) -> (B, 1)
            idx = torch.cat([idx, nxt], dim=1)  # (B, T) -> (B, T+1)
            if step < max_new_tokens - 1:
                logits, cache = self.forward_cached(nxt, cache)  # only the token just chosen: (B, 1, V)
        return idx

    @torch.no_grad()
    def stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Iterator[int]:
        """Like `generate` for one sequence, but yields each token id as soon as it is chosen.

        Lets an interactive program print the text while it is being written.
        """
        # idx: (1, T) starting context
        for _ in range(max_new_tokens):
            cropped = idx[:, -self.block_size :]  # (1, min(T, block_size))
            logits, _ = self(cropped)  # (1, T, V)
            nxt = sample_next(logits[:, -1, :], temperature, top_k)  # (1, V) -> (1, 1)
            idx = torch.cat([idx, nxt], dim=1)  # (1, T) -> (1, T+1)
            yield int(nxt.item())

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
            cropped = idx[:, -self.block_size :]  # (B, min(T, block_size)): the model was built for block_size slots
            logits, _ = self(cropped)  # (B, T, V)
            last = logits[:, -1, :]  # (B, V) only the last position predicts what comes next
            nxt = sample_next(last, temperature, top_k)  # (B, V) -> (B, 1)
            idx = torch.cat([idx, nxt], dim=1)  # (B, T) -> (B, T+1)
        return idx
