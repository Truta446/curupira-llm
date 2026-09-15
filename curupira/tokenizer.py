"""Tokenizers for curupira-llm.

`Tokenizer` is what the rest of the code relies on. `CharTokenizer` is the
simplest possible one: every distinct character in the corpus becomes one
token, i.e. an integer in [0, V), where V = vocabulary size. The subword
tokenizer lives in `curupira/bpe.py`.
"""

import json
import os
from collections.abc import Iterable, Sequence
from typing import Any, Protocol, Self

import torch


class Tokenizer(Protocol):
    """Anything that turns text into ids and back."""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


class CharTokenizer:
    chars: list[str]
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]

    def __init__(self, chars: Iterable[str]) -> None:
        # Sorting makes ids deterministic: the same corpus always yields the
        # same mapping, so saved checkpoints stay valid.
        self.chars = sorted(set(chars))
        self.vocab_size = len(self.chars)  # V
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # int -> string

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Build the vocabulary from every character in `text`."""
        return cls(set(text))

    def encode(self, text: str) -> list[int]:
        # text: str of N chars -> list[int] of N ids, each id in [0, V)
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise KeyError(f"character not in vocabulary: {e.args[0]!r}") from None

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str:
        # ids: list[int] or tensor (N,) -> str of N chars
        id_list: list[int] = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return "".join(self.itos[i] for i in id_list)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "char", "chars": self.chars}

    def save(self, path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        with open(path, encoding="utf-8") as f:
            chars: list[str] = json.load(f)["chars"]
        return cls(chars)
