"""Byte-pair encoding (BPE) tokenizer, written by hand.

Char-level tokens are tiny: "casamento" costs 9 predictions. BPE starts from
the characters and repeatedly glues the most frequent adjacent pair into a new
token ("e" + "n" -> "en", then "en" + "to" -> "ento", ...) until the vocabulary
reaches the requested size. Frequent words end up as a single token; rare words
are spelled with smaller pieces, so no text is ever impossible to encode.

This version starts from characters rather than raw bytes: the corpus is
small and entirely Portuguese, and it keeps every token readable ("ç" stays
one symbol instead of two bytes).

Merges never cross chunk boundaries. The text is first split into words (with
their leading space), numbers, punctuation and whitespace, so no token glues
the end of one word to the start of the next.
"""

import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Final, Self

import torch

# A word with its leading space | a number | a punctuation run | whitespace | anything else.
PRETOKEN: Final = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\w\s]+|\s+|.", re.DOTALL)
NO_MERGE: Final = 1 << 62  # rank for pairs that were never merged

Pair = tuple[int, int]


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, left to right, by `new_id`."""
    # ids: n token ids -> m <= n token ids
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    base_chars: list[str]
    merges: list[Pair]
    vocab: list[str]  # id -> the text that token stands for
    vocab_size: int

    def __init__(self, base_chars: Iterable[str], merges: Sequence[Pair]) -> None:
        self.base_chars = sorted(set(base_chars))
        self.merges = [(a, b) for a, b in merges]
        # Token ids: the base characters first, then one id per merge, in order.
        self.vocab = list(self.base_chars)
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.vocab_size = len(self.vocab)
        self._char_id = {ch: i for i, ch in enumerate(self.base_chars)}
        n_base = len(self.base_chars)
        # The id of a merged token doubles as its rank: lower id = merged earlier.
        self._rank: dict[Pair, int] = {pair: n_base + i for i, pair in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int, alphabet: Iterable[str] = ()) -> Self:
        """Learn merges on `text` until the vocabulary has `vocab_size` tokens.

        `alphabet` adds characters that must be encodable even if absent from
        `text` (e.g. a character that only shows up in the validation split).

        Instead of rescanning the whole corpus after every merge, it keeps a
        running count of every adjacent pair and remembers which words contain
        each pair, so a merge only touches the words it changes.
        """
        base_chars = sorted(set(text) | set(alphabet))
        char_id = {ch: i for i, ch in enumerate(base_chars)}

        chunk_counts = Counter(PRETOKEN.findall(text))  # each distinct chunk -> occurrences
        words: list[list[int]] = [[char_id[c] for c in chunk] for chunk in chunk_counts]
        freqs: list[int] = list(chunk_counts.values())

        pair_counts: Counter[Pair] = Counter()  # pair -> occurrences in the whole corpus
        where: defaultdict[Pair, set[int]] = defaultdict(set)  # pair -> indices of words containing it
        for wi, (word, freq) in enumerate(zip(words, freqs)):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                where[pair].add(wi)

        merges: list[Pair] = []
        next_id = len(base_chars)
        while next_id < vocab_size and pair_counts:
            best = max(pair_counts, key=pair_counts.__getitem__)  # the most frequent adjacent pair
            if pair_counts[best] < 2:
                break  # nothing left that repeats
            merges.append(best)

            touched: set[Pair] = set()
            for wi in where.pop(best):
                word, freq = words[wi], freqs[wi]
                # Take the word's old pairs out of the counts, merge, put the new pairs in.
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    touched.add(pair)
                merged = merge_pair(word, best, next_id)
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += freq
                    where[pair].add(wi)
                    touched.add(pair)
                words[wi] = merged

            for pair in touched:  # keep the counter small, so max() stays fast
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    where.pop(pair, None)
            next_id += 1

        return cls(base_chars, merges)

    def truncated(self, vocab_size: int) -> "BPETokenizer":
        """The same tokenizer with only its first merges: merges are learned in order,
        so a prefix of them is exactly what training to a smaller size would give."""
        n_merges = max(0, vocab_size - len(self.base_chars))
        return BPETokenizer(self.base_chars, self.merges[:n_merges])

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        try:
            ids = [self._char_id[c] for c in chunk]  # chunk of n chars -> n base ids
        except KeyError as e:
            raise KeyError(f"character not in vocabulary: {e.args[0]!r}") from None
        # Replay the merges in the order they were learned: always apply the
        # adjacent pair that was merged earliest during training.
        while len(ids) >= 2:
            rank, pair = min((self._rank.get(p, NO_MERGE), p) for p in zip(ids, ids[1:]))
            if rank == NO_MERGE:
                break
            ids = merge_pair(ids, pair, rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        # text: str of N chars -> list[int] of N_tok ids, N_tok <= N
        return [i for chunk in PRETOKEN.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str:
        # ids: list[int] or tensor (N_tok,) -> str
        id_list: list[int] = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return "".join(self.vocab[i] for i in id_list)

    def pieces(self, text: str) -> list[str]:
        """The text split into the strings of its tokens, for display."""
        return [self.vocab[i] for i in self.encode(text)]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "bpe", "base_chars": self.base_chars, "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(data["base_chars"], [(int(a), int(b)) for a, b in data["merges"]])

    def save(self, path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
