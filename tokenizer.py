"""Character-level tokenizer for curupira-llm.

The simplest possible tokenization: every distinct character in the corpus
becomes one token, i.e. an integer in [0, V), where V = vocabulary size.
No merges, no subwords (that comes with BPE in phase 7).
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        # Sorting makes ids deterministic: the same corpus always yields the
        # same mapping, so saved checkpoints stay valid.
        self.chars = sorted(set(chars))
        self.vocab_size = len(self.chars)  # V
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # int -> string

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary from every character in `text`."""
        return cls(set(text))

    def encode(self, text):
        # text: str of N chars -> list[int] of N ids, each id in [0, V)
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise KeyError(f"character not in vocabulary: {e.args[0]!r}") from None

    def decode(self, ids):
        # ids: list[int] or tensor (N,) -> str of N chars
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])
