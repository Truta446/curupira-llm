"""Crude but objective measures of generated text.

The loss says how surprised the model is by real text. It says nothing about
the text the model WRITES, which also depends on how we sample. These two
numbers do:
    real_pct     - share of generated words that exist in the training books
    distinct_pct - share of generated words that differ from each other
"""

import re
from collections.abc import Set
from dataclasses import dataclass
from typing import Final

WORD: Final = re.compile(r"[a-zà-öø-ÿ]+")  # lowercase letters, including accented ones
MIN_LENGTH: Final = 3  # "a", "e", "o", "de" are too easy to hit by chance


@dataclass(frozen=True)
class WordStats:
    n_words: int
    real_pct: float
    distinct_pct: float


def extract_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if len(w) >= MIN_LENGTH]


def corpus_vocabulary(text: str) -> set[str]:
    """Every word (3+ letters) that appears in `text`."""
    return set(extract_words(text))


def word_stats(text: str, vocabulary: Set[str]) -> WordStats:
    words = extract_words(text)
    if not words:
        return WordStats(n_words=0, real_pct=0.0, distinct_pct=0.0)
    real = sum(w in vocabulary for w in words)
    return WordStats(
        n_words=len(words),
        real_pct=100 * real / len(words),
        distinct_pct=100 * len(set(words)) / len(words),
    )
