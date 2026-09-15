"""Comparing model variants fairly: same data, same recipe, same seeds.

A single training run can win by luck: the random initialization and the order
of the batches move the final loss. So every variant is trained with the same
list of seeds, and a difference only counts when it keeps its sign across seeds
and is larger than the spread between seeds of the same variant.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import torch

from curupira.bpe import BPETokenizer, load_or_train
from curupira.checkpoint import ModelConfig, load_checkpoint, save_checkpoint
from curupira.dataset import DATA_DIR, encode_splits, load_texts
from curupira.models.transformer import NormKind, PositionKind
from curupira.text_stats import WordStats, corpus_vocabulary, word_stats
from curupira.training import LossPoint, TrainConfig, train_model

BPE_VOCAB_SIZE: Final = 1024
BPE_CACHE: Final = DATA_DIR / "bpe_2048.json"
DEFAULT_SEEDS: Final = (1337, 2024)


@dataclass(frozen=True)
class BPEData:
    tok: BPETokenizer
    train: torch.Tensor  # (N_train_tokens,)
    val: torch.Tensor    # (N_val_tokens,)
    train_text: str
    chars_per_token: float  # measured on the validation split


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    config: ModelConfig
    checkpoint: Path | None = None  # if set, the best model of the FIRST seed is saved here


@dataclass(frozen=True)
class RunResult:
    key: str
    seed: int
    best: LossPoint
    final: LossPoint
    n_params: int


def load_bpe_data(vocab_size: int = BPE_VOCAB_SIZE) -> BPEData:
    train_text, val_text = load_texts()
    tok = load_or_train(BPE_CACHE, train_text, vocab_size, alphabet=val_text)
    train, val = encode_splits(tok, train_text, val_text)
    return BPEData(tok=tok, train=train, val=val, train_text=train_text,
                   chars_per_token=len(val_text) / len(val))


BPE_BASE_CONFIG: Final = ModelConfig(vocab_size=BPE_VOCAB_SIZE, n_embd=128, n_head=4, n_layer=4, block_size=128)


def bpe_config(position: PositionKind = "learned", norm: NormKind = "layernorm") -> ModelConfig:
    """The phase 5 architecture on BPE tokens, with the phase 7 upgrades chosen explicitly."""
    return replace(BPE_BASE_CONFIG, position=position, norm=norm)


def train_variant(variant: Variant, seed: int, data: BPEData, steps: int, device: str,
                  save: bool) -> RunResult:
    torch.manual_seed(seed)
    model = variant.config.build().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {variant.label}, semente {seed}: {n_params:,} parâmetros")
    best: list[LossPoint] = []

    def keep_best(point: LossPoint) -> None:
        if not best or point.val < best[0].val:
            best[:] = [point]
            if save and variant.checkpoint is not None:
                save_checkpoint(variant.checkpoint, model, variant.config, data.tok, point.step, point.val)

    history = train_model(model, data.train, data.val, TrainConfig(steps=steps), device,
                          chars_per_token=data.chars_per_token, on_eval=keep_best)
    return RunResult(key=variant.key, seed=seed, best=best[0], final=history[-1], n_params=n_params)


def run_ablation(variants: Sequence[Variant], seeds: Sequence[int], data: BPEData, steps: int,
                 device: str) -> list[RunResult]:
    """Train every variant with every seed, seed by seed."""
    results: list[RunResult] = []
    for seed in seeds:
        for variant in variants:
            results.append(train_variant(variant, seed, data, steps, device, save=seed == seeds[0]))
    return results


def verdict(diffs: Sequence[float], spread: float) -> str:
    """Judge a per-seed list of (variant - baseline) differences against the seed spread."""
    if len(diffs) < 2:
        return "uma semente só: não dá para separar do acaso"
    same_sign = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
    if same_sign and abs(statistics.mean(diffs)) > spread:
        return "efeito consistente e maior que a variação entre sementes"
    if same_sign:
        return "mesmo sinal em todas as sementes, mas do tamanho da variação entre elas"
    return "o sinal muda entre sementes: não dá para separar do acaso"


def per_char_losses(variants: Sequence[Variant], seeds: Sequence[int], results: Sequence[RunResult],
                    chars_per_token: float) -> dict[str, list[float]]:
    """variant key -> best validation loss per character, one value per seed in `seeds` order."""
    by_run = {(r.key, r.seed): r for r in results}
    return {v.key: [by_run[(v.key, s)].best.val / chars_per_token for s in seeds] for v in variants}


def print_scoreboard(variants: Sequence[Variant], seeds: Sequence[int], results: Sequence[RunResult],
                     chars_per_token: float) -> dict[str, list[float]]:
    """Print the table and the verdict of every variant against the first one. Returns the losses."""
    by_run = {(r.key, r.seed): r for r in results}
    losses = per_char_losses(variants, seeds, results, chars_per_token)

    width = max(len(v.label) for v in variants)
    header = " | ".join(f"semente {s:>4}" for s in seeds)
    print(f"  {'':<{width}} | {header} | {'média':>7} | {'treino→val':>10} | {'parâmetros':>10}")
    for v in variants:
        runs = [by_run[(v.key, s)] for s in seeds]
        gap = statistics.mean((r.final.val - r.final.train) / chars_per_token for r in runs)
        cells = " | ".join(f"{x:>12.4f}" for x in losses[v.key])
        print(f"  {v.label:<{width}} | {cells} | {statistics.mean(losses[v.key]):>7.4f} | "
              f"{gap:>10.3f} | {runs[0].n_params:>10,}")

    spread = max((max(xs) - min(xs) for xs in losses.values()), default=0.0)
    baseline = variants[0]
    for v in variants[1:]:
        diffs = [x - b for x, b in zip(losses[v.key], losses[baseline.key])]
        mean_diff = statistics.mean(diffs)
        wins = sum(d < 0 for d in diffs)
        print(f"\n  {v.label} - {baseline.label}, por semente: {', '.join(f'{d:+.4f}' for d in diffs)} "
              f"(média {mean_diff:+.4f} = {100 * mean_diff / statistics.mean(losses[baseline.key]):+.1f}%)")
        print(f"  {v.label} venceu em {wins} de {len(diffs)} semente(s)")
        if len(seeds) >= 2:
            print(f"  variação entre sementes (mesma variante): até {spread:.4f}")
        print(f"  veredito: {verdict(diffs, spread)}")
    return losses


def generation_stats(checkpoint: Path, data: BPEData, device: str, seed: int) -> WordStats:
    """Print a sample from a saved variant and measure 8 x ~400 characters (temperature 1.0, no top-k)."""
    model, tok, info = load_checkpoint(checkpoint, device)
    print(f"  {checkpoint.name} (passo {info.step})\n")
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode("Capitu olhou para mim e ")], device=device)  # (1, T)
    print(tok.decode(model.generate(idx, math.ceil(300 / data.chars_per_token))[0]))

    torch.manual_seed(seed)
    start = torch.full((8, 1), tok.encode("\n")[0], device=device)  # (B, 1)
    out = model.generate(start, math.ceil(400 / data.chars_per_token))  # (B, 1 + n)
    return word_stats("\n".join(tok.decode(row[1:]) for row in out), corpus_vocabulary(data.train_text))
