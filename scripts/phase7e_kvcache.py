"""Phase 7e: KV cache - the same text, without recomputing the past.

1. Count the repeated work of plain generation.
2. Check that the cache changes nothing: loss, logits and generated text (two trained models).
3. Time both ways on the trained modern model.
4. Time both ways as the text gets longer (same architecture with room for 1024 tokens).
5. Measure what the cache costs in memory.

Usage:
    .venv/bin/python -m scripts.phase7e_kvcache
    .venv/bin/python -m scripts.phase7e_kvcache --timing-devices cpu --lengths 128 256
"""

import argparse
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Final

import torch

from curupira.checkpoint import CHECKPOINT_DIR, load_checkpoint
from curupira.dataset import load_texts, pick_device
from curupira.models.transformer import GPT, ModelCache
from curupira.ops import cross_entropy
from curupira.tokenizer import Tokenizer

MODERN_CHECKPOINT: Final = CHECKPOINT_DIR / "bpe1024_rope_rmsnorm_swiglu_best.pt"  # phase 7d
CHAR_CHECKPOINT: Final = CHECKPOINT_DIR / "step04000.pt"  # phase 5: characters, learned positions
PROMPT: Final = "Capitu olhou para mim e "


@dataclass(frozen=True)
class Args:
    device: str
    timing_devices: tuple[str, ...]
    repeats: int
    lengths: tuple[int, ...]


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", help="device for the exactness checks")
    default_timing = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    p.add_argument("--timing-devices", nargs="+", default=default_timing)
    p.add_argument("--repeats", type=int, default=5, help="interleaved timing rounds per measurement")
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1000],
                   help="generated lengths for the scaling test (at most 1023)")
    ns = p.parse_args()
    return Args(device=ns.device, timing_devices=tuple(ns.timing_devices), repeats=ns.repeats,
                lengths=tuple(ns.lengths))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def demo_repeated_work() -> None:
    print("  Para escrever o token n, a geração normal passa de novo os n tokens pelo modelo inteiro,")
    print("  mesmo que as keys e values dos n-1 primeiros já tenham sido calculadas no passo anterior.\n")
    print(f"  {'tokens gerados':>14} | {'posições processadas, sem cache':>31} | {'com cache':>9}")
    for n in (16, 32, 64, 128, 1000):
        print(f"  {n:>14} | {n * (n + 1) // 2:>31,} | {n:>9,}")
    print("\n  Sem cache o trabalho cresce com o QUADRADO do comprimento; com cache, em linha reta.")


@torch.no_grad()
def cached_logits_one_by_one(model: GPT, x: torch.Tensor) -> torch.Tensor:
    """Feed the sequence one token at a time through the cache and collect every step's scores."""
    # x: (1, T)
    cache: ModelCache | None = None
    steps: list[torch.Tensor] = []
    for t in range(x.shape[1]):
        logits, cache = model.forward_cached(x[:, t : t + 1], cache)  # (1, 1, V)
        steps.append(logits)
    return torch.cat(steps, dim=1)  # T tensors of (1, 1, V) -> (1, T, V)


@torch.no_grad()
def check_exactness(label: str, model: GPT, tok: Tokenizer, val_text: str, device: str) -> None:
    T = model.block_size
    ids = tok.encode(val_text[: 8 * T])[: T + 1]  # a real validation window of T+1 tokens
    x = torch.tensor([ids[:-1]], device=device)  # (1, T)
    y = torch.tensor([ids[1:]], device=device)   # (1, T)

    full_logits, full_loss = model(x, y)  # every position at once, like in training: (1, T, V)
    cached_logits = cached_logits_one_by_one(model, x)  # token by token through the cache: (1, T, V)
    assert full_loss is not None
    cached_loss = cross_entropy(cached_logits, y)

    prompt = torch.tensor([tok.encode(PROMPT)], device=device)  # (1, P)
    new = T - prompt.shape[1]  # fill the window exactly
    greedy_plain = model.generate(prompt, new, temperature=0)
    greedy_cached = model.generate_cached(prompt, new, temperature=0)
    torch.manual_seed(1337)
    sampled_plain = model.generate(prompt, new, temperature=1.0)
    torch.manual_seed(1337)
    sampled_cached = model.generate_cached(prompt, new, temperature=1.0)

    print(f"\n  {label}")
    print(f"    loss num trecho de validação: sem cache {full_loss.item():.6f} | "
          f"com cache {cached_loss.item():.6f}")
    print(f"    maior diferença entre os {T * tok.vocab_size:,} logits: "
          f"{(full_logits - cached_logits).abs().max().item():.1e}")
    print(f"    {new} tokens gerados, greedy: idênticos? {torch.equal(greedy_plain, greedy_cached)}")
    print(f"    {new} tokens gerados, sorteando com a mesma semente: "
          f"idênticos? {torch.equal(sampled_plain, sampled_cached)}")
    print(f"    início do texto: {tok.decode(sampled_cached[0])[:110]!r}")


def timed(fn: Callable[[], object], device: str) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def compare_speed(model: GPT, prompt: torch.Tensor, new_tokens: int, device: str,
                  repeats: int) -> tuple[float, float, float]:
    """Median seconds without and with cache, and the worst max/min ratio across rounds."""
    plain: Callable[[], object] = lambda: model.generate(prompt, new_tokens, temperature=0)  # noqa: E731
    cached: Callable[[], object] = lambda: model.generate_cached(prompt, new_tokens, temperature=0)  # noqa: E731
    timed(plain, device)
    timed(cached, device)  # warm-up: first calls pay one-off costs
    rounds: dict[str, list[float]] = {"plain": [], "cached": []}
    for _ in range(repeats):  # interleaved, so background noise hits both alike
        rounds["plain"].append(timed(plain, device))
        rounds["cached"].append(timed(cached, device))
    wobble = max(max(ts) / min(ts) for ts in rounds.values())
    return statistics.median(rounds["plain"]), statistics.median(rounds["cached"]), wobble


def reliability_note(device: str, wobble: float) -> str:
    """Flag timings that cannot be trusted.

    The load average includes this very benchmark (on CPU it uses every core), so
    a busy machine only shows up as MORE runnable threads than cores. The most
    direct evidence is the timing itself: rounds of the same measurement that disagree.
    """
    problems: list[str] = []
    if wobble > 1.5:
        problems.append(f"rodadas da mesma medição variaram {wobble:.1f}x")
    if device == "cpu":
        load = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        if load > cores:
            problems.append(f"carga {load:.0f} em {cores} núcleos")
    if not problems:
        return ""
    return f"  ({device}: {'; '.join(problems)} - tempos pouco confiáveis)"


def print_speed_row(label: str, plain: float, cached: float, tokens: int, wobble: float) -> None:
    print(f"  {label:<22} | {plain:>8.3f} s | {cached:>8.3f} s | {tokens / plain:>7.0f} tok/s | "
          f"{tokens / cached:>7.0f} tok/s | {plain / cached:>5.1f}x | {wobble:>4.2f}x")


def speed_header() -> None:
    print(f"  {'':<22} | {'sem cache':>10} | {'com cache':>10} | {'sem cache':>13} | {'com cache':>13} | "
          f"{'ganho':>6} | {'oscil.':>6}")


def cache_bytes(cache: ModelCache) -> int:
    return sum(t.numel() * t.element_size() for layer in cache for kv in layer for t in kv)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    _, val_text = load_texts()

    # ------------------------------------------------------------------
    section("1. The repeated work")
    demo_repeated_work()

    # ------------------------------------------------------------------
    section(f"2. Does the cache change anything? (on {device})")
    modern, modern_tok, _ = load_checkpoint(MODERN_CHECKPOINT, device)
    check_exactness("modelo moderno da 7d (BPE, RoPE, RMSNorm, SwiGLU)", modern, modern_tok, val_text, device)
    char_model, char_tok, _ = load_checkpoint(CHAR_CHECKPOINT, device)
    check_exactness("modelo de letras da fase 5 (posição aprendida, LayerNorm, ReLU)", char_model,
                    char_tok, val_text, device)
    print("\n  O cache não muda o que o modelo calcula, só evita refazer contas: a loss é a mesma.")

    # ------------------------------------------------------------------
    section("3. Speed on the trained modern model (window of 128 tokens, greedy)")
    speed_header()
    for timing_device in args.timing_devices:
        model, tok, _ = load_checkpoint(MODERN_CHECKPOINT, timing_device)
        prompt = torch.tensor([tok.encode(PROMPT)], device=timing_device)  # (1, P)
        new = model.block_size - prompt.shape[1]
        plain, cached, wobble = compare_speed(model, prompt, new, timing_device, args.repeats)
        print_speed_row(f"{new} tokens, {timing_device}", plain, cached, new, wobble)
        note = reliability_note(timing_device, wobble)
        if note:
            print(note)

    # ------------------------------------------------------------------
    section("4. Longer texts: the same architecture with room for 1024 tokens")
    print("  Pesos aleatórios, só para medir tempo: a velocidade não depende do que o modelo aprendeu.\n")
    long_config = replace(load_checkpoint(MODERN_CHECKPOINT, "cpu")[2].config, block_size=1024)
    speed_header()
    for timing_device in args.timing_devices:
        torch.manual_seed(0)
        long_model = long_config.build().to(timing_device).eval()
        start = torch.zeros((1, 1), dtype=torch.long, device=timing_device)  # (1, 1)
        worst_wobble = 1.0
        for n in sorted(args.lengths):
            # At least 2 rounds: with a single one there is no spread to judge the timing by.
            plain, cached, wobble = compare_speed(long_model, start, n, timing_device, max(2, args.repeats // 2))
            print_speed_row(f"{n} tokens, {timing_device}", plain, cached, n, wobble)
            worst_wobble = max(worst_wobble, wobble)
        note = reliability_note(timing_device, worst_wobble)
        if note:
            print(note)

    # ------------------------------------------------------------------
    section("5. What the cache costs in memory")
    prompt = torch.tensor([modern_tok.encode(val_text[:2000])[: modern.block_size]], device=device)  # (1, 128)
    with torch.no_grad():
        _, cache = modern.forward_cached(prompt, None)
    n_layers, n_heads = len(cache), len(cache[0])
    head_size = cache[0][0][0].shape[-1]
    weights = sum(p.numel() * p.element_size() for p in modern.parameters())
    measured = cache_bytes(cache)
    print(f"  cache medido com 128 tokens: {measured / 1024:,.0f} KiB "
          f"= 2 (key e value) x {n_layers} blocos x {n_heads} cabeças x {head_size} números x 128 tokens x 4 bytes")
    print(f"  pesos do modelo: {weights / 1024:,.0f} KiB")
    print("  A mesma conta para um modelo grande (32 blocos, 32 cabeças de 128, 4096 tokens, 2 bytes por número):")
    big = 2 * 32 * 32 * 128 * 4096 * 2
    print(f"  {big / 1024**3:.1f} GiB por conversa. O KV-cache troca memória por velocidade.")


if __name__ == "__main__":
    main()
