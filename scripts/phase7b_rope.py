"""Phase 7b: RoPE (rotary position embeddings) in place of learned positions.

1. What a rotation does: frequencies, lengths preserved, relative positions.
2. Where it goes in the model, and the parameters it removes.
3. Train learned positions vs RoPE with the same data, recipe and seeds.
4. Compare per character, against the spread between seeds.

Usage:
    .venv/bin/python -m scripts.phase7b_rope                          # 2 seeds x 2 variants, use a GPU
    .venv/bin/python -m scripts.phase7b_rope --device cpu --seeds 1337 --steps 1000
"""

import argparse
import math
import statistics
from dataclasses import dataclass
from typing import Final

import torch

from curupira.bpe import load_or_train
from curupira.checkpoint import CHECKPOINT_DIR, ModelConfig, load_checkpoint, save_checkpoint
from curupira.dataset import DATA_DIR, encode_splits, load_texts, pick_device
from curupira.models.transformer import PositionKind
from curupira.ops import apply_rope, rope_tables
from curupira.text_stats import corpus_vocabulary, word_stats
from curupira.tokenizer import Tokenizer
from curupira.training import LossPoint, TrainConfig, train_model

VOCAB_SIZE: Final = 1024
BPE_CACHE: Final = DATA_DIR / "bpe_2048.json"
BASELINE_7A_PER_CHAR: Final = 1.3086  # phase 7a: learned positions, seed 1337, 4000 steps
BASELINE_7A_WORDS: Final = (78.1, 76.6)  # phase 7a: % real / % distinct words, T 1.0, no top-k
CHECKPOINT_SEED: Final = 1337  # the RoPE run with this seed is saved
ROPE_CHECKPOINT: Final = CHECKPOINT_DIR / f"bpe{VOCAB_SIZE}_rope_best.pt"
VARIANTS: Final[tuple[PositionKind, ...]] = ("learned", "rope")
LABELS: Final[dict[PositionKind, str]] = {"learned": "posição aprendida", "rope": "RoPE"}


@dataclass(frozen=True)
class Args:
    device: str
    seeds: tuple[int, ...]
    steps: int


@dataclass(frozen=True)
class RunResult:
    variant: PositionKind
    seed: int
    best: LossPoint
    final: LossPoint
    n_params: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024],
                   help="each seed trains both variants; 2+ seeds show how much is just luck")
    p.add_argument("--steps", type=int, default=4000)
    ns = p.parse_args()
    return Args(device=ns.device, seeds=tuple(ns.seeds), steps=ns.steps)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def demo_rotation() -> None:
    """Show, with small numbers, the three properties that make RoPE work."""
    H = 8  # 4 pairs of channels, small enough to print
    cos, sin = rope_tables(H, max_len=200)  # (200, 4), (200, 4)
    theta = torch.atan2(sin[1], cos[1])  # (4,) the angle each pair turns per position = its frequency
    print("  cada par de canais gira com uma velocidade (radianos por posição):")
    print("    " + "  ".join(f"par {i}: {t:.3f}" for i, t in enumerate(theta.tolist())))
    print(f"    o par 0 dá uma volta completa a cada ~{2 * math.pi / theta[0].item():.0f} posições; "
          f"o último, a cada ~{2 * math.pi / theta[-1].item():.0f}.")

    torch.manual_seed(0)
    q = torch.randn(H)  # (H,) one query vector
    k = torch.randn(H)  # (H,) one key vector
    q_at = apply_rope(q.expand(200, H), cos, sin)  # (200, H): the same query placed at every position
    k_at = apply_rope(k.expand(200, H), cos, sin)  # (200, H): the same key placed at every position

    print(f"\n  girar não muda o tamanho do vetor: |q| = {q.norm():.4f}, "
          f"|q girado na posição 57| = {q_at[57].norm():.4f}")

    print("\n  o score depende só da DISTÂNCIA entre as posições, não de onde elas estão:")
    print(f"    {'query na posição':>17} | {'key na posição':>14} | {'distância':>9} | {'q · k':>8}")
    for m, n in [(3, 1), (50, 48), (150, 148), (10, 5), (120, 115), (5, 5), (190, 190)]:
        score = torch.dot(q_at[m], k_at[n]).item()  # (H,) · (H,) -> scalar
        print(f"    {m:>17} | {n:>14} | {m - n:>9} | {score:>8.4f}")
    print("  Com posição aprendida, o modelo teria que aprender separadamente que 'a palavra")
    print("  anterior' na posição 3 e na posição 150 são a mesma relação. Com RoPE isso vem de graça.")


def train_variant(
    variant: PositionKind,
    seed: int,
    tok: Tokenizer,
    train: torch.Tensor,
    val: torch.Tensor,
    steps: int,
    device: str,
    chars_per_token: float,
) -> RunResult:
    torch.manual_seed(seed)
    config = ModelConfig(vocab_size=tok.vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128,
                         position=variant)
    model = config.build().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {LABELS[variant]}, semente {seed}: {n_params:,} parâmetros")
    save_best = variant == "rope" and seed == CHECKPOINT_SEED
    best: list[LossPoint] = []

    def keep_best(point: LossPoint) -> None:
        if not best or point.val < best[0].val:
            best[:] = [point]
            if save_best:
                save_checkpoint(ROPE_CHECKPOINT, model, config, tok, point.step, point.val)

    history = train_model(model, train, val, TrainConfig(steps=steps), device,
                          chars_per_token=chars_per_token, on_eval=keep_best)
    return RunResult(variant=variant, seed=seed, best=best[0], final=history[-1], n_params=n_params)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # ------------------------------------------------------------------
    section("1. What a rotation does")
    demo_rotation()

    # ------------------------------------------------------------------
    section("2. Where it goes in the model")
    print("  posição aprendida: x = embedding do token + embedding da posição (um vetor treinável por slot)")
    print("  RoPE:              x = embedding do token, só isso; em CADA cabeça de CADA bloco,")
    print("                     query e key são giradas pela posição antes do produto escalar.")
    print("                     O value não gira: a posição decide QUEM olhar, não O QUE passar adiante.")

    train_text, val_text = load_texts()
    tok = load_or_train(BPE_CACHE, train_text, VOCAB_SIZE, alphabet=val_text)
    train, val = encode_splits(tok, train_text, val_text)
    chars_per_token = len(val_text) / len(val)
    print(f"\n  dados: BPE {tok.vocab_size}, {chars_per_token:.2f} caracteres por token (os mesmos da fase 7a)")

    # ------------------------------------------------------------------
    section(f"3. Training: {len(VARIANTS)} variants x {len(args.seeds)} seed(s), "
            f"{args.steps} steps each, on {device}")
    results: list[RunResult] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            results.append(train_variant(variant, seed, tok, train, val, args.steps, device, chars_per_token))

    # ------------------------------------------------------------------
    section("4. Scoreboard: validation loss per character (lower is better)")
    per_char: dict[PositionKind, list[float]] = {v: [] for v in VARIANTS}
    gaps: dict[PositionKind, list[float]] = {v: [] for v in VARIANTS}
    params: dict[PositionKind, int] = {}
    for r in results:
        per_char[r.variant].append(r.best.val / chars_per_token)
        gaps[r.variant].append((r.final.val - r.final.train) / chars_per_token)
        params[r.variant] = r.n_params

    header = " | ".join(f"semente {s:>4}" for s in args.seeds)
    print(f"  {'':<18} | {header} | {'média':>7} | {'treino→val':>10} | {'parâmetros':>10}")
    for v in VARIANTS:
        cells = " | ".join(f"{x:>12.4f}" for x in per_char[v])
        print(f"  {LABELS[v]:<18} | {cells} | {statistics.mean(per_char[v]):>7.4f} | "
              f"{statistics.mean(gaps[v]):>10.3f} | {params[v]:>10,}")
    if CHECKPOINT_SEED in args.seeds and args.steps == 4000:
        reproduced = per_char["learned"][args.seeds.index(CHECKPOINT_SEED)]
        print(f"\n  conferência: posição aprendida com semente {CHECKPOINT_SEED} deu {reproduced:.4f}; "
              f"a fase 7a tinha dado {BASELINE_7A_PER_CHAR:.4f}")

    diffs = [r - l for l, r in zip(per_char["learned"], per_char["rope"])]  # one difference per seed
    mean_diff = statistics.mean(diffs)
    rope_wins = sum(d < 0 for d in diffs)
    print(f"\n  RoPE - aprendida, por semente: {', '.join(f'{d:+.4f}' for d in diffs)} "
          f"(média {mean_diff:+.4f} = {100 * mean_diff / statistics.mean(per_char['learned']):+.1f}%)")
    print(f"  RoPE venceu em {rope_wins} de {len(diffs)} semente(s)")
    if len(args.seeds) >= 2:
        spread = max(max(xs) - min(xs) for xs in per_char.values())
        print(f"  variação entre sementes (mesma variante): até {spread:.4f}")
        same_sign = rope_wins in (0, len(diffs))
        if same_sign and abs(mean_diff) > spread:
            verdict = "efeito consistente e maior que a variação entre sementes"
        elif same_sign:
            verdict = "mesmo sinal em todas as sementes, mas do tamanho da variação entre elas"
        else:
            verdict = "o sinal muda entre sementes: não dá para separar do acaso"
        print(f"  veredito: {verdict}")

    # ------------------------------------------------------------------
    if CHECKPOINT_SEED not in args.seeds:
        return
    section("5. Text from the RoPE model (temperature 1.0, no top-k)")
    model, rope_tok, info = load_checkpoint(ROPE_CHECKPOINT, device)
    print(f"  {ROPE_CHECKPOINT.relative_to(CHECKPOINT_DIR.parent)} (passo {info.step})\n")
    torch.manual_seed(CHECKPOINT_SEED)
    idx = torch.tensor([rope_tok.encode("Capitu olhou para mim e ")], device=device)  # (1, T)
    print(rope_tok.decode(model.generate(idx, math.ceil(300 / chars_per_token))[0]))

    torch.manual_seed(CHECKPOINT_SEED)
    start = torch.full((8, 1), rope_tok.encode("\n")[0], device=device)  # (B, 1)
    out = model.generate(start, math.ceil(400 / chars_per_token))  # (B, 1 + n)
    stats = word_stats("\n".join(rope_tok.decode(row[1:]) for row in out), corpus_vocabulary(train_text))
    print(f"\n  palavras reais {stats.real_pct:.1f}% | distintas {stats.distinct_pct:.1f}% "
          f"(fase 7a, posição aprendida: {BASELINE_7A_WORDS[0]}% | {BASELINE_7A_WORDS[1]}%)")


if __name__ == "__main__":
    main()
