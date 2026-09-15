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
from dataclasses import dataclass
from typing import Final

import torch

from curupira.ablation import DEFAULT_SEEDS, Variant, bpe_config, generation_stats, load_bpe_data, print_scoreboard, run_ablation
from curupira.checkpoint import CHECKPOINT_DIR
from curupira.dataset import pick_device
from curupira.ops import apply_rope, rope_tables

BASELINE_7A_PER_CHAR: Final = 1.3086  # phase 7a: learned positions, seed 1337, 4000 steps
BASELINE_7A_WORDS: Final = (78.1, 76.6)  # phase 7a: % real / % distinct words, T 1.0, no top-k
ROPE_CHECKPOINT: Final = CHECKPOINT_DIR / "bpe1024_rope_best.pt"
VARIANTS: Final = (
    Variant("learned", "posição aprendida", bpe_config(position="learned")),
    Variant("rope", "RoPE", bpe_config(position="rope"), checkpoint=ROPE_CHECKPOINT),
)


@dataclass(frozen=True)
class Args:
    device: str
    seeds: tuple[int, ...]
    steps: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
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
    data = load_bpe_data()
    print(f"\n  dados: BPE {data.tok.vocab_size}, {data.chars_per_token:.2f} caracteres por token "
          f"(os mesmos da fase 7a)")

    # ------------------------------------------------------------------
    section(f"3. Training: {len(VARIANTS)} variants x {len(args.seeds)} seed(s), "
            f"{args.steps} steps each, on {device}")
    results = run_ablation(VARIANTS, args.seeds, data, args.steps, device)

    # ------------------------------------------------------------------
    section("4. Scoreboard: validation loss per character (lower is better)")
    losses = print_scoreboard(VARIANTS, args.seeds, results, data.chars_per_token)
    if 1337 in args.seeds and args.steps == 4000:
        reproduced = losses["learned"][args.seeds.index(1337)]
        print(f"\n  conferência: posição aprendida com semente 1337 deu {reproduced:.4f}; "
              f"a fase 7a tinha dado {BASELINE_7A_PER_CHAR:.4f}")

    # ------------------------------------------------------------------
    section("5. Text from the RoPE model (temperature 1.0, no top-k)")
    stats = generation_stats(ROPE_CHECKPOINT, data, device, seed=1337)
    print(f"\n  palavras reais {stats.real_pct:.1f}% | distintas {stats.distinct_pct:.1f}% "
          f"(fase 7a, posição aprendida: {BASELINE_7A_WORDS[0]}% | {BASELINE_7A_WORDS[1]}%)")


if __name__ == "__main__":
    main()
