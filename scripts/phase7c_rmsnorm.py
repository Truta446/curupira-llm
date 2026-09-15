"""Phase 7c: RMSNorm in place of LayerNorm.

1. What each normalization does to a vector, and what RMSNorm leaves out.
2. Where it goes in the model, and the parameters it removes.
3. Train RoPE + LayerNorm vs RoPE + RMSNorm with the same data, recipe and seeds.
4. Compare per character, against the spread between seeds.

Usage:
    .venv/bin/python -m scripts.phase7c_rmsnorm                          # 2 seeds x 2 variants, use a GPU
    .venv/bin/python -m scripts.phase7c_rmsnorm --device cpu --seeds 1337 --steps 1000
"""

import argparse
from dataclasses import dataclass
from typing import Final

import torch

from curupira.ablation import DEFAULT_SEEDS, Variant, bpe_config, generation_stats, load_bpe_data, print_scoreboard, run_ablation
from curupira.checkpoint import CHECKPOINT_DIR
from curupira.dataset import pick_device
from curupira.ops import LayerNorm, RMSNorm

# Phase 7b, RoPE + LayerNorm, per character: the baseline this phase must reproduce.
BASELINE_7B_PER_CHAR: Final = {1337: 1.2388, 2024: 1.2529}
BASELINE_7B_WORDS: Final = (90.2, 74.2)  # % real / % distinct words, T 1.0, no top-k
RMS_CHECKPOINT: Final = CHECKPOINT_DIR / "bpe1024_rope_rmsnorm_best.pt"
VARIANTS: Final = (
    Variant("layernorm", "RoPE + LayerNorm", bpe_config(position="rope", norm="layernorm")),
    Variant("rmsnorm", "RoPE + RMSNorm", bpe_config(position="rope", norm="rmsnorm"), checkpoint=RMS_CHECKPOINT),
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


def fmt(v: torch.Tensor) -> str:
    return "[" + ", ".join(f"{x:+.2f}" for x in v.tolist()) + "]"


def describe(v: torch.Tensor) -> str:
    rms = v.pow(2).mean().sqrt().item()
    return f"média {v.mean().item():+.2f}, desvio {v.std(unbiased=False).item():.2f}, RMS {rms:.2f}"


@torch.no_grad()
def demo_normalization() -> None:
    """Normalize the same small vectors both ways, with gamma = 1 and beta = 0."""
    layer = LayerNorm(4)
    rms = RMSNorm(4)
    examples = [
        ("um vetor qualquer", torch.tensor([2.0, -1.0, 0.5, 3.5])),
        ("o mesmo vetor, 10x maior", torch.tensor([20.0, -10.0, 5.0, 35.0])),
        ("o mesmo vetor, +5 em tudo", torch.tensor([7.0, 4.0, 5.5, 8.5])),
    ]
    for name, v in examples:
        print(f"\n  {name}: {fmt(v)}  ({describe(v)})")
        ln_out = layer(v)  # (4,) -> (4,)
        rms_out = rms(v)   # (4,) -> (4,)
        print(f"    LayerNorm -> {fmt(ln_out)}  ({describe(ln_out)})")
        print(f"    RMSNorm   -> {fmt(rms_out)}  ({describe(rms_out)})")
    print("\n  Multiplicar por 10 não muda a saída de NENHUM dos dois: os dois tiram a escala.")
    print("  Somar 5 em tudo não muda a saída do LayerNorm (ele centraliza), mas muda a do RMSNorm.")
    print("  A aposta do RMSNorm: controlar a ESCALA é o que estabiliza o treino; a média não importa tanto.")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # ------------------------------------------------------------------
    section("1. What each normalization does")
    print("  LayerNorm: y = gamma * (x - média) / desvio + beta      (centraliza e escala)")
    print("  RMSNorm:   y = gamma * x / sqrt(média de x²)           (só escala; sem beta)")
    demo_normalization()

    # ------------------------------------------------------------------
    section("2. Where it goes in the model")
    n_norms = 2 * VARIANTS[0].config.n_layer + 1
    C = VARIANTS[0].config.n_embd
    print(f"  {n_norms} normalizações: 2 por bloco x {VARIANTS[0].config.n_layer} blocos + 1 antes da saída.")
    print(f"  LayerNorm guarda gamma e beta ({2 * C} números cada); RMSNorm só gamma ({C}).")
    print(f"  Economia: {n_norms} x {C} = {n_norms * C:,} parâmetros, um pedaço minúsculo do modelo.")
    data = load_bpe_data()
    print(f"\n  dados: BPE {data.tok.vocab_size}, {data.chars_per_token:.2f} caracteres por token; "
          f"linha de base: RoPE da fase 7b")

    # ------------------------------------------------------------------
    section(f"3. Training: {len(VARIANTS)} variants x {len(args.seeds)} seed(s), "
            f"{args.steps} steps each, on {device}")
    results = run_ablation(VARIANTS, args.seeds, data, args.steps, device)

    # ------------------------------------------------------------------
    section("4. Scoreboard: validation loss per character (lower is better)")
    losses = print_scoreboard(VARIANTS, args.seeds, results, data.chars_per_token)
    if args.steps == 4000:
        for seed, expected in BASELINE_7B_PER_CHAR.items():
            if seed in args.seeds:
                got = losses["layernorm"][args.seeds.index(seed)]
                print(f"\n  conferência: RoPE + LayerNorm com semente {seed} deu {got:.4f}; "
                      f"a fase 7b tinha dado {expected:.4f}")

    # ------------------------------------------------------------------
    section("5. Text from the RMSNorm model (temperature 1.0, no top-k)")
    stats = generation_stats(RMS_CHECKPOINT, data, device, seed=1337)
    print(f"\n  palavras reais {stats.real_pct:.1f}% | distintas {stats.distinct_pct:.1f}% "
          f"(fase 7b, RoPE + LayerNorm: {BASELINE_7B_WORDS[0]}% | {BASELINE_7B_WORDS[1]}%)")


if __name__ == "__main__":
    main()
