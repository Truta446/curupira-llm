"""Phase 7d: SwiGLU in place of the ReLU MLP.

1. ReLU vs swish: values and gradients, and ReLU units that never fire.
2. What the gate does.
3. Matching the parameter count, so the comparison is fair.
4. Train RoPE + RMSNorm + ReLU vs RoPE + RMSNorm + SwiGLU with the same seeds.
5. Compare per character, against the spread between seeds.

Usage:
    .venv/bin/python -m scripts.phase7d_swiglu                          # 2 seeds x 2 variants, use a GPU
    .venv/bin/python -m scripts.phase7d_swiglu --device cpu --seeds 1337 --steps 1000
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn

from curupira.ablation import (DEFAULT_SEEDS, BPEData, Variant, bpe_config, generation_stats, load_bpe_data,
                               print_scoreboard, run_ablation)
from curupira.checkpoint import CHECKPOINT_DIR, load_checkpoint
from curupira.dataset import get_batch, pick_device
from curupira.models.transformer import Block, FeedForward, SwiGLU, swiglu_hidden
from curupira.ops import swish

# Phase 7c, RoPE + RMSNorm + ReLU, per character: the baseline this phase must reproduce.
BASELINE_7C_PER_CHAR: Final = {1337: 1.2427, 2024: 1.2540}
BASELINE_7C_WORDS: Final = (89.1, 76.5)  # % real / % distinct words, T 1.0, no top-k
BASELINE_CHECKPOINT: Final = CHECKPOINT_DIR / "bpe1024_rope_rmsnorm_best.pt"
SWIGLU_CHECKPOINT: Final = CHECKPOINT_DIR / "bpe1024_rope_rmsnorm_swiglu_best.pt"
VARIANTS: Final = (
    Variant("relu", "MLP com ReLU", bpe_config(position="rope", norm="rmsnorm", mlp="relu")),
    Variant("swiglu", "SwiGLU", bpe_config(position="rope", norm="rmsnorm", mlp="swiglu"),
            checkpoint=SWIGLU_CHECKPOINT),
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


def demo_activations() -> None:
    """Values and slopes of ReLU and swish at a few points."""
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], requires_grad=True)  # (7,)
    relu_out = torch.relu(x)  # (7,)
    (relu_grad,) = torch.autograd.grad(relu_out.sum(), x)  # (7,) slope at each point
    swish_out = swish(x)  # (7,)
    (swish_grad,) = torch.autograd.grad(swish_out.sum(), x)  # (7,)

    print(f"  {'x':>6} | {'ReLU':>6} | {'inclinação':>10} | {'swish':>6} | {'inclinação':>10}")
    for i in range(len(x)):
        print(f"  {x[i].item():>6.1f} | {relu_out[i].item():>6.2f} | {relu_grad[i].item():>10.2f} | "
              f"{swish_out[i].item():>6.2f} | {swish_grad[i].item():>10.2f}")
    print("  Para qualquer x negativo, a inclinação da ReLU é EXATAMENTE zero: nenhum gradiente passa.")
    print("  Se um neurônio só recebe valores negativos, ele nunca mais aprende: 'morre'.")


PeakHook = Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]


def peak_hook(peaks: list[torch.Tensor], index: int) -> PeakHook:
    """A forward hook that keeps, for block `index`, the highest value each ReLU unit ever produced."""

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        # output: (B, T, 4C) after the ReLU -> amax over batch and time: (4C,)
        peaks[index] = torch.maximum(peaks[index], output.amax(dim=(0, 1)))

    return hook


@torch.no_grad()
def count_silent_relu_units(data: BPEData, device: str, batches: int = 16) -> None:
    """In the phase 7c model, how many ReLU units stay at zero for every validation token?"""
    if not BASELINE_CHECKPOINT.exists():
        print(f"  ({BASELINE_CHECKPOINT.name} não existe: rode a fase 7c para ver esta medição)")
        return
    model, _, _ = load_checkpoint(BASELINE_CHECKPOINT, device)
    peaks: list[torch.Tensor] = []  # one (4C,) tensor per block
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for i, blk in enumerate(model.blocks):
        assert isinstance(blk, Block) and isinstance(blk.ffwd, FeedForward)
        expand = blk.ffwd.net[0]
        assert isinstance(expand, nn.Linear)
        peaks.append(torch.zeros(expand.out_features, device=device))  # (4C,)
        handles.append(blk.ffwd.net[1].register_forward_hook(peak_hook(peaks, i)))

    torch.manual_seed(0)
    for _ in range(batches):
        x, _ = get_batch(data.val, 32, 128, device)  # (B, T)
        model(x)
    for handle in handles:
        handle.remove()

    silent = [int((p == 0).sum().item()) for p in peaks]  # units that never produced anything > 0
    total = sum(len(p) for p in peaks)
    print(f"\n  no modelo da fase 7c, {sum(silent)} de {total} neurônios da ReLU "
          f"({100 * sum(silent) / total:.1f}%) ficaram em zero para todos os "
          f"{batches * 32 * 128:,} tokens de validação")
    print("  (" + ", ".join(f"bloco {i}: {s}" for i, s in enumerate(silent)) + ")")


def demo_gate() -> None:
    """Four hidden units of one token: the gate scales each candidate by its own amount."""
    gate_scores = torch.tensor([-3.0, 0.0, 1.0, 3.0])  # (4,) W_gate x for one token
    candidates = torch.tensor([2.0, 2.0, 2.0, 2.0])     # (4,) W_up x for the same token
    gate = swish(gate_scores)                            # (4,)
    print(f"  W_gate x (portão)    : {[round(v, 2) for v in gate_scores.tolist()]}")
    print(f"  swish(W_gate x)      : {[round(v, 2) for v in gate.tolist()]}")
    print(f"  W_up x (candidatos)  : {[round(v, 2) for v in candidates.tolist()]}")
    print(f"  portão x candidatos  : {[round(v, 2) for v in (gate * candidates).tolist()]}")
    print("  Os mesmos candidatos saem bloqueados, atenuados ou amplificados: quanto passa em cada")
    print("  unidade é CALCULADO a partir do próprio token, em vez de ser um corte fixo em zero.")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    data = load_bpe_data()

    # ------------------------------------------------------------------
    section("1. ReLU vs swish")
    demo_activations()
    count_silent_relu_units(data, device)

    # ------------------------------------------------------------------
    section("2. The gate")
    demo_gate()

    # ------------------------------------------------------------------
    section("3. A fair fight: the same number of parameters")
    C = VARIANTS[0].config.n_embd
    relu_mlp = FeedForward(C)
    swiglu_mlp = SwiGLU(C)
    n = lambda module: sum(p.numel() for p in module.parameters())  # noqa: E731
    print(f"  MLP com ReLU : 2 matrizes {C} x {4 * C} (+ bias)      = {n(relu_mlp):,} parâmetros por bloco")
    print(f"  SwiGLU       : 3 matrizes {C} x {swiglu_hidden(C)} (sem bias)   = {n(swiglu_mlp):,} parâmetros por bloco")
    unfair = 3 * C * 4 * C
    print(f"  Se a SwiGLU usasse a mesma largura {4 * C}, teria {unfair:,} ({unfair / n(relu_mlp):.1f}x):")
    print("  qualquer ganho poderia ser só 'modelo maior'. Com 8C/3 de largura, a briga é justa.")

    # ------------------------------------------------------------------
    section(f"4. Training: {len(VARIANTS)} variants x {len(args.seeds)} seed(s), "
            f"{args.steps} steps each, on {device}")
    results = run_ablation(VARIANTS, args.seeds, data, args.steps, device)

    # ------------------------------------------------------------------
    section("5. Scoreboard: validation loss per character (lower is better)")
    losses = print_scoreboard(VARIANTS, args.seeds, results, data.chars_per_token)
    if args.steps == 4000:
        for seed, expected in BASELINE_7C_PER_CHAR.items():
            if seed in args.seeds:
                got = losses["relu"][args.seeds.index(seed)]
                print(f"\n  conferência: RoPE + RMSNorm + ReLU com semente {seed} deu {got:.4f}; "
                      f"a fase 7c tinha dado {expected:.4f}")

    # ------------------------------------------------------------------
    section("6. Text from the SwiGLU model (temperature 1.0, no top-k)")
    stats = generation_stats(SWIGLU_CHECKPOINT, data, device, seed=1337)
    print(f"\n  palavras reais {stats.real_pct:.1f}% | distintas {stats.distinct_pct:.1f}% "
          f"(fase 7c, ReLU: {BASELINE_7C_WORDS[0]}% | {BASELINE_7C_WORDS[1]}%)")


if __name__ == "__main__":
    main()
