"""Phase 2: bigram baseline.

1. Count every pair of consecutive characters -> the best possible bigram.
2. Train a (V, V) table by gradient descent and watch it approach that number.
3. Peek inside the learned table, generate text and plot the loss curve.

Usage:
    .venv/bin/python phase2.py
"""

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import matplotlib

matplotlib.use("Agg")  # no window, just files
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from bigram import BigramLM  # noqa: E402
from dataset import get_batch, load_data, pick_device  # noqa: E402


@dataclass(frozen=True)
class Args:
    device: str
    batch_size: int
    block_size: int
    steps: int
    lr: float
    eval_every: int
    eval_iters: int
    seed: int


@dataclass(frozen=True)
class LossPoint:
    step: int
    train: float
    val: float


@dataclass(frozen=True)
class Theme:
    """Colors for one rendering mode, from the validated reference palette."""

    name: str
    surface: str
    ink: str
    muted: str
    grid: str
    train: str  # categorical slot 1 (blue)
    val: str    # categorical slot 2 (orange)


LIGHT: Final = Theme(
    name="light", surface="#fcfcfb", ink="#0b0b0b", muted="#898781",
    grid="#e1e0d9", train="#2a78d6", val="#eb6834",
)
DARK: Final = Theme(
    name="dark", surface="#1a1a19", ink="#ffffff", muted="#898781",
    grid="#2c2c2a", train="#3987e5", val="#d95926",
)


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=50.0, help="learning rate (plain SGD)")
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    ns = p.parse_args()
    return Args(
        device=ns.device, batch_size=ns.batch_size, block_size=ns.block_size, steps=ns.steps,
        lr=ns.lr, eval_every=ns.eval_every, eval_iters=ns.eval_iters, seed=ns.seed,
    )


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


@torch.no_grad()
def estimate_loss(
    model: BigramLM,
    splits: Mapping[str, torch.Tensor],
    batch_size: int,
    block_size: int,
    iters: int,
    device: str,
) -> dict[str, float]:
    """Average loss over `iters` random batches of each split (less noisy than one batch)."""
    model.eval()
    out: dict[str, float] = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)  # (iters,)
        for k in range(iters):
            x, y = get_batch(data, batch_size, block_size, device)  # (B, T), (B, T)
            _, loss = model(x, y)
            assert loss is not None
            losses[k] = loss.item()
        out[name] = float(losses.mean().item())
    model.train()
    return out


def count_bigram_loss(train: torch.Tensor, val: torch.Tensor, V: int) -> tuple[float, float, torch.Tensor]:
    """The best bigram: probability of b after a = count(a,b) / count(a, anything)."""
    pairs = train[:-1] * V + train[1:]  # (N-1,) each pair (a, b) encoded as one int a*V+b
    counts = torch.bincount(pairs, minlength=V * V).view(V, V).float()  # (V*V,) -> (V, V)

    probs = (counts + 1) / (counts + 1).sum(dim=1, keepdim=True)  # (V, V); +1 so no pair has prob 0

    def loss_on(data: torch.Tensor) -> float:
        p = probs[data[:-1], data[1:]]  # (N-1,) prob of each actual next char
        return float(-p.log().mean().item())

    return loss_on(train), loss_on(val), probs


def plot_loss_curve(
    history: Sequence[LossPoint], count_val: float, uniform: float, path: str, theme: Theme
) -> None:
    """Save the train/val loss curve with the two reference levels."""
    steps = [h.step for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    # Reference levels: what "knowing nothing" and "the perfect bigram" cost.
    ax.axhline(uniform, color=theme.muted, linewidth=1, linestyle=(0, (5, 4)))
    ax.text(steps[-1] * 0.28, uniform + 0.07, f"chute uniforme  {uniform:.2f}", color=theme.muted,
            fontsize=9, ha="left")
    ax.axhline(count_val, color=theme.muted, linewidth=1, linestyle=(0, (5, 4)))
    ax.text(steps[-1] * 0.28, count_val + 0.07, f"bigram por contagem  {count_val:.2f}", color=theme.muted,
            fontsize=9, ha="left")

    ax.plot(steps, [h.train for h in history], color=theme.train, linewidth=2, label="treino")
    ax.plot(steps, [h.val for h in history], color=theme.val, linewidth=2, label="validação")

    # Direct labels at the end of each line, so identity is never color-alone.
    ax.annotate(f"treino {history[-1].train:.2f}", (steps[-1], history[-1].train),
                textcoords="offset points", xytext=(6, 8), color=theme.train, fontsize=10, weight="bold")
    ax.annotate(f"validação {history[-1].val:.2f}", (steps[-1], history[-1].val),
                textcoords="offset points", xytext=(6, -14), color=theme.val, fontsize=10, weight="bold")

    ax.set_title("Bigram: a loss cai de 4,75 até o limite do que pares de letras conseguem",
                 color=theme.ink, fontsize=12, loc="left", pad=14)
    ax.set_xlabel("passo de treino", color=theme.muted, fontsize=10)
    ax.set_ylabel("loss (cross-entropy)", color=theme.muted, fontsize=10)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.grid(axis="y", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in leg.get_texts():
        text.set_color(theme.ink)
    ax.set_xlim(0, steps[-1] * 1.18)
    ax.set_ylim(2.1, 5.0)

    fig.tight_layout()
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok, train, val = load_data()
    V = tok.vocab_size

    # ------------------------------------------------------------------
    section("1. Reference numbers")
    uniform = math.log(V)
    count_train, count_val, _ = count_bigram_loss(train, val, V)
    print(f"uniform guess (knows nothing)      : {uniform:.4f}")
    print(f"bigram by counting  train / val    : {count_train:.4f} / {count_val:.4f}")

    # ------------------------------------------------------------------
    section(f"2. Training the (V, V) table with SGD on {device}")
    model = BigramLM(V).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} (= {V} x {V})")
    splits: dict[str, torch.Tensor] = {"train": train, "val": val}
    history: list[LossPoint] = []
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, splits, args.batch_size, args.block_size, args.eval_iters, device)
            history.append(LossPoint(step=step, train=losses["train"], val=losses["val"]))
            print(f"step {step:>5} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| {time.perf_counter() - t0:5.1f}s")
        if step == args.steps:
            break

        x, y = get_batch(train, args.batch_size, args.block_size, device)  # (B, T), (B, T)
        _, loss = model(x, y)  # forward: scalar loss
        assert loss is not None

        model.zero_grad(set_to_none=True)
        loss.backward()  # table.grad: (V, V), how the loss changes when each score changes
        with torch.no_grad():
            for p in model.parameters():
                assert p.grad is not None
                p -= args.lr * p.grad  # plain SGD: step against the gradient

    # ------------------------------------------------------------------
    section("3. Looking inside the learned table")
    learned = torch.softmax(model.table.detach().cpu(), dim=-1)  # (V, V) scores -> probabilities
    for ch in ["q", "ç", ".", "\n"]:
        i = tok.stoi[ch]
        top = torch.topk(learned[i], 5)  # values: (5,), indices: (5,)
        guesses = "  ".join(
            f"{tok.itos[j]!r} {v:.0%}" for v, j in zip(top.values.tolist(), top.indices.tolist())
        )
        print(f"after {ch!r:>5}: {guesses}")

    # ------------------------------------------------------------------
    section("4. Text generated by the bigram (500 chars)")
    torch.manual_seed(args.seed)
    start = torch.tensor([[tok.stoi["\n"]]], device=device)  # (1, 1)
    print(tok.decode(model.generate(start, 500)[0]))

    # ------------------------------------------------------------------
    section("5. Loss curve")
    for theme in (LIGHT, DARK):
        plot_loss_curve(history, count_val, uniform, f"assets/phase2_loss_{theme.name}.png", theme)


if __name__ == "__main__":
    main()
