"""Loss-curve plotting, in light and dark versions for the README."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # no window, just files
import matplotlib.pyplot as plt  # noqa: E402

from curupira.dataset import ROOT  # noqa: E402
from curupira.training import LossPoint  # noqa: E402

ASSETS_DIR: Final = ROOT / "assets"


class Theme:
    """Colors for one rendering mode, from a validated palette."""

    def __init__(self, name: str, surface: str, ink: str, muted: str, grid: str, train: str, val: str) -> None:
        self.name = name
        self.surface = surface
        self.ink = ink
        self.muted = muted
        self.grid = grid
        self.train = train  # categorical slot 1 (blue)
        self.val = val      # categorical slot 2 (orange)


LIGHT: Final = Theme("light", "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#eb6834")
DARK: Final = Theme("dark", "#1a1a19", "#ffffff", "#898781", "#2c2c2a", "#3987e5", "#d95926")


def plot_loss_curve(
    history: Sequence[LossPoint],
    references: Sequence[tuple[str, float]],
    title: str,
    path: Path,
    theme: Theme,
    ylim: tuple[float, float],
) -> None:
    """Save one train/val loss curve with horizontal reference levels."""
    steps = [h.step for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    for i, (label, value) in enumerate(references):
        ax.axhline(value, color=theme.muted, linewidth=1, linestyle=(0, (5, 4)))
        # Spread the labels horizontally: reference levels can sit very close
        # together, and stacked text would overlap.
        ax.text(steps[-1] * (0.16 + 0.30 * i), value + (ylim[1] - ylim[0]) * 0.02,
                f"{label}  {value:.2f}", color=theme.muted, fontsize=9, ha="left")

    ax.plot(steps, [h.train for h in history], color=theme.train, linewidth=2, label="treino")
    ax.plot(steps, [h.val for h in history], color=theme.val, linewidth=2, label="validação")

    # Direct labels at the end of each line, so identity is never color-alone.
    ax.annotate(f"treino {history[-1].train:.2f}", (steps[-1], history[-1].train),
                textcoords="offset points", xytext=(6, 8), color=theme.train, fontsize=10, weight="bold")
    ax.annotate(f"validação {history[-1].val:.2f}", (steps[-1], history[-1].val),
                textcoords="offset points", xytext=(6, -14), color=theme.val, fontsize=10, weight="bold")

    ax.set_title(title, color=theme.ink, fontsize=12, loc="left", pad=14)
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
    ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def save_both_themes(
    history: Sequence[LossPoint],
    references: Sequence[tuple[str, float]],
    title: str,
    stem: str,
    ylim: tuple[float, float],
) -> None:
    """Write assets/<stem>_light.png and assets/<stem>_dark.png."""
    ASSETS_DIR.mkdir(exist_ok=True)
    for theme in (LIGHT, DARK):
        plot_loss_curve(history, references, title, ASSETS_DIR / f"{stem}_{theme.name}.png", theme, ylim)
