"""Loss-curve plotting, in light and dark versions for the README."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # no window, just files
import matplotlib.pyplot as plt  # noqa: E402

from curupira.dataset import ROOT  # noqa: E402
from curupira.text_stats import WordStats  # noqa: E402
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
        # A halo in the surface color keeps the label readable where it crosses
        # the curve, and the labels are spread horizontally because reference
        # levels can sit very close to each other.
        ax.text(steps[-1] * (0.16 + 0.30 * i), value + (ylim[1] - ylim[0]) * 0.025,
                f"{label}  {value:.2f}", color=theme.muted, fontsize=9, ha="left",
                bbox={"facecolor": theme.surface, "edgecolor": "none", "pad": 1.5})

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


def plot_lr_schedule(schedule: Sequence[float], path: Path, theme: Theme) -> None:
    """Save the learning-rate curve: warmup ramp then cosine decay."""
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    ax.plot(range(len(schedule)), schedule, color=theme.train, linewidth=2)
    peak = max(range(len(schedule)), key=lambda i: schedule[i])
    ax.annotate(f"pico {schedule[peak]:.1e}", (peak, schedule[peak]), textcoords="offset points",
                xytext=(12, 10), color=theme.train, fontsize=10, weight="bold")

    ax.set_title("Taxa de aprendizado: warmup linear e depois cosine decay",
                 color=theme.ink, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("passo de treino", color=theme.muted, fontsize=10)
    ax.set_ylabel("learning rate", color=theme.muted, fontsize=10)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.grid(axis="y", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.set_ylim(0, max(schedule) * 1.2)

    fig.tight_layout()
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def save_lr_schedule(schedule: Sequence[float], stem: str) -> None:
    """Write assets/<stem>_light.png and assets/<stem>_dark.png."""
    ASSETS_DIR.mkdir(exist_ok=True)
    for theme in (LIGHT, DARK):
        plot_lr_schedule(schedule, ASSETS_DIR / f"{stem}_{theme.name}.png", theme)


def plot_temperature_sweep(
    points: Sequence[tuple[float, WordStats]], reference: WordStats, path: Path, theme: Theme
) -> None:
    """Share of real words and of distinct words, as the temperature goes up."""
    temps = [t for t, _ in points]
    real = [s.real_pct for _, s in points]
    distinct = [s.distinct_pct for _, s in points]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    ax.axhline(reference.real_pct, color=theme.muted, linewidth=1, linestyle=(0, (5, 4)))
    ax.text(temps[0], reference.real_pct + 2.5, f"Machado de verdade: {reference.real_pct:.0f}% reais",
            color=theme.muted, fontsize=9, ha="left",
            bbox={"facecolor": theme.surface, "edgecolor": "none", "pad": 1.5})

    ax.plot(temps, real, color=theme.train, linewidth=2, marker="o", markersize=5,
            label="palavras que existem nos livros")
    ax.plot(temps, distinct, color=theme.val, linewidth=2, marker="o", markersize=5,
            label="palavras diferentes entre si")

    ax.annotate(f"reais {real[-1]:.0f}%", (temps[-1], real[-1]), textcoords="offset points",
                xytext=(8, -4), color=theme.train, fontsize=10, weight="bold")
    ax.annotate(f"distintas {distinct[-1]:.0f}%", (temps[-1], distinct[-1]), textcoords="offset points",
                xytext=(8, -4), color=theme.val, fontsize=10, weight="bold")

    ax.set_title("Temperature: mais alta = mais variedade, e mais palavras inventadas",
                 color=theme.ink, fontsize=12, loc="left", pad=14)
    ax.set_xlabel("temperature", color=theme.muted, fontsize=10)
    ax.set_ylabel("% das palavras geradas (3+ letras)", color=theme.muted, fontsize=10)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.grid(axis="y", color=theme.grid, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=10)
    for text in leg.get_texts():
        text.set_color(theme.ink)
    span = temps[-1] - temps[0]
    ax.set_xlim(temps[0] - span * 0.04, temps[-1] + span * 0.22)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def save_temperature_sweep(
    points: Sequence[tuple[float, WordStats]], reference: WordStats, stem: str
) -> None:
    """Write assets/<stem>_light.png and assets/<stem>_dark.png."""
    ASSETS_DIR.mkdir(exist_ok=True)
    for theme in (LIGHT, DARK):
        plot_temperature_sweep(points, reference, ASSETS_DIR / f"{stem}_{theme.name}.png", theme)


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
