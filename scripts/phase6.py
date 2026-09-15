"""Phase 6: generation - temperature, top-k and three checkpoints compared.

1. Load three snapshots from training (phase 5).
2. Look at the probabilities the model gives to the next character.
3. See how temperature and top-k reshape them.
4. Generate from the same prompt and seed with different settings.
5. Measure the trade-off: real words vs varied words.
6. Compare the text written at three moments of training.

Usage:
    .venv/bin/python -m scripts.phase5   # once: produces checkpoints/
    .venv/bin/python -m scripts.phase6 --device cpu
"""

import argparse
from dataclasses import dataclass
from typing import Final

import torch

from curupira.checkpoint import CHECKPOINT_DIR, CheckpointInfo, load_checkpoint
from curupira.dataset import TRAIN_FILE, VAL_FILE, pick_device
from curupira.models.transformer import GPT
from curupira.plots import save_temperature_sweep
from curupira.sampling import effective_choices, next_token_probs
from curupira.text_stats import WordStats, corpus_vocabulary, word_stats
from curupira.tokenizer import Tokenizer

SWEEP_TEMPERATURES: Final = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0)
SWEEP_SEQUENCES: Final = 8   # generated in parallel, as one batch
SWEEP_LENGTH: Final = 400    # characters per sequence
COMPARE_TEMPERATURE: Final = 0.8
COMPARE_TOP_K: Final = 20


@dataclass(frozen=True)
class Args:
    device: str
    checkpoints: tuple[str, ...]
    prompt: str
    length: int
    seed: int
    no_plot: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoints", nargs=3, default=["step01000.pt", "step02000.pt", "step04000.pt"],
                   help="three files inside checkpoints/, from early to late")
    p.add_argument("--prompt", default="Capitu olhou para mim e ")
    p.add_argument("--length", type=int, default=300, help="characters per printed sample")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-plot", action="store_true", help="skip writing the chart into assets/")
    ns = p.parse_args()
    return Args(device=ns.device, checkpoints=tuple(ns.checkpoints), prompt=ns.prompt,
                length=ns.length, seed=ns.seed, no_plot=ns.no_plot)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


@torch.no_grad()
def show_distribution(model: GPT, tok: Tokenizer, prompt: str, temperature: float,
                      top_k: int | None, device: str, rows: int = 8) -> None:
    """Print the most likely next characters as a bar chart."""
    idx = torch.tensor([tok.encode(prompt)], device=device)  # (1, T)
    logits, _ = model(idx[:, -model.block_size :])  # (1, T, V)
    probs = next_token_probs(logits[:, -1, :], temperature, top_k)[0].cpu()  # (1, V) -> (V,)
    top = torch.topk(probs, rows)  # values: (rows,), indices: (rows,)

    setting = f"temperature {temperature}" + (f" + top-k {top_k}" if top_k else "")
    kept = int((probs > 0).sum().item())
    print(f"\n  {setting}: hesitando entre ~{effective_choices(probs):.1f} caracteres "
          f"({kept} com chance > 0)")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        shown = tok.decode([i]).replace(" ", "espaço").replace("\n", "\\n")
        print(f"    {shown:>7} {p:6.1%} {'█' * round(p * 50)}")


def generate_text(model: GPT, tok: Tokenizer, prompt: str, length: int, temperature: float,
                  top_k: int | None, seed: int, device: str) -> str:
    torch.manual_seed(seed)  # same seed = same random draws, so only the settings differ
    idx = torch.tensor([tok.encode(prompt)], device=device)  # (1, T)
    out = model.generate(idx, length, temperature=temperature, top_k=top_k)  # (1, T + length)
    return tok.decode(out[0])


def sample_stats(model: GPT, tok: Tokenizer, vocabulary: set[str], temperature: float,
                 top_k: int | None, seed: int, device: str) -> WordStats:
    """Generate a batch of independent sequences and measure their words."""
    torch.manual_seed(seed)
    start = torch.full((SWEEP_SEQUENCES, 1), tok.encode("\n")[0], device=device)  # (B, 1)
    out = model.generate(start, SWEEP_LENGTH, temperature=temperature, top_k=top_k)  # (B, 1 + length)
    text = "\n".join(tok.decode(row[1:]) for row in out)  # B sequences, start token dropped
    return word_stats(text, vocabulary)


def indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    vocabulary = corpus_vocabulary(TRAIN_FILE.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    section("1. The three checkpoints")
    loaded: list[tuple[GPT, Tokenizer, CheckpointInfo]] = []
    for name in args.checkpoints:
        model, tok, info = load_checkpoint(CHECKPOINT_DIR / name, device)
        loaded.append((model, tok, info))
        print(f"  {name:<14} passo {info.step:>5} | loss de validação {info.val_loss:.4f}")
    model, tok, final_info = loaded[-1]
    print(f"\nsections 2-5 use the last one (passo {final_info.step})")

    # ------------------------------------------------------------------
    section(f"2-3. What comes after {args.prompt!r}?")
    show_distribution(model, tok, args.prompt, 1.0, None, device)
    show_distribution(model, tok, args.prompt, 0.5, None, device)
    show_distribution(model, tok, args.prompt, 1.5, None, device)
    show_distribution(model, tok, args.prompt, 1.0, 3, device)

    # ------------------------------------------------------------------
    section("4. Same prompt, same seed, different settings")
    settings: list[tuple[str, float, int | None]] = [
        ("greedy: sempre a letra mais provável", 0.0, None),
        ("temperature 0.5", 0.5, None),
        ("temperature 1.0: a distribuição do jeito que o modelo deu", 1.0, None),
        ("temperature 1.5", 1.5, None),
        ("temperature 1.0 + top-k 5", 1.0, 5),
        (f"temperature {COMPARE_TEMPERATURE} + top-k {COMPARE_TOP_K}", COMPARE_TEMPERATURE, COMPARE_TOP_K),
    ]
    for label, temperature, top_k in settings:
        text = generate_text(model, tok, args.prompt, args.length, temperature, top_k, args.seed, device)
        print(f"\n  --- {label} ---")
        print(indent(text))

    # ------------------------------------------------------------------
    section("5. Measuring the trade-off")
    val_text = VAL_FILE.read_text(encoding="utf-8")
    # The share of distinct words depends on how much text is measured, so the
    # reference uses the same amount of real text as the generated samples.
    reference = word_stats(val_text[: SWEEP_SEQUENCES * SWEEP_LENGTH], vocabulary)
    print(f"  {SWEEP_SEQUENCES} sequências de {SWEEP_LENGTH} caracteres por temperature; "
          f"palavras de 3+ letras\n")
    print(f"  {'':>13} | {'palavras reais':>14} | {'palavras distintas':>18}")
    print(f"  {'Machado real':>13} | {reference.real_pct:13.1f}% | {reference.distinct_pct:17.1f}%")
    sweep: list[tuple[float, WordStats]] = []
    for temperature in SWEEP_TEMPERATURES:
        stats = sample_stats(model, tok, vocabulary, temperature, None, args.seed, device)
        sweep.append((temperature, stats))
        print(f"  {'T = ' + str(temperature):>13} | {stats.real_pct:13.1f}% | {stats.distinct_pct:17.1f}%")
    capped = sample_stats(model, tok, vocabulary, COMPARE_TEMPERATURE, COMPARE_TOP_K, args.seed, device)
    print(f"  {f'T {COMPARE_TEMPERATURE} + k {COMPARE_TOP_K}':>13} | {capped.real_pct:13.1f}% | "
          f"{capped.distinct_pct:17.1f}%")

    # ------------------------------------------------------------------
    section(f"6. Three moments of training (temperature {COMPARE_TEMPERATURE}, top-k {COMPARE_TOP_K})")
    for m, t, info in loaded:
        stats = sample_stats(m, t, vocabulary, COMPARE_TEMPERATURE, COMPARE_TOP_K, args.seed, device)
        text = generate_text(m, t, args.prompt, args.length, COMPARE_TEMPERATURE, COMPARE_TOP_K,
                             args.seed, device)
        print(f"\n  --- passo {info.step} | loss {info.val_loss:.3f} | "
              f"palavras reais {stats.real_pct:.0f}% ---")
        print(indent(text))

    # ------------------------------------------------------------------
    section("7. Chart")
    if args.no_plot:
        print("  (--no-plot: gráfico não gerado)")
        return
    save_temperature_sweep(sweep, reference, stem="phase6_temperature")


if __name__ == "__main__":
    main()
