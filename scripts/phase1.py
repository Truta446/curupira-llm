"""Phase 1: inspect the corpus, the char-level tokenizer and get_batch.

Usage:
    .venv/bin/python -m scripts.prepare_data   # once: download and clean the corpus
    .venv/bin/python -m scripts.phase1         # inspection
"""

import argparse
import math
import time
from dataclasses import dataclass

import torch

from curupira.dataset import get_batch, load_data, pick_device


@dataclass(frozen=True)
class Args:
    device: str
    batch_size: int
    block_size: int
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--batch-size", type=int, default=32, help="B proposed for the next phases")
    p.add_argument("--block-size", type=int, default=128, help="T (context length) proposed")
    p.add_argument("--seed", type=int, default=1337)
    ns = p.parse_args()
    return Args(device=ns.device, batch_size=ns.batch_size, block_size=ns.block_size, seed=ns.seed)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)  # reproducible batches

    t0 = time.perf_counter()
    tok, train, val = load_data()
    load_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    section("1. Corpus and split")
    n_total = len(train) + len(val)
    print(f"device               : {device}")
    print(f"train tokens         : {len(train):>10,}  ({len(train) / n_total:.1%})")
    print(f"val tokens           : {len(val):>10,}  ({len(val) / n_total:.1%})")
    print(f"dtype / memory       : {train.dtype}, {train.element_size() * n_total / 1e6:.1f} MB")
    print(f"read+encode time     : {load_time:.2f}s")

    # ------------------------------------------------------------------
    section("2. Vocabulary (char-level)")
    V = tok.vocab_size
    print(f"V = {V} distinct characters:")
    print(repr("".join(tok.chars)))

    counts = torch.bincount(train, minlength=V)  # train: (N,) -> counts: (V,)
    order: list[int] = torch.argsort(counts, descending=True).tolist()  # V ids, most to least frequent
    print("\nmost frequent:")
    for i in order[:12]:
        print(f"  {tok.itos[i]!r:>6}  id={i:<3} {counts[i].item():>9,}")
    print("rarest:")
    for i in order[-10:]:
        print(f"  {tok.itos[i]!r:>6}  id={i:<3} {counts[i].item():>9,}")

    # ------------------------------------------------------------------
    section("3. encode -> decode (round trip)")
    sentence = "Não consultes diccionarios."
    ids = tok.encode(sentence)
    print(f"text  : {sentence!r}")
    print(f"ids   : {ids}")
    print(f"back  : {tok.decode(ids)!r}")
    assert tok.decode(ids) == sentence, "decode(encode(s)) should return s"

    # ------------------------------------------------------------------
    section("4. A tiny batch (B=4, T=8) to look at")
    x, y = get_batch(train, batch_size=4, block_size=8, device=device, generator=g)
    print(f"x.shape = {tuple(x.shape)}   y.shape = {tuple(y.shape)}")
    print("x =\n", x.cpu())
    print("y =\n", y.cpu())
    for b in range(4):
        print(f"  row {b}: x={tok.decode(x[b])!r:<14} y={tok.decode(y[b])!r}")

    # ------------------------------------------------------------------
    section("5. One window of T=8 tokens holds 8 prediction problems")
    for t in range(8):
        context = x[0, : t + 1]      # (t+1,)
        target = int(y[0, t].item())  # scalar
        print(f"  context {tok.decode(context)!r:<12} -> next {tok.decode([target])!r}")

    # ------------------------------------------------------------------
    section(f"6. Batch at the proposed size (B={args.batch_size}, T={args.block_size})")
    x, y = get_batch(train, args.batch_size, args.block_size, device=device, generator=g)
    print(f"x: {tuple(x.shape)} {x.dtype} on {x.device}")
    print(f"predictions per batch: B*T = {x.numel():,}")
    print(f"start of row 0       : {tok.decode(x[0, :80])!r}")
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        get_batch(train, args.batch_size, args.block_size, device=device, generator=g)
    print(f"get_batch            : {(time.perf_counter() - t0) / n * 1e3:.2f} ms/batch")

    # ------------------------------------------------------------------
    section("7. Reference number for phase 2")
    print(f"A model that guesses uniformly among the {V} characters has")
    print(f"loss (cross-entropy) = -ln(1/V) = ln({V}) = {math.log(V):.4f}")


if __name__ == "__main__":
    main()
