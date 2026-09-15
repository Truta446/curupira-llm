"""Generate text from a saved checkpoint.

Usage:
    .venv/bin/python -m scripts.generate --prompt "Capitu olhou para mim e "
    .venv/bin/python -m scripts.generate --temperature 1.2 --top-k 0 --length 800
"""

import argparse
from dataclasses import dataclass

import torch

from curupira.checkpoint import CHECKPOINT_DIR, load_checkpoint
from curupira.dataset import pick_device


@dataclass(frozen=True)
class Args:
    checkpoint: str
    prompt: str
    length: int
    temperature: float
    top_k: int | None
    seed: int | None
    device: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="best.pt", help="file inside checkpoints/")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--length", type=int, default=500,
                   help="tokens to generate (one character each for char models, ~2.6 for BPE 1024)")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=20, help="0 disables top-k")
    p.add_argument("--seed", type=int, default=None, help="fix it to get the same text again")
    p.add_argument("--device", default="auto")
    ns = p.parse_args()
    return Args(checkpoint=ns.checkpoint, prompt=ns.prompt, length=ns.length, temperature=ns.temperature,
                top_k=ns.top_k or None, seed=ns.seed, device=ns.device)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, tok, info = load_checkpoint(CHECKPOINT_DIR / args.checkpoint, device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    idx = torch.tensor([tok.encode(args.prompt)], device=device)  # (1, T)
    out = model.generate(idx, args.length, temperature=args.temperature, top_k=args.top_k)  # (1, T + length)
    print(f"[{args.checkpoint}: passo {info.step}, loss {info.val_loss:.3f} | "
          f"temperature {args.temperature}, top-k {args.top_k}]\n")
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
