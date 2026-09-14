"""Phase 5: a real training loop.

AdamW written by hand, warmup + cosine decay, periodic evaluation, checkpoints
and the train-vs-validation loss curve.

Usage:
    .venv/bin/python -m scripts.phase5 --device cpu --steps 1000   # rápido
    .venv/bin/python -m scripts.phase5                             # completo (use GPU)
"""

import argparse
import time
from dataclasses import dataclass

import torch

from curupira.checkpoint import CHECKPOINT_DIR, ModelConfig, save_checkpoint
from curupira.dataset import get_batch, load_data, pick_device
from curupira.ops import AdamW
from curupira.plots import save_both_themes, save_lr_schedule
from curupira.training import LossPoint, estimate_loss, lr_at

SGD_PHASE4_VAL = 1.887  # 4 blocks trained with SGD+momentum for 1500 steps


@dataclass(frozen=True)
class Args:
    device: str
    n_embd: int
    n_head: int
    n_layer: int
    batch_size: int
    block_size: int
    steps: int
    lr: float
    min_lr_frac: float
    warmup: int
    weight_decay: float
    eval_every: int
    eval_iters: int
    ckpt_every: int
    seed: int
    no_plot: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--min-lr-frac", type=float, default=0.1, help="final lr = lr * this")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--ckpt-every", type=int, default=1000, help="0 disables periodic checkpoints")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-plot", action="store_true", help="skip writing charts into assets/")
    ns = p.parse_args()
    return Args(
        device=ns.device, n_embd=ns.n_embd, n_head=ns.n_head, n_layer=ns.n_layer,
        batch_size=ns.batch_size, block_size=ns.block_size, steps=ns.steps, lr=ns.lr,
        min_lr_frac=ns.min_lr_frac, warmup=ns.warmup, weight_decay=ns.weight_decay,
        eval_every=ns.eval_every, eval_iters=ns.eval_iters, ckpt_every=ns.ckpt_every, seed=ns.seed,
        no_plot=ns.no_plot,
    )


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok, train, val = load_data()
    splits: dict[str, torch.Tensor] = {"train": train, "val": val}
    min_lr = args.lr * args.min_lr_frac

    config = ModelConfig(
        vocab_size=tok.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
        n_layer=args.n_layer, block_size=args.block_size,
    )
    model = config.build().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------
    section("1. Setup")
    tokens_per_step = args.batch_size * args.block_size
    print(f"device            : {device}")
    print(f"modelo            : {args.n_layer} blocos x {args.n_head} cabeças, C={args.n_embd}")
    print(f"parâmetros        : {n_params:,}")
    print(f"tokens por passo  : {tokens_per_step:,} (B={args.batch_size} x T={args.block_size})")
    print(f"tokens no treino  : {args.steps * tokens_per_step:,} "
          f"({args.steps * tokens_per_step / len(train):.1f}x o corpus de treino)")
    print(f"otimizador        : AdamW à mão, lr {args.lr:g} -> {min_lr:g}, "
          f"warmup {args.warmup}, weight decay {args.weight_decay}")

    # ------------------------------------------------------------------
    section("2. The learning-rate schedule")
    schedule = [lr_at(s, args.lr, min_lr, args.warmup, args.steps) for s in range(args.steps)]
    marks = [0, args.warmup // 2, args.warmup - 1, args.steps // 2, args.steps - 1]
    for s in marks:
        stage = "warmup" if s < args.warmup else "cosine decay"
        print(f"  passo {s:>5}: lr {schedule[s]:.2e}  ({stage})")
    if not args.no_plot:
        save_lr_schedule(schedule, stem="phase5_lr")

    # ------------------------------------------------------------------
    section("3. Training")
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                      weight_decay=args.weight_decay)
    history: list[LossPoint] = []
    best_val = float("inf")
    t0 = time.perf_counter()

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, args.batch_size, args.block_size,
                                   args.eval_iters, device)
            history.append(LossPoint(step=step, train=losses["train"], val=losses["val"]))
            elapsed = time.perf_counter() - t0
            flag = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(CHECKPOINT_DIR / "best.pt", model, config, tok, step, best_val)
                flag = " <- melhor até agora, salvo"
            print(f"  step {step:>5} | lr {lr_at(step, args.lr, min_lr, args.warmup, args.steps):.2e} "
                  f"| train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:6.1f}s{flag}")

        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            # Snapshots along the way: phase 6 compares the text they generate.
            save_checkpoint(CHECKPOINT_DIR / f"step{step:05d}.pt", model, config, tok,
                            step, history[-1].val)

        if step == args.steps:
            break

        x, y = get_batch(train, args.batch_size, args.block_size, device)  # (B, T), (B, T)
        _, loss = model(x, y)  # scalar
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lr=lr_at(step, args.lr, min_lr, args.warmup, args.steps))

    # ------------------------------------------------------------------
    section("4. Scoreboard")
    at_1500 = next((h for h in history if h.step == 1500), None)
    print(f"  SGD + momento, 1500 passos (fase 4) : val {SGD_PHASE4_VAL:.4f}")
    if at_1500 is not None:
        print(f"  AdamW, mesmos 1500 passos           : val {at_1500.val:.4f}")
    print(f"  AdamW, {args.steps} passos (melhor)        : val {best_val:.4f}")
    gap = history[-1].val - history[-1].train
    print(f"  distância treino -> validação       : {gap:+.4f} "
          f"({'começando a decorar' if gap > 0.15 else 'ainda saudável'})")
    print(f"\ncheckpoints em {CHECKPOINT_DIR.name}/: "
          f"{', '.join(sorted(p.name for p in CHECKPOINT_DIR.glob('*.pt')))}")

    # ------------------------------------------------------------------
    section("5. Sample from the trained model (600 chars)")
    torch.manual_seed(args.seed)
    start = torch.tensor([[tok.stoi["\n"]]], device=device)  # (1, 1)
    print(tok.decode(model.generate(start, 600)[0]))

    # ------------------------------------------------------------------
    section("6. Loss curve")
    if args.no_plot:
        print("  (--no-plot: gráfico não gerado)")
        return
    save_both_themes(
        history,
        references=[("SGD da fase 4", SGD_PHASE4_VAL)],
        title=f"Treino de verdade: AdamW + warmup + cosine, {args.steps} passos",
        stem="phase5_loss",
        ylim=(1.0, 4.9),
    )


if __name__ == "__main__":
    main()
