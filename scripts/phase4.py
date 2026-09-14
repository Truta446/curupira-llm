"""Phase 4: the Transformer block (multi-head + MLP + residual + LayerNorm).

1. Take one block apart and print what each piece does to the shapes.
2. Measure what LayerNorm and the residual connections actually do.
3. Train 1 block vs N blocks and compare with phases 2 and 3.
4. Generate text and plot the loss curve.

Usage:
    .venv/bin/python -m scripts.phase4 --device cpu
"""

import argparse
import time
from dataclasses import dataclass

import torch

from curupira.dataset import get_batch, load_data, pick_device
from curupira.models.transformer import GPT, Block
from curupira.ops import SGD
from curupira.plots import save_both_themes
from curupira.training import LossPoint, estimate_loss

BIGRAM_VAL_LOSS = 2.367     # phase 2
ONE_HEAD_VAL_LOSS = 2.3225  # phase 3


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
    momentum: float
    eval_every: int
    eval_iters: int
    seed: int
    no_plot: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-embd", type=int, default=128, help="C: width of the residual stream")
    p.add_argument("--n-head", type=int, default=4, help="heads per block")
    p.add_argument("--n-layer", type=int, default=4, help="how many blocks to stack")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=1500)
    # Deeper stacks are more sensitive: the 0.5 that worked for one head in
    # phase 3 makes the 4-block model diverge into NaN here.
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-plot", action="store_true", help="skip writing the chart into assets/")
    ns = p.parse_args()
    return Args(
        device=ns.device, n_embd=ns.n_embd, n_head=ns.n_head, n_layer=ns.n_layer,
        batch_size=ns.batch_size, block_size=ns.block_size, steps=ns.steps, lr=ns.lr,
        momentum=ns.momentum, eval_every=ns.eval_every, eval_iters=ns.eval_iters, seed=ns.seed,
        no_plot=ns.no_plot,
    )


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def train_model(model: GPT, args: Args, splits: dict[str, torch.Tensor], device: str,
                label: str) -> list[LossPoint]:
    """Train with hand-written SGD + momentum, evaluating along the way."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{label}: {n_params:,} parâmetros")
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    history: list[LossPoint] = []
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, splits, args.batch_size, args.block_size, args.eval_iters, device)
            history.append(LossPoint(step=step, train=losses["train"], val=losses["val"]))
            print(f"  step {step:>5} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| {time.perf_counter() - t0:6.1f}s")
        if step == args.steps:
            break

        x, y = get_batch(splits["train"], args.batch_size, args.block_size, device)  # (B, T), (B, T)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return history


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok, train, val = load_data()
    V = tok.vocab_size
    splits: dict[str, torch.Tensor] = {"train": train, "val": val}

    # ------------------------------------------------------------------
    section("1. One block, piece by piece")
    torch.manual_seed(args.seed)
    block = Block(args.n_embd, args.n_head, args.block_size).to(device)
    x = torch.randn(2, 16, args.n_embd, device=device)  # (B, T, C) fake activations
    print(f"entrada x               : {tuple(x.shape)}  (B, T, C)")
    with torch.no_grad():
        normed = block.ln1(x)               # (B, T, C)
        attended = block.attn(normed)       # (B, T, C) <- 4 cabeças concatenadas + projeção
        after_attn = x + attended           # (B, T, C) residual
        thought = block.ffwd(block.ln2(after_attn))  # (B, T, C)
        out = after_attn + thought          # (B, T, C)
    print(f"ln1(x)                  : {tuple(normed.shape)}")
    print(f"multi-head attention    : {tuple(attended.shape)}  ({args.n_head} cabeças de "
          f"{args.n_embd // args.n_head} canais cada, concatenadas)")
    print(f"x + attn (residual)     : {tuple(after_attn.shape)}")
    print(f"MLP (4x mais largo)     : {tuple(thought.shape)}")
    print(f"saída do bloco          : {tuple(out.shape)}  <- mesmo shape da entrada: dá para empilhar")

    counts = {
        "atenção (4 cabeças + proj)": sum(p.numel() for p in block.attn.parameters()),
        "MLP": sum(p.numel() for p in block.ffwd.parameters()),
        "LayerNorm (ln1 + ln2)": sum(p.numel() for p in block.ln1.parameters())
        + sum(p.numel() for p in block.ln2.parameters()),
    }
    print("\nparâmetros do bloco:")
    for name, n in counts.items():
        print(f"  {name:<28} {n:>8,}")

    # ------------------------------------------------------------------
    section("2. What LayerNorm and the residual actually do")
    messy = torch.randn(2, 16, args.n_embd, device=device) * 7 + 3  # (B, T, C) escala e deslocamento feios
    with torch.no_grad():
        clean = block.ln1(messy)  # (B, T, C)
    print(f"antes do LayerNorm: média {messy.mean():+.3f}  desvio {messy.std():.3f}")
    print(f"depois            : média {clean.mean():+.3f}  desvio {clean.std():.3f}")
    print("  (cada token é normalizado sozinho, nos seus canais: nada vaza de uma posição para outra)")

    torch.manual_seed(args.seed)
    deep = GPT(V, args.n_embd, args.n_head, args.n_layer, args.block_size).to(device)
    xb, yb = get_batch(train, 8, 64, device)  # (B, T), (B, T)
    _, loss = deep(xb, yb)
    assert loss is not None
    loss.backward()
    print("\ntamanho médio do gradiente em cada bloco (depois de um backward):")
    for i, blk in enumerate(deep.blocks):
        grads = [p.grad.abs().mean().item() for p in blk.parameters() if p.grad is not None]
        print(f"  bloco {i}: {sum(grads) / len(grads):.6f}")
    print("  Com as conexões residuais, o gradiente chega aos primeiros blocos com tamanho parecido:")
    print("  o '+' da residual copia o gradiente para trás sem encolher. Sem ela, ele morreria no caminho.")
    deep.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    section(f"3. Training: 1 block vs {args.n_layer} blocks")
    torch.manual_seed(args.seed)
    shallow = GPT(V, args.n_embd, args.n_head, 1, args.block_size).to(device)
    hist_shallow = train_model(shallow, args, splits, device, label="1 bloco")

    torch.manual_seed(args.seed)
    deep = GPT(V, args.n_embd, args.n_head, args.n_layer, args.block_size).to(device)
    hist_deep = train_model(deep, args, splits, device, label=f"{args.n_layer} blocos")

    section("Placar")
    print(f"  bigram (fase 2)          val {BIGRAM_VAL_LOSS:.4f}")
    print(f"  uma cabeça (fase 3)      val {ONE_HEAD_VAL_LOSS:.4f}")
    print(f"  1 bloco Transformer      val {hist_shallow[-1].val:.4f}")
    print(f"  {args.n_layer} blocos Transformer     val {hist_deep[-1].val:.4f}")

    # ------------------------------------------------------------------
    section("4. Text generated by the deep model (500 chars)")
    torch.manual_seed(args.seed)
    start = torch.tensor([[tok.stoi["\n"]]], device=device)  # (1, 1)
    print(tok.decode(deep.generate(start, 500)[0]))

    # ------------------------------------------------------------------
    section("5. Loss curve")
    if args.no_plot:
        print("  (--no-plot: gráfico não gerado)")
        return
    save_both_themes(
        hist_deep,
        references=[("bigram", BIGRAM_VAL_LOSS), ("uma cabeça", ONE_HEAD_VAL_LOSS)],
        title=f"{args.n_layer} blocos Transformer: atenção + MLP + residual + LayerNorm",
        stem="phase4_loss",
        ylim=(1.6, 4.9),
    )


if __name__ == "__main__":
    main()
