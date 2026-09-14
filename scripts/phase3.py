"""Phase 3: self-attention with ONE head, step by step.

1. Run a single head on a short real sentence and print every tensor.
2. Show why the scores are divided by sqrt(head_size).
3. Train the model and compare with the bigram baseline (val 2.37).
4. Print the attention matrix again, now that it has learned something.

Usage:
    .venv/bin/python -m scripts.phase3 --device cpu
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from curupira.dataset import get_batch, load_data, pick_device
from curupira.models.attention import AttentionLM, Head
from curupira.ops import SGD
from curupira.training import estimate_loss

BIGRAM_VAL_LOSS = 2.37  # the number to beat, from phase 2


@dataclass(frozen=True)
class Args:
    device: str
    n_embd: int
    head_size: int
    batch_size: int
    block_size: int
    steps: int
    lr: float
    momentum: float
    eval_every: int
    eval_iters: int
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-embd", type=int, default=64, help="C: size of each token's vector")
    p.add_argument("--head-size", type=int, default=64, help="H: size of query/key/value")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--momentum", type=float, default=0.9, help="0.0 = plain SGD, like phase 2")
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    ns = p.parse_args()
    return Args(
        device=ns.device, n_embd=ns.n_embd, head_size=ns.head_size, batch_size=ns.batch_size,
        block_size=ns.block_size, steps=ns.steps, lr=ns.lr, momentum=ns.momentum,
        eval_every=ns.eval_every, eval_iters=ns.eval_iters, seed=ns.seed,
    )


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_attention(attention: torch.Tensor, tokens: Sequence[str]) -> None:
    """Print a (T, T) attention matrix as percentages, rows = who is looking."""
    # attention: (T, T); row t says how much position t reads from each position
    labels = [t.replace("\n", "\\n").replace(" ", "_") for t in tokens]
    print("          " + "".join(f"{lab:>5}" for lab in labels) + "   <- lido")
    for t, row in enumerate(attention.tolist()):
        cells = "".join(f"{v * 100:5.0f}" if v > 0 else "    ." for v in row)
        print(f"  {labels[t]:>5} |{cells}   soma {sum(row):.2f}")
    print("  (leia a linha: quanto o token da linha puxa de cada token da coluna)")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok, train, val = load_data()
    V = tok.vocab_size

    # ------------------------------------------------------------------
    section("1. One head, step by step, on a real sentence")
    sentence = "Dom Casmurro"
    idx = torch.tensor([tok.encode(sentence)], device=device)  # (1, T)
    T = idx.shape[1]
    C, H = 8, 4  # tiny on purpose, so the numbers fit on screen

    torch.manual_seed(args.seed)
    demo_tokens = nn.Embedding(V, C).to(device)
    demo_head = Head(n_embd=C, head_size=H, block_size=args.block_size).to(device)

    x = demo_tokens(idx)  # (1, T) -> (1, T, C)
    print(f"sentence     : {sentence!r}  ->  T = {T} tokens")
    print(f"x            : {tuple(x.shape)}   (B, T, C) = cada token virou um vetor de {C} números")
    with torch.no_grad():
        q = demo_head.query(x)  # (1, T, C) -> (1, T, H)
        k = demo_head.key(x)    # (1, T, C) -> (1, T, H)
        v = demo_head.value(x)  # (1, T, C) -> (1, T, H)
        print(f"q, k, v      : {tuple(q.shape)} cada  (B, T, H)")

        raw = q @ k.transpose(-2, -1)  # (1, T, H) @ (1, H, T) -> (1, T, T)
        print(f"scores brutos: {tuple(raw.shape)}   (B, T, T) = quanto cada token quer ler de cada token")
        scaled = raw * H**-0.5  # (1, T, T)
        masked = scaled.masked_fill(demo_head.tril[:T, :T] == 0, float("-inf"))  # (1, T, T)
        att = torch.softmax(masked, dim=-1)  # (1, T, T)
        out = att @ v  # (1, T, T) @ (1, T, H) -> (1, T, H)
        print(f"saída        : {tuple(out.shape)}   (B, T, H)")

    print("\nscores depois da escala 1/sqrt(H), ANTES da máscara (posição 0 x posição 0..T-1):")
    print("  " + "".join(f"{s:7.2f}" for s in scaled[0, 0].tolist()))
    print("\nmatriz de atenção (softmax da matriz mascarada), em %:")
    print_attention(att[0], list(sentence))
    print("\n  Repare: o triângulo superior é zero. A posição 0 só pode olhar para si mesma (100%),")
    print("  a posição 1 divide entre 0 e 1, e assim por diante. É o Curupira: só olha para trás.")
    print("  Sem treino, os pesos são aleatórios, então cada linha fica perto de dividir igualmente.")

    # ------------------------------------------------------------------
    section("2. Why divide by sqrt(head_size)")
    torch.manual_seed(args.seed)
    for h in (4, 64, 256):
        qq = torch.randn(1, T, h)  # (1, T, h)
        kk = torch.randn(1, T, h)  # (1, T, h)
        s_raw = qq @ kk.transpose(-2, -1)          # (1, T, T)
        s_scaled = s_raw * h**-0.5                 # (1, T, T)
        p_raw = torch.softmax(s_raw[0, -1], -1)    # (T,)
        p_scaled = torch.softmax(s_scaled[0, -1], -1)  # (T,)
        print(f"H={h:>4} | desvio dos scores: bruto {s_raw.std():6.2f} / escalado {s_scaled.std():5.2f}"
              f" | maior peso do softmax: bruto {p_raw.max():5.1%} / escalado {p_scaled.max():5.1%}")
    print("\n  Sem a escala, os scores crescem com sqrt(H), o softmax vira um 'tudo ou nada'")
    print("  e o gradiente some. Com a escala, eles ficam na mesma faixa para qualquer H.")

    # ------------------------------------------------------------------
    section(f"3. Training: one head vs the bigram (val {BIGRAM_VAL_LOSS})")
    torch.manual_seed(args.seed)
    model = AttentionLM(V, args.n_embd, args.head_size, args.block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} | lr {args.lr} | momentum {args.momentum}")
    splits: dict[str, torch.Tensor] = {"train": train, "val": val}
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            losses = estimate_loss(model, splits, args.batch_size, args.block_size, args.eval_iters, device)
            print(f"step {step:>5} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| {time.perf_counter() - t0:5.1f}s")
        if step == args.steps:
            break

        x_batch, y_batch = get_batch(train, args.batch_size, args.block_size, device)  # (B, T), (B, T)
        _, loss = model(x_batch, y_batch)
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()  # fills p.grad for every parameter
        optimizer.step()  # hand-written SGD with momentum

    # ------------------------------------------------------------------
    section("4. The attention matrix AFTER training")
    print_attention(model.attention_for(idx).cpu(), list(sentence))

    # ------------------------------------------------------------------
    section("5. Text generated by the one-head model (400 chars)")
    torch.manual_seed(args.seed)
    start = torch.tensor([[tok.stoi["\n"]]], device=device)  # (1, 1)
    print(tok.decode(model.generate(start, 400)[0]))


if __name__ == "__main__":
    main()
