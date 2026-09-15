"""Phase 7a: a BPE tokenizer written by hand, and what it does to the model.

1. Learn the merges on the training text (cached in data/).
2. Look at the first merges and the longest tokens.
3. Tokenize the same sentence with characters and with BPE.
4. Measure compression for several vocabulary sizes.
5. Train the phase 5 model on BPE tokens and compare, per CHARACTER.
6. Generate text and compare with the char-level model.

Usage:
    .venv/bin/python -m scripts.phase7a_bpe            # uses the GPU if there is one
    .venv/bin/python -m scripts.phase7a_bpe --device cpu --steps 1000
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Final

import torch

from curupira.bpe import BPETokenizer
from curupira.checkpoint import CHECKPOINT_DIR, ModelConfig, save_checkpoint
from curupira.dataset import DATA_DIR, encode_splits, load_texts, pick_device
from curupira.text_stats import corpus_vocabulary, word_stats
from curupira.tokenizer import CharTokenizer
from curupira.training import LossPoint, TrainConfig, train_model

# Phase 5/6 numbers for the same architecture and training recipe on characters.
CHAR_VAL_LOSS: Final = 1.4515
CHAR_TRAIN_LOSS: Final = 1.4128
CHAR_PARAMS: Final = 838_004
# (temperature, top-k) -> (% real words, % distinct words) for checkpoints/best.pt,
# measured with scripts.phase6.sample_stats on CPU, seed 1337.
CHAR_WORDS: Final[dict[tuple[float, int | None], tuple[float, float]]] = {
    (0.8, 20): (73.3, 75.8),
    (1.0, None): (59.7, 80.4),
}

MAX_TRAINED_VOCAB: Final = 2048  # merges are learned once up to here; smaller sizes are prefixes
COMPRESSION_SIZES: Final = (512, 1024, 2048)
SAMPLE_SEQUENCES: Final = 8
SAMPLE_CHARS: Final = 400
TEMPERATURE: Final = 0.8
TOP_K: Final = 20


@dataclass(frozen=True)
class Args:
    device: str
    vocab_size: int
    steps: int
    retrain_bpe: bool
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--retrain-bpe", action="store_true", help="ignore the cached merges in data/")
    p.add_argument("--seed", type=int, default=1337)
    ns = p.parse_args()
    return Args(device=ns.device, vocab_size=ns.vocab_size, steps=ns.steps,
                retrain_bpe=ns.retrain_bpe, seed=ns.seed)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(token: str) -> str:
    return repr(token).replace("\\n", "⏎")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    train_text, val_text = load_texts()

    # ------------------------------------------------------------------
    section("1. Learning the merges")
    target = max(MAX_TRAINED_VOCAB, args.vocab_size)
    cache = DATA_DIR / f"bpe_{target}.json"
    if cache.exists() and not args.retrain_bpe:
        full = BPETokenizer.load(cache)
        print(f"  merges carregados de {cache.relative_to(DATA_DIR.parent)}")
    else:
        t0 = time.perf_counter()
        full = BPETokenizer.train(train_text, target, alphabet=val_text)
        full.save(cache)
        print(f"  {len(full.merges)} merges aprendidos em {time.perf_counter() - t0:.1f}s "
              f"(salvos em {cache.relative_to(DATA_DIR.parent)})")
    tok = full.truncated(args.vocab_size)
    n_base = len(tok.base_chars)
    print(f"  vocabulário: {n_base} caracteres + {len(tok.merges)} merges = {tok.vocab_size} tokens")

    print("\n  os primeiros merges (os pares mais frequentes do livro):")
    for i, (a, b) in enumerate(tok.merges[:16]):
        print(f"    {i + 1:>3}. {show(tok.vocab[a]):>8} + {show(tok.vocab[b]):<8} -> {show(tok.vocab[n_base + i])}")

    longest = sorted(tok.vocab[n_base:], key=len, reverse=True)[:16]
    print("\n  os tokens mais longos:")
    print("    " + "  ".join(show(t) for t in longest))

    # ------------------------------------------------------------------
    section("2. The same sentence, two tokenizers")
    char_tok = CharTokenizer.from_text(train_text + val_text)
    sentence = "--Não consultes diccionarios, disse Capitu, olhando para mim."
    char_ids = char_tok.encode(sentence)
    print(f"  texto : {sentence!r} ({len(sentence)} caracteres)\n")
    print(f"  letras: {len(char_ids)} tokens")
    print("    |" + "|".join(char_tok.decode([i]) for i in char_ids) + "|")
    bpe_pieces = tok.pieces(sentence)
    print(f"\n  BPE   : {len(bpe_pieces)} tokens")
    print("    |" + "|".join(bpe_pieces) + "|")
    assert tok.decode(tok.encode(sentence)) == sentence, "decode(encode(s)) must give s back"

    # ------------------------------------------------------------------
    section("3. Compression: how much text fits in the same 128 tokens")
    print(f"  {'vocabulário':>11} | {'caracteres/token':>16} | {'tokens de treino':>16} | "
          f"{'contexto de 128 tokens':>22}")
    print(f"  {char_tok.vocab_size:>11} | {1.0:>16.2f} | {len(train_text):>16,} | {128:>13} caracteres")
    chars_per_token = 1.0
    train_ids: list[int] = []
    for size in sorted(set(COMPRESSION_SIZES) | {args.vocab_size}):
        sized = full.truncated(size)
        ids = sized.encode(train_text)
        ratio = len(train_text) / len(ids)
        mark = "  <- usado no treino" if size == args.vocab_size else ""
        print(f"  {size:>11} | {ratio:>16.2f} | {len(ids):>16,} | {128 * ratio:>13.0f} caracteres{mark}")
        if size == args.vocab_size:
            train_ids = ids
    decoded = tok.decode(train_ids)
    assert decoded == train_text, "BPE round trip failed on the training text"
    print("  (ida e volta conferida no texto de treino inteiro: decode(encode(texto)) == texto)")

    train, val = encode_splits(tok, train_text, val_text)
    chars_per_token = len(val_text) / len(val)
    print(f"\n  na validação: {chars_per_token:.2f} caracteres por token")

    # ------------------------------------------------------------------
    section("4. Why the loss must be compared per character")
    print(f"  loss inicial esperada (chute uniforme): letras ln({char_tok.vocab_size}) = "
          f"{math.log(char_tok.vocab_size):.2f} | BPE ln({tok.vocab_size}) = {math.log(tok.vocab_size):.2f}")
    print("  Um token BPE vale ~" + f"{chars_per_token:.1f}" + " letras: errar um token é errar várias letras de uma vez.")
    print("  Por isso: loss por caractere = loss por token / caracteres por token.")

    # ------------------------------------------------------------------
    section(f"5. Training the same model on BPE tokens ({args.steps} steps, on {device})")
    torch.manual_seed(args.seed)
    config = ModelConfig(vocab_size=tok.vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128)
    model = config.build().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    embedding_growth = n_params - CHAR_PARAMS
    print(f"  parâmetros: {n_params:,} ({embedding_growth:+,} vs. letras: só embedding e lm_head crescem)")
    ckpt_path = CHECKPOINT_DIR / f"bpe{tok.vocab_size}_best.pt"
    best: list[LossPoint] = []

    def keep_best(point: LossPoint) -> None:
        if not best or point.val < best[0].val:
            best[:] = [point]
            save_checkpoint(ckpt_path, model, config, tok, point.step, point.val)

    history = train_model(model, train, val, TrainConfig(steps=args.steps), device,
                          chars_per_token=chars_per_token, on_eval=keep_best)
    best_point = best[0]
    final = history[-1]

    # ------------------------------------------------------------------
    section("6. Scoreboard (lower is better)")
    bpe_per_char = best_point.val / chars_per_token
    print(f"  {'':<26} | {'loss/token':>10} | {'loss/caractere':>14} | {'parâmetros':>10}")
    print(f"  {'letras (fase 5)':<26} | {CHAR_VAL_LOSS:>10.4f} | {CHAR_VAL_LOSS:>14.4f} | {CHAR_PARAMS:>10,}")
    print(f"  {f'BPE {tok.vocab_size} (passo {best_point.step})':<26} | {best_point.val:>10.4f} | "
          f"{bpe_per_char:>14.4f} | {n_params:>10,}")
    change = 100 * (bpe_per_char - CHAR_VAL_LOSS) / CHAR_VAL_LOSS
    print(f"\n  loss por caractere: {change:+.1f}% em relação às letras")

    # The train -> val gap is a sign of memorization; it also has to be per character.
    gap_per_char = (final.val - final.train) / chars_per_token
    char_gap = CHAR_VAL_LOSS - CHAR_TRAIN_LOSS
    epochs = args.steps * TrainConfig().batch_size * TrainConfig().block_size / len(train)
    print(f"  distância treino -> validação, por caractere: letras {char_gap:.3f} | "
          f"BPE {gap_per_char:.3f} ({gap_per_char / char_gap:.1f}x)")
    print(f"  com BPE o livro vira {len(train):,} tokens: {args.steps} passos passam ~{epochs:.0f} vezes por ele")
    print(f"  checkpoint: {ckpt_path.relative_to(CHECKPOINT_DIR.parent)}")

    # ------------------------------------------------------------------
    section(f"7. Generated text (temperature {TEMPERATURE}, top-k {TOP_K})")
    torch.manual_seed(args.seed)
    prompt = "Capitu olhou para mim e "
    idx = torch.tensor([tok.encode(prompt)], device=device)  # (1, T)
    tokens_for_300_chars = math.ceil(300 / chars_per_token)
    print(tok.decode(model.generate(idx, tokens_for_300_chars, TEMPERATURE, TOP_K)[0]))

    vocabulary = corpus_vocabulary(train_text)
    start = torch.full((SAMPLE_SEQUENCES, 1), tok.encode("\n")[0], device=device)  # (B, 1)
    print(f"\n  {SAMPLE_SEQUENCES} sequências de ~{SAMPLE_CHARS} caracteres por configuração")
    print(f"  {'':<30} | {'palavras reais':>14} | {'palavras distintas':>18}")
    for (temperature, top_k), (char_real, char_distinct) in CHAR_WORDS.items():
        torch.manual_seed(args.seed)
        out = model.generate(start, math.ceil(SAMPLE_CHARS / chars_per_token), temperature, top_k)  # (B, 1 + n)
        stats = word_stats("\n".join(tok.decode(row[1:]) for row in out), vocabulary)
        setting = f"T {temperature}" + (f" + top-k {top_k}" if top_k else ", sem top-k")
        print(f"  {'letras, ' + setting:<30} | {char_real:>13.1f}% | {char_distinct:>17.1f}%")
        print(f"  {f'BPE {tok.vocab_size}, ' + setting:<30} | {stats.real_pct:>13.1f}% | "
              f"{stats.distinct_pct:>17.1f}%")
    print("\n  top-k 20 entre TOKENS é mais restritivo que entre letras: cada token vale várias letras.")


if __name__ == "__main__":
    main()
