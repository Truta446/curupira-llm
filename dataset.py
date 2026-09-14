"""Load tokenized data and sample batches.

Assumes `prepare_data.py` has already produced data/train.txt and data/val.txt.
"""

import torch

from tokenizer import CharTokenizer

TRAIN_FILE = "data/train.txt"
VAL_FILE = "data/val.txt"


def pick_device(requested: str = "auto") -> str:
    """'auto' uses cuda when available, otherwise cpu. Anything else is honored."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_data() -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """Read the texts, build the tokenizer and return (tokenizer, train, val).

    The vocabulary is built from train + val together. At char level this is
    safe (it leaks no content, only the character set) and prevents a rare
    validation character from having no id.
    """
    with open(TRAIN_FILE, encoding="utf-8") as f:
        train_text = f.read()
    with open(VAL_FILE, encoding="utf-8") as f:
        val_text = f.read()

    tok = CharTokenizer.from_text(train_text + val_text)

    # str of N_train chars -> (N_train,) integers
    train = torch.tensor(tok.encode(train_text), dtype=torch.long)
    # str of N_val chars -> (N_val,) integers
    val = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train, val


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` windows of `block_size` tokens from `data`.

    Returns (x, y), where y is x shifted one position forward in the text:
    y[b, t] is the token that comes right after x[b, :t+1]. That is the target
    of a causal model: predict the next token looking only backwards.
    """
    # data: (N,)
    # Random start positions. The (exclusive) upper bound is N - block_size so
    # that i + block_size + 1 <= N, i.e. y fits inside the tensor.
    ix: list[int] = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    # ix: B ints

    x = torch.stack([data[i : i + block_size] for i in ix])
    # each slice: (T,) -> stacked: x: (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # y: (B, T), same windows shifted 1 token ahead

    return x.to(device), y.to(device)
