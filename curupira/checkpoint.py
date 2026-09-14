"""Saving and restoring models.

A checkpoint holds everything needed to rebuild the model later: the weights,
the shape of the architecture, the vocabulary and where training was.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch

from curupira.dataset import ROOT
from curupira.models.transformer import GPT
from curupira.tokenizer import CharTokenizer

CHECKPOINT_DIR: Final = ROOT / "checkpoints"


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to rebuild the same architecture."""

    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    block_size: int

    def build(self) -> GPT:
        return GPT(self.vocab_size, self.n_embd, self.n_head, self.n_layer, self.block_size)


@dataclass(frozen=True)
class CheckpointInfo:
    """Where training was when the checkpoint was written."""

    step: int
    val_loss: float
    config: ModelConfig


def save_checkpoint(
    path: Path, model: GPT, config: ModelConfig, tokenizer: CharTokenizer, step: int, val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "chars": tokenizer.chars,  # so text can be decoded without the corpus
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[GPT, CharTokenizer, CheckpointInfo]:
    """Rebuild the model from a checkpoint file, ready for inference."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m scripts.phase5` first")
    blob: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**blob["config"])
    model = config.build().to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    tokenizer = CharTokenizer(blob["chars"])
    info = CheckpointInfo(step=int(blob["step"]), val_loss=float(blob["val_loss"]), config=config)
    return model, tokenizer, info
