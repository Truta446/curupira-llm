"""Talk to a trained curupira model: type the beginning of a text and it writes the rest.

It is NOT a chatbot. It was only ever trained to continue Machado de Assis, so it
does not answer questions or follow instructions. Commands start with "/".

Usage:
    .venv/bin/python -m scripts.chat
    .venv/bin/python -m scripts.chat --checkpoint best.pt --temperature 1.0
"""

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Final

import torch

from curupira.checkpoint import CHECKPOINT_DIR, CheckpointInfo, load_checkpoint
from curupira.dataset import pick_device
from curupira.models.transformer import GPT
from curupira.tokenizer import Tokenizer

PREFERRED_CHECKPOINTS: Final = ("bpe1024_rope_best.pt", "best.pt")  # best measured model first

HELP: Final = """
  Digite o começo de um texto e aperte Enter: o modelo continua escrevendo.
  Ele imita Machado de Assis. NÃO responde perguntas nem obedece pedidos:
  "--Capitu, disse eu, " funciona muito melhor que "Me conte sobre Capitu".

  /mais              continua o último texto de onde parou
  /temp 0.8          temperature: baixa = mais seguro, alta = mais criativo, 0 = sempre o mais provável
  /topk 20           sorteia só entre os 20 tokens mais prováveis (0 desliga)
  /tamanho 150       quantos tokens escrever por vez
  /semente 42        repete sempre o mesmo sorteio (/semente off volta ao acaso)
  /modelos           lista os checkpoints disponíveis
  /modelo best.pt    troca de modelo
  /config            mostra o modelo e as configurações atuais
  /ajuda             mostra esta ajuda
  /sair              sai (Ctrl+D também)

  Ctrl+C interrompe a escrita no meio.
"""


@dataclass(frozen=True)
class Args:
    checkpoint: str | None
    device: str
    temperature: float
    top_k: int | None
    length: int
    seed: int | None


@dataclass(frozen=True)
class Settings:
    temperature: float
    top_k: int | None
    length: int
    seed: int | None


@dataclass
class Session:
    """Everything that can change while the program runs."""

    name: str
    model: GPT
    tok: Tokenizer
    info: CheckpointInfo
    settings: Settings
    device: str
    last_text: str = ""


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help=f"file inside checkpoints/ (default: the first that exists of {', '.join(PREFERRED_CHECKPOINTS)})")
    p.add_argument("--device", default="auto")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    p.add_argument("--length", type=int, default=150, help="tokens written per turn")
    p.add_argument("--seed", type=int, default=None)
    ns = p.parse_args()
    return Args(checkpoint=ns.checkpoint, device=ns.device, temperature=ns.temperature,
                top_k=ns.top_k or None, length=ns.length, seed=ns.seed)


def available_checkpoints() -> list[str]:
    return sorted(p.name for p in CHECKPOINT_DIR.glob("*.pt"))


def pick_checkpoint(requested: str | None) -> str | None:
    if requested is not None:
        return requested
    available = set(available_checkpoints())
    return next((name for name in PREFERRED_CHECKPOINTS if name in available), None)


def load_into(session: Session | None, name: str, device: str, settings: Settings) -> Session:
    model, tok, info = load_checkpoint(CHECKPOINT_DIR / name, device)
    if session is None:
        return Session(name=name, model=model, tok=tok, info=info, settings=settings, device=device)
    session.name, session.model, session.tok, session.info = name, model, tok, info
    session.last_text = ""  # another tokenizer may not even be able to encode it
    return session


def describe(session: Session) -> None:
    c = session.info.config
    n_params = sum(p.numel() for p in session.model.parameters())
    s = session.settings
    print(f"\n  modelo   : {session.name} ({n_params:,} parâmetros, passo {session.info.step}, "
          f"loss {session.info.val_loss:.3f} por token)")
    print(f"  arquitetura: vocabulário {c.vocab_size}, {c.n_layer} blocos, posição {c.position}, "
          f"{c.norm}, MLP {c.mlp}, janela de {c.block_size} tokens")
    print(f"  escrita  : temperature {s.temperature}, top-k {s.top_k or 'desligado'}, "
          f"{s.length} tokens por vez, semente {s.seed if s.seed is not None else 'aleatória'}")


def unknown_characters(tok: Tokenizer, text: str) -> list[str]:
    """Characters of `text` that the tokenizer cannot encode."""
    unknown: set[str] = set()
    for ch in set(text):
        try:
            tok.encode(ch)
        except KeyError:
            unknown.add(ch)
    return sorted(unknown)


def write(session: Session, prompt: str, show_prompt: bool) -> None:
    """Continue `prompt`, printing every token as soon as the model picks it."""
    unknown = unknown_characters(session.tok, prompt)
    if unknown:
        print(f"  este modelo nunca viu estes caracteres e não sabe escrevê-los: {' '.join(unknown)}")
        return
    ids = session.tok.encode(prompt)
    if not ids:
        return
    s = session.settings
    if s.seed is not None:
        torch.manual_seed(s.seed)

    idx = torch.tensor([ids], device=session.device)  # (1, T)
    written: list[str] = []
    sys.stdout.write("\ncurupira > " + (prompt if show_prompt else "…"))
    try:
        for token in session.model.stream(idx, s.length, s.temperature, s.top_k):
            piece = session.tok.decode([token])
            written.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write(" [interrompido]")
    sys.stdout.write("\n")
    session.last_text = prompt + "".join(written)


def run_command(session: Session, line: str) -> bool:
    """Apply a /command. Returns False when the program should end."""
    name, _, value = line[1:].strip().partition(" ")
    value = value.strip()
    s = session.settings
    try:
        if name == "sair":
            return False
        if name == "ajuda":
            print(HELP)
        elif name == "config":
            describe(session)
        elif name == "mais":
            if session.last_text:
                write(session, session.last_text, show_prompt=False)
            else:
                print("  ainda não há texto para continuar")
        elif name == "temp":
            session.settings = replace(s, temperature=float(value))
        elif name == "topk":
            session.settings = replace(s, top_k=int(value) or None)
        elif name == "tamanho":
            session.settings = replace(s, length=max(1, int(value)))
        elif name == "semente":
            session.settings = replace(s, seed=None if value in ("", "off") else int(value))
        elif name == "modelos":
            for checkpoint in available_checkpoints():
                print(f"  {'*' if checkpoint == session.name else ' '} {checkpoint}")
        elif name == "modelo":
            if value not in available_checkpoints():
                print(f"  {value!r} não está em checkpoints/ (veja /modelos)")
            else:
                load_into(session, value, session.device, s)
                describe(session)
        else:
            print(f"  comando desconhecido: /{name} (veja /ajuda)")
    except ValueError:
        print(f"  valor inválido para /{name}: {value!r}")
    if name in ("temp", "topk", "tamanho", "semente"):
        describe(session)
    return True


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    name = pick_checkpoint(args.checkpoint)
    if name is None or name not in available_checkpoints():
        print(f"nenhum checkpoint encontrado em {CHECKPOINT_DIR}: rode `python -m scripts.phase5` "
              "ou `python -m scripts.phase7b_rope` primeiro")
        sys.exit(1)

    settings = Settings(temperature=args.temperature, top_k=args.top_k, length=args.length, seed=args.seed)
    session = load_into(None, name, device, settings)
    print("\n  curupira-llm: você começa o texto, o modelo continua.  /ajuda para os comandos")
    describe(session)

    while True:
        try:
            line = input("\nvocê > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            continue
        if line.startswith("/"):
            if not run_command(session, line):
                break
        else:
            write(session, line, show_prompt=True)


if __name__ == "__main__":
    main()
