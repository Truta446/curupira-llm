"""Download Machado de Assis novels from Project Gutenberg, clean them and
split into train/validation.

Output:
    data/raw/pg<id>.txt   original texts (cached, not downloaded again)
    data/train.txt        first 90% of each book
    data/val.txt          last 10% of each book

Usage:
    .venv/bin/python prepare_data.py
"""

import os
import re
import urllib.request
from typing import Final

# Gutenberg id -> title
BOOKS: Final[dict[int, str]] = {
    53101: "A Mão e a Luva",
    67162: "Helena",
    67780: "Iaiá Garcia",
    54829: "Memórias Póstumas de Brás Cubas",
    55682: "Quincas Borba",
    55752: "Dom Casmurro",
    56737: "Esaú e Jacó",
    55797: "Memorial de Aires",
}
URL: Final = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
DATA_DIR: Final = "data"
RAW_DIR: Final = os.path.join(DATA_DIR, "raw")
VAL_FRACTION: Final = 0.1


def download(book_id: int) -> str:
    path = os.path.join(RAW_DIR, f"pg{book_id}.txt")
    if not os.path.exists(path):
        print(f"  downloading {URL.format(id=book_id)}")
        req = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": "curupira-llm"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path, encoding="utf-8") as f:
        return f.read()


def clean(text: str) -> list[str]:
    """Return the book as a list of paragraphs, one string per paragraph."""
    text = text.replace("\r\n", "\n")  # Gutenberg files use CRLF line endings

    # Keep only the body, between Gutenberg's license start/end markers.
    start = text.index("\n", text.index("*** START OF")) + 1
    end = text.index("*** END OF")
    text = text[start:end]

    text = text.replace("_", "")  # _italic_ is markup, not language

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):  # paragraphs are separated by a blank line
        # Inside a paragraph the line breaks every ~70 columns are just layout:
        # join them so the model doesn't waste capacity predicting fake "\n".
        p = " ".join(line.strip() for line in block.split("\n"))
        p = re.sub(r" {2,}", " ", p).strip()
        if not p or set(p) <= {"*", " "}:  # empty or a "*   *   *" separator
            continue
        paragraphs.append(p)
    return paragraphs


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    train_parts: list[str] = []
    val_parts: list[str] = []

    for book_id, title in BOOKS.items():
        paragraphs = clean(download(book_id))
        total = sum(len(p) + 1 for p in paragraphs)

        # Validation = the final stretch of EACH book (not the whole last book),
        # so train and val share the same style and vocabulary. The cut falls
        # on a paragraph boundary and each side is contiguous, so near-identical
        # sentences don't leak across the split.
        running, cut = 0, len(paragraphs)
        for i, p in enumerate(paragraphs):
            running += len(p) + 1
            if running >= (1 - VAL_FRACTION) * total:
                cut = i + 1
                break
        train_parts.append("\n".join(paragraphs[:cut]))
        val_parts.append("\n".join(paragraphs[cut:]))
        print(f"  {title:<34} {total:>8,} chars  {len(paragraphs):>5} paragraphs")

    # Books separated by one blank line.
    train = "\n\n".join(train_parts) + "\n"
    val = "\n\n".join(val_parts) + "\n"
    with open(os.path.join(DATA_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.write(train)
    with open(os.path.join(DATA_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.write(val)
    print(f"\ntrain.txt: {len(train):,} chars | val.txt: {len(val):,} chars")


if __name__ == "__main__":
    main()
