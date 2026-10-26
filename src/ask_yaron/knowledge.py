"""Load the markdown knowledge base into citable chunks, one per `##` section."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "knowledge"


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_markdown(markdown: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    title: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if title and body:
            chunks.append(Chunk(id=slugify(title), title=title, text=body, source=source))

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            title, lines = line[3:].strip(), []
        elif title is not None:
            lines.append(line)
    flush()
    return chunks


def load_chunks(directory: Path = DEFAULT_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(parse_markdown(path.read_text(encoding="utf-8"), path.name))
    ids = [c.id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate section titles: {sorted(duplicates)}")
    return chunks
