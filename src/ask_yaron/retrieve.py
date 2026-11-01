"""Small BM25 retriever. The corpus is a few dozen sections, so no vector database is needed."""

from __future__ import annotations

import math
import re
from collections import Counter

from ask_yaron.knowledge import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have he his how in is it of on or "
    "she that the their this to was what when where which who why with yaron's yaron you".split()
)


# Recruiters' wording rarely matches CV wording; a tiny synonym map closes the common gaps.
SYNONYMS = {
    "study": "education degree bsc college",
    "studied": "education degree bsc college",
    "degree": "education bsc",
    "university": "education college",
    "speak": "languages",
    "german": "languages",
    "job": "experience",
    "jobs": "experience",
    "worked": "experience",
    "work": "experience",
    "employer": "experience",
    "company": "experience",
    "built": "project",
    "side": "project",
    "ai": "agents agent llm",
    "cloud": "azure aws docker devops",
    "frontend": "react vue",
    "backend": "java spring node nestjs express",
    "email": "contact",
    "reach": "contact",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if t not in STOPWORDS]


def expand(query: str) -> str:
    tokens = tokenize(query)
    return " ".join(tokens + [SYNONYMS[t] for t in tokens if t in SYNONYMS])


class Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.avg_len = sum(sum(d.values()) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, doc: Counter) -> float:
        length = sum(doc.values())
        total = 0.0
        for term in tokenize(expand(query)):
            tf = doc.get(term, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query: str, k: int = 8) -> list[Chunk]:
        """Top-k sections for the query, always including the summary for context."""
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: self.score(query, self.docs[i]), reverse=True
        )
        picked = [self.chunks[i] for i in ranked[:k]]
        summary = next((c for c in self.chunks if c.id == "summary"), None)
        if summary and summary not in picked:
            picked = [summary, *picked[: k - 1]]
        return picked
