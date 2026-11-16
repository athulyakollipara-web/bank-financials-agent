from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "by", "with", "what",
    "which", "show", "me", "list", "give", "how", "did", "does", "was", "were", "per", "each", "on", "at",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
