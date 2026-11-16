"""Hybrid retrieval: BM25 + vector k-NN, fused with Reciprocal Rank Fusion, then reranked.

Why hybrid: banking questions mix exact jargon ("NIM", "Tier 1") that BM25 nails
with paraphrases ("how profitable", "bad loans") that need vector similarity.
RRF combines the two rankings without having to calibrate their score scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bank_agent.retrieval.bm25 import BM25
from bank_agent.retrieval.embeddings import Embedder, cosine
from bank_agent.retrieval.text import tokenize
from bank_agent.semantic.catalog import Document


@dataclass
class Hit:
    doc: Document
    score: float
    sources: tuple[str, ...]


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class LexicalReranker:
    """Second-stage reranker used locally.

    Boosts documents whose KPI synonyms or column names appear verbatim in the
    question, and prefers KPI/glossary docs over raw column docs on ties. A
    hosted cross-encoder can replace it through the same ``Reranker`` interface.
    """

    KIND_PRIOR = {"kpi": 0.15, "glossary": 0.1, "table": 0.05, "column": 0.0}

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        q = query.lower()
        q_tokens = set(tokenize(query))
        rescored = []
        for h in hits:
            boost = self.KIND_PRIOR.get(h.doc.kind, 0.0)
            text = h.doc.text.lower()
            if h.doc.kind == "kpi":
                phrases = text.split("also called:")[1].split(".")[0].split(",") if "also called:" in text else []
                if any(p.strip() and p.strip() in q for p in phrases):
                    boost += 0.6
            col = h.doc.metadata.get("column")
            if col and col.replace("_", " ") in q:
                boost += 0.3
            overlap = len(q_tokens & set(tokenize(text))) / max(1, len(q_tokens))
            rescored.append(Hit(h.doc, h.score + boost + 0.2 * overlap, h.sources))
        return sorted(rescored, key=lambda h: h.score, reverse=True)


class HybridRetriever:
    def __init__(self, docs: list[Document], embedder: Embedder, reranker: Reranker | None = None, rrf_k: int = 60):
        self.docs = docs
        self.embedder = embedder
        self.reranker = reranker or LexicalReranker()
        self.rrf_k = rrf_k
        self.bm25 = BM25([d.text for d in docs])
        self.vectors = embedder.embed([d.text for d in docs])

    def _rank(self, scores: list[float], top_n: int) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:top_n] if scores[i] > 0]

    def search(self, query: str, k: int = 8, candidates: int = 20) -> list[Hit]:
        lexical = self._rank(self.bm25.scores(query), candidates)
        qv = self.embedder.embed([query])[0]
        semantic = self._rank([cosine(qv, v) for v in self.vectors], candidates)

        fused: dict[int, float] = {}
        sources: dict[int, set[str]] = {}
        for name, ranking in (("bm25", lexical), ("knn", semantic)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                sources.setdefault(idx, set()).add(name)
        # Scale RRF scores into roughly 0..1 so reranker boosts are comparable.
        top = max(fused.values(), default=1.0)
        hits = [Hit(self.docs[i], s / top, tuple(sorted(sources[i]))) for i, s in fused.items()]
        return self.reranker.rerank(query, hits)[:k]
