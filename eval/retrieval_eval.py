"""Retrieval quality: for every benchmark question that needs a governed KPI, is the
right KPI definition in the top-k retrieved documents?

Compares BM25-only, vector-only and hybrid (RRF + rerank) so the value of each
stage is measured, not assumed. Runs offline (hashing embedder) or with Titan
embeddings when ``BANK_AGENT_MODE=aws``.

    python eval/retrieval_eval.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bank_agent.retrieval.embeddings import HashingEmbedder, cosine
from bank_agent.retrieval.hybrid import HybridRetriever
from bank_agent.semantic.catalog import SemanticCatalog

HERE = Path(__file__).parent


def expected_kpi(gold_sql: str, kpi_sqls: dict[str, str]) -> str | None:
    norm = re.sub(r"\s+", " ", gold_sql)
    for name, expr in kpi_sqls.items():
        if expr in norm:
            return name
    return None


def load_cases(suite: str, catalog: SemanticCatalog) -> list[tuple[str, str]]:
    """``benchmark``: KPI questions from the NL2SQL benchmark (wording uses catalog terms).
    ``paraphrase``: held-out questions that avoid the KPI names and synonyms entirely."""
    if suite == "paraphrase":
        rows = [json.loads(l) for l in (HERE / "retrieval_paraphrases.jsonl").read_text().splitlines() if l.strip()]
        return [(r["question"], r["kpi"]) for r in rows]
    kpi_sqls = {k.name: k.sql for k in catalog.kpis.values()}
    items = [json.loads(l) for l in (HERE / "benchmark.jsonl").read_text().splitlines() if l.strip()]
    cases = [(i["question"], expected_kpi(i["gold_sql"], kpi_sqls)) for i in items]
    return [(q, k) for q, k in cases if k]


def evaluate(suite: str = "benchmark", ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    catalog = SemanticCatalog.from_yaml()
    docs = catalog.documents()
    retriever = HybridRetriever(docs, HashingEmbedder())
    cases = load_cases(suite, catalog)

    def kpi_ranking_hybrid(q: str) -> list[str]:
        return [h.doc.metadata["name"] for h in retriever.search(q, k=20) if h.doc.kind == "kpi"]

    def kpi_ranking_bm25(q: str) -> list[str]:
        scores = retriever.bm25.scores(q)
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i].metadata["name"] for i in order if docs[i].kind == "kpi" and scores[i] > 0]

    def kpi_ranking_knn(q: str) -> list[str]:
        qv = retriever.embedder.embed([q])[0]
        sims = [cosine(qv, v) for v in retriever.vectors]
        order = sorted(range(len(docs)), key=lambda i: sims[i], reverse=True)
        return [docs[i].metadata["name"] for i in order if docs[i].kind == "kpi"]

    report = {"suite": suite, "questions": len(cases)}
    for label, fn in (("bm25", kpi_ranking_bm25), ("knn", kpi_ranking_knn), ("hybrid", kpi_ranking_hybrid)):
        rankings = [(fn(q), k) for q, k in cases]
        report[label] = {f"recall@{k}": round(sum(exp in r[:k] for r, exp in rankings) / len(rankings), 4) for k in ks}
        misses = [q for (r, exp), (q, _) in zip(rankings, cases) if exp not in r[:3]]
        report[label]["misses@3"] = misses[:5]
    return report


if __name__ == "__main__":
    for suite in ("benchmark", "paraphrase"):
        print(json.dumps(evaluate(suite), indent=2))
