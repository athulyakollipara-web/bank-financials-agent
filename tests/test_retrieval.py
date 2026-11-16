import sys
from pathlib import Path

from bank_agent.retrieval.bm25 import BM25
from bank_agent.retrieval.embeddings import HashingEmbedder
from bank_agent.retrieval.hybrid import HybridRetriever
from bank_agent.semantic.catalog import SemanticCatalog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))


def _retriever():
    return HybridRetriever(SemanticCatalog.from_yaml().documents(), HashingEmbedder())


def test_bm25_prefers_matching_document():
    bm = BM25(["net interest margin definition", "tier 1 capital ratio", "deposit growth"])
    scores = bm.scores("interest margin")
    assert scores.index(max(scores)) == 0


def test_synonym_maps_to_governed_kpi():
    top = _retriever().search("Which banks have the worst cost to income ratio?", k=3)
    assert top[0].doc.doc_id == "kpi:efficiency_ratio"


def test_hybrid_uses_both_retrievers():
    hits = _retriever().search("tier 1 capital adequacy", k=5)
    assert any(set(h.sources) == {"bm25", "knn"} for h in hits)


def test_hybrid_recall_on_held_out_paraphrases():
    from retrieval_eval import evaluate

    report = evaluate("paraphrase")
    assert report["hybrid"]["recall@3"] >= 0.9
    assert report["hybrid"]["recall@1"] >= max(report["bm25"]["recall@1"], report["knn"]["recall@1"])
