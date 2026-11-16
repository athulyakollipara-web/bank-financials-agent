"""Amazon OpenSearch backend for AWS mode.

The index stores each semantic-layer document with both its text (for BM25) and
its Titan embedding (for k-NN). ``hybrid_query`` issues both searches; fusion and
reranking reuse the same RRF + reranker code as local mode, so behaviour matches.

Requires the optional ``opensearch-py`` dependency: ``pip install -e .[aws]``.
"""
from __future__ import annotations

from bank_agent.retrieval.embeddings import Embedder
from bank_agent.retrieval.hybrid import Hit, LexicalReranker
from bank_agent.semantic.catalog import Document


def index_body(dim: int) -> dict:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "kind": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "english"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "engine": "faiss", "space_type": "innerproduct"},
                },
            }
        },
    }


def bm25_query(text: str, size: int) -> dict:
    return {"size": size, "query": {"match": {"text": {"query": text}}}}


def knn_query(vector: list[float], size: int) -> dict:
    return {"size": size, "query": {"knn": {"embedding": {"vector": vector, "k": size}}}}


class OpenSearchRetriever:
    def __init__(self, client, index: str, embedder: Embedder, rrf_k: int = 60):
        self.client, self.index, self.embedder, self.rrf_k = client, index, embedder, rrf_k
        self.reranker = LexicalReranker()

    @classmethod
    def connect(cls, host: str, region: str, index: str, embedder: Embedder) -> "OpenSearchRetriever":
        import boto3
        from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

        auth = AWSV4SignerAuth(boto3.Session().get_credentials(), region, "es")
        client = OpenSearch(hosts=[{"host": host, "port": 443}], http_auth=auth, use_ssl=True,
                            verify_certs=True, connection_class=RequestsHttpConnection)
        return cls(client, index, embedder)

    def ensure_index(self, docs: list[Document]) -> None:
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, body=index_body(self.embedder.dim))
        vectors = self.embedder.embed([d.text for d in docs])
        for d, v in zip(docs, vectors):
            self.client.index(index=self.index, id=d.doc_id, body={
                "doc_id": d.doc_id, "kind": d.kind, "text": d.text, "embedding": v, "metadata": d.metadata,
            })

    def search(self, query: str, k: int = 8, candidates: int = 20) -> list[Hit]:
        lexical = self.client.search(index=self.index, body=bm25_query(query, candidates))["hits"]["hits"]
        vector = self.embedder.embed([query])[0]
        semantic = self.client.search(index=self.index, body=knn_query(vector, candidates))["hits"]["hits"]
        fused: dict[str, float] = {}
        docs: dict[str, Document] = {}
        sources: dict[str, set[str]] = {}
        for name, ranking in (("bm25", lexical), ("knn", semantic)):
            for rank, hit in enumerate(ranking):
                src = hit["_source"]
                docs[src["doc_id"]] = Document(src["doc_id"], src["kind"], src["text"], src.get("metadata", {}))
                fused[src["doc_id"]] = fused.get(src["doc_id"], 0.0) + 1.0 / (self.rrf_k + rank + 1)
                sources.setdefault(src["doc_id"], set()).add(name)
        top = max(fused.values(), default=1.0)
        hits = [Hit(docs[i], s / top, tuple(sorted(sources[i]))) for i, s in fused.items()]
        return self.reranker.rerank(query, hits)[:k]
