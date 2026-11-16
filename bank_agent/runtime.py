"""Wires the components together for the configured mode (local/aws, replay/bedrock)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from bank_agent.config import Settings, get_settings
from bank_agent.retrieval.embeddings import HashingEmbedder, TitanEmbedder
from bank_agent.retrieval.hybrid import HybridRetriever
from bank_agent.safety import bedrock_screen, local_screen
from bank_agent.semantic.catalog import SemanticCatalog
from bank_agent.sql.validator import validate_sql


@dataclass
class Runtime:
    settings: Settings
    catalog: SemanticCatalog
    retriever: HybridRetriever
    executor: Any
    llm: Any
    agent: Any

    def ask(self, question: str) -> dict:
        state = self.agent.invoke({"question": question})
        return {
            "question": question,
            "answer": state.get("answer"),
            "sql": state.get("validated_sql") or state.get("sql"),
            "columns": state.get("columns", []),
            "rows": state.get("rows", []),
            "attempts": state.get("attempts", 0),
            "grounding": state.get("grounding"),
            "models": state.get("models", []),
            "trace": state.get("trace", []),
        }

    def run_readonly_sql(self, sql: str, max_rows: int = 200) -> dict:
        check = validate_sql(sql, self.catalog.allowed_columns())
        if not check.ok:
            return {"ok": False, "errors": check.errors}
        res = self.executor.run(check.sql, max_rows=max_rows)
        return {"ok": True, "sql": check.sql, "columns": res.columns, "rows": res.rows}


def build_runtime(settings: Settings | None = None, llm: Any = None) -> Runtime:
    from bank_agent.agent.graph import build_agent

    s = settings or get_settings()
    if s.mode == "aws":
        catalog = SemanticCatalog.from_dynamodb(s.kpi_table, s.aws_region)
        embedder = TitanEmbedder(s.embedding_model, s.aws_region)
        from bank_agent.sql.executor import AthenaExecutor

        executor = AthenaExecutor(s.athena_database, s.athena_workgroup, s.athena_output, s.aws_region)
    else:
        catalog = SemanticCatalog.from_yaml()
        embedder = HashingEmbedder()
        from bank_agent.pipeline.medallion import get_spark
        from bank_agent.sql.executor import SparkExecutor

        executor = SparkExecutor(get_spark(), s.warehouse)

    if s.mode == "aws" and os.getenv("OPENSEARCH_HOST"):
        from bank_agent.retrieval.opensearch_backend import OpenSearchRetriever

        retriever = OpenSearchRetriever.connect(os.environ["OPENSEARCH_HOST"], s.aws_region, "bank-agent-semantic", embedder)
        retriever.ensure_index(catalog.documents())
    else:
        retriever = HybridRetriever(catalog.documents(), embedder)

    if llm is None:
        if s.llm == "bedrock":
            from bank_agent.llm.bedrock import BedrockLLM

            llm = BedrockLLM(s.primary_model, s.fallback_model, s.aws_region)
        else:
            from bank_agent.llm.replay import ReplayLLM

            llm = ReplayLLM()

    if s.guardrail_id and s.llm == "bedrock":
        def screen(text: str):
            return bedrock_screen(text, s.guardrail_id, s.guardrail_version, s.aws_region)
    else:
        screen = local_screen

    agent = build_agent(llm, retriever, catalog, executor, screen, max_attempts=s.max_sql_attempts)
    return Runtime(s, catalog, retriever, executor, llm, agent)


@lru_cache(maxsize=1)
def default_runtime() -> Runtime:
    return build_runtime()
