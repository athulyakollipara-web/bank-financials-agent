"""Runtime settings, read from environment variables.

Two switches control where things run:

* ``BANK_AGENT_MODE``  - ``local`` (PySpark SQL over Parquet) or ``aws`` (Athena + DynamoDB + OpenSearch)
* ``BANK_AGENT_LLM``   - ``replay`` (offline, deterministic; used by tests and CI) or ``bedrock``
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    mode: str
    llm: str
    warehouse: Path
    raw_dir: Path
    aws_region: str
    primary_model: str
    fallback_model: str
    embedding_model: str
    guardrail_id: str | None
    guardrail_version: str
    athena_database: str
    athena_workgroup: str
    athena_output: str | None
    kpi_table: str
    max_sql_attempts: int


def get_settings() -> Settings:
    warehouse = Path(os.getenv("BANK_AGENT_WAREHOUSE", ROOT / "warehouse"))
    return Settings(
        mode=os.getenv("BANK_AGENT_MODE", "local"),
        llm=os.getenv("BANK_AGENT_LLM", "replay"),
        warehouse=warehouse,
        raw_dir=Path(os.getenv("BANK_AGENT_RAW_DIR", warehouse / "raw")),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        # Override with the newest model IDs enabled in your Bedrock account/region.
        primary_model=os.getenv("BEDROCK_PRIMARY_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        fallback_model=os.getenv("BEDROCK_FALLBACK_MODEL", "us.amazon.nova-pro-v1:0"),
        embedding_model=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
        guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID"),
        guardrail_version=os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
        athena_database=os.getenv("ATHENA_DATABASE", "bank_gold"),
        athena_workgroup=os.getenv("ATHENA_WORKGROUP", "bank-agent"),
        athena_output=os.getenv("ATHENA_OUTPUT_S3"),
        kpi_table=os.getenv("KPI_TABLE", "bank-agent-kpis"),
        max_sql_attempts=int(os.getenv("BANK_AGENT_MAX_SQL_ATTEMPTS", "3")),
    )
