"""Semantic layer: governed KPI definitions, schema docs and glossary.

Locally the definitions are read from ``kpis.yaml``. In AWS mode they are served
from a DynamoDB table (seeded from the same file with ``sync_to_dynamodb``), so
business owners can update a definition without a code deploy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

KPI_FILE = Path(__file__).with_name("kpis.yaml")


@dataclass(frozen=True)
class Kpi:
    name: str
    label: str
    description: str
    sql: str
    unit: str = "ratio"
    good_direction: str = "neutral"
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    """A retrievable piece of context for the SQL writer."""

    doc_id: str
    kind: str  # kpi | table | column | glossary
    text: str
    metadata: dict = field(default_factory=dict)


class SemanticCatalog:
    def __init__(self, kpis: list[Kpi], tables: dict, glossary: list[dict]):
        self.kpis = {k.name: k for k in kpis}
        self.tables = tables
        self.glossary = glossary

    # ---------- loading ----------
    @classmethod
    def from_yaml(cls, path: Path = KPI_FILE) -> "SemanticCatalog":
        raw = yaml.safe_load(path.read_text())
        kpis = [Kpi(**{**k, "synonyms": tuple(k.get("synonyms", []))}) for k in raw["kpis"]]
        return cls(kpis, raw["tables"], raw.get("glossary", []))

    @classmethod
    def from_dynamodb(cls, table_name: str, region: str, path: Path = KPI_FILE) -> "SemanticCatalog":
        import boto3

        table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        items = table.scan()["Items"]
        kpis = [
            Kpi(
                name=i["name"], label=i["label"], description=i["description"], sql=i["sql"],
                unit=i.get("unit", "ratio"), good_direction=i.get("good_direction", "neutral"),
                synonyms=tuple(i.get("synonyms", [])),
            )
            for i in items
        ]
        base = cls.from_yaml(path)  # schema docs + glossary ship with the code
        return cls(kpis, base.tables, base.glossary)

    def sync_to_dynamodb(self, table_name: str, region: str) -> int:
        import boto3

        table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        with table.batch_writer() as batch:
            for k in self.kpis.values():
                batch.put_item(Item={
                    "name": k.name, "label": k.label, "description": k.description, "sql": k.sql,
                    "unit": k.unit, "good_direction": k.good_direction, "synonyms": list(k.synonyms),
                })
        return len(self.kpis)

    # ---------- views used by the agent ----------
    def allowed_columns(self) -> dict[str, set[str]]:
        return {t: set(spec["columns"]) for t, spec in self.tables.items()}

    def schema_text(self) -> str:
        lines = []
        for table, spec in self.tables.items():
            lines.append(f"TABLE {table} -- {spec['description']}")
            for col, desc in spec["columns"].items():
                lines.append(f"  {col}: {desc}")
        return "\n".join(lines)

    def documents(self) -> list[Document]:
        docs: list[Document] = []
        for k in self.kpis.values():
            text = (
                f"KPI {k.label} ({k.name}). Also called: {', '.join(k.synonyms)}. "
                f"{k.description} SQL: {k.sql}. Better when {k.good_direction}."
            )
            docs.append(Document(f"kpi:{k.name}", "kpi", text, {"name": k.name, "sql": k.sql}))
        for table, spec in self.tables.items():
            docs.append(Document(f"table:{table}", "table", f"Table {table}: {spec['description']}", {"table": table}))
            for col, desc in spec["columns"].items():
                docs.append(Document(f"column:{table}.{col}", "column", f"Column {table}.{col}: {desc}", {"table": table, "column": col}))
        for g in self.glossary:
            docs.append(Document(f"glossary:{g['term']}", "glossary", f"Glossary '{g['term']}': {g['meaning']}", {}))
        return docs
