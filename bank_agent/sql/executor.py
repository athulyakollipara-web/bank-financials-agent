"""Query execution: Amazon Athena in AWS mode, Spark SQL over the local gold layer otherwise.

The agent always writes Athena (Trino) SQL. Locally, SQLGlot transpiles it to
Spark SQL first, so the same query text is tested in both places.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from bank_agent.sql.validator import transpile


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    engine_sql: str


class SqlError(RuntimeError):
    pass


class Executor(Protocol):
    def run(self, sql: str, max_rows: int = 200) -> QueryResult: ...


def _plain(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


class SparkExecutor:
    def __init__(self, spark, warehouse: Path):
        from bank_agent.pipeline.medallion import register_gold_views

        self.spark = spark
        register_gold_views(spark, warehouse)

    def run(self, sql: str, max_rows: int = 200) -> QueryResult:
        spark_sql = transpile(sql, "spark")
        try:
            df = self.spark.sql(spark_sql)
            rows = [[_plain(v) for v in r] for r in df.limit(max_rows).collect()]
        except Exception as e:  # pyspark raises several exception types
            raise SqlError(str(e).splitlines()[0][:400]) from e
        return QueryResult(df.columns, rows, spark_sql)


class AthenaExecutor:
    def __init__(self, database: str, workgroup: str, output_s3: str | None, region: str, timeout_s: int = 60):
        import boto3

        self.client = boto3.client("athena", region_name=region)
        self.database, self.workgroup, self.output_s3, self.timeout_s = database, workgroup, output_s3, timeout_s

    def run(self, sql: str, max_rows: int = 200) -> QueryResult:
        kwargs = {
            "QueryString": sql,
            "QueryExecutionContext": {"Database": self.database},
            "WorkGroup": self.workgroup,
        }
        if self.output_s3:
            kwargs["ResultConfiguration"] = {"OutputLocation": self.output_s3}
        qid = self.client.start_query_execution(**kwargs)["QueryExecutionId"]
        deadline = time.time() + self.timeout_s
        while True:
            status = self.client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
            if status["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            if time.time() > deadline:
                self.client.stop_query_execution(QueryExecutionId=qid)
                raise SqlError(f"Athena query timed out after {self.timeout_s}s")
            time.sleep(0.5)
        if status["State"] != "SUCCEEDED":
            raise SqlError(status.get("StateChangeReason", status["State"]))
        result = self.client.get_query_results(QueryExecutionId=qid, MaxResults=max_rows + 1)
        rs = result["ResultSet"]
        columns = [c["Name"] for c in rs["ResultSetMetadata"]["ColumnInfo"]]
        data = [[d.get("VarCharValue") for d in row["Data"]] for row in rs["Rows"][1:]]  # first row = header
        return QueryResult(columns, data, sql)
