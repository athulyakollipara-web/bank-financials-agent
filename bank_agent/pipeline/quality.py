"""Great Expectations checks on the gold layer.

The agent only ever reads gold tables, so this is the contract that protects it:
if a check fails, the pipeline stops before bad numbers reach an answer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")

import great_expectations as gx  # noqa: E402
import pandas as pd


@dataclass
class QualityReport:
    table: str
    success: bool
    failed: list[str]
    evaluated: int


def _suite_for(table: str) -> list:
    E = gx.expectations
    if table == "dim_bank":
        return [
            E.ExpectColumnValuesToNotBeNull(column="cert"),
            E.ExpectColumnValuesToBeUnique(column="cert"),
            E.ExpectColumnValuesToNotBeNull(column="bank_name"),
            E.ExpectColumnValueLengthsToEqual(column="state", value=2),
            E.ExpectColumnValuesToBeInSet(column="asset_size_bucket", value_set=["under_1b", "1b_to_10b", "over_10b"]),
        ]
    if table == "fact_bank_quarter":
        return [
            E.ExpectColumnValuesToNotBeNull(column="cert"),
            E.ExpectColumnValuesToNotBeNull(column="report_date"),
            E.ExpectCompoundColumnsToBeUnique(column_list=["cert", "report_date"]),
            E.ExpectColumnValuesToBeBetween(column="total_assets", min_value=0),
            E.ExpectColumnValuesToBeBetween(column="total_deposits", min_value=0),
            E.ExpectColumnPairValuesAToBeGreaterThanB(column_A="total_assets", column_B="total_deposits", or_equal=True),
            E.ExpectColumnValuesToNotBeNull(column="noncurrent_loans", mostly=0.95),
            E.ExpectColumnValuesToMatchRegex(column="quarter", regex=r"^\d{4}Q[1-4]$"),
        ]
    raise ValueError(f"No quality suite for {table}")


def validate_table(table: str, df: pd.DataFrame) -> QualityReport:
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name=f"{table}_src")
    asset = source.add_dataframe_asset(name=table)
    batch = asset.add_batch_definition_whole_dataframe(f"{table}_batch").get_batch(batch_parameters={"dataframe": df})
    suite = gx.ExpectationSuite(name=f"{table}_suite")
    for expectation in _suite_for(table):
        suite.add_expectation(expectation)
    result = batch.validate(suite)
    failed = [r.expectation_config.type for r in result.results if not r.success]
    return QualityReport(table=table, success=bool(result.success), failed=failed, evaluated=len(result.results))
