import pytest

from bank_agent.semantic.catalog import SemanticCatalog
from bank_agent.sql.validator import transpile, validate_sql

ALLOWED = SemanticCatalog.from_yaml().allowed_columns()


def test_valid_query_gets_limit():
    r = validate_sql("SELECT bank_name FROM dim_bank WHERE state = 'MA'", ALLOWED)
    assert r.ok
    assert "LIMIT 1000" in r.sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE dim_bank",
    "DELETE FROM dim_bank",
    "INSERT INTO dim_bank (cert) VALUES (1)",
    "SELECT 1; DROP TABLE dim_bank",
])
def test_rejects_writes_and_multiple_statements(sql):
    assert not validate_sql(sql, ALLOWED).ok


def test_rejects_unknown_table_and_column():
    r = validate_sql("SELECT secret FROM customers", ALLOWED)
    assert not r.ok
    assert any("Unknown table" in e for e in r.errors)
    r = validate_sql("SELECT revenue FROM fact_bank_quarter", ALLOWED)
    assert any("Unknown column 'revenue'" in e for e in r.errors)


def test_allows_aliases_and_ctes():
    sql = (
        "WITH g AS (SELECT cert, LAG(total_deposits) OVER (PARTITION BY cert ORDER BY report_date) AS prev FROM fact_bank_quarter) "
        "SELECT cert, prev AS previous_deposits FROM g ORDER BY previous_deposits DESC LIMIT 5"
    )
    assert validate_sql(sql, ALLOWED).ok


def test_unparseable_sql_returns_message():
    r = validate_sql("SELEC bank_name FROM", ALLOWED)
    assert not r.ok and r.errors


def test_transpile_trino_to_spark():
    out = transpile("SELECT DATE '2025-12-31' AS d, approx_distinct(cert) FROM dim_bank", "spark")
    assert "APPROX_COUNT_DISTINCT" in out.upper()
