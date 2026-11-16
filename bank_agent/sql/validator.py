"""Structural validation of model-written SQL with SQLGlot, before anything executes.

Checks, in order:
1. Exactly one statement, and it parses in the Athena (Trino) dialect.
2. Read-only: the statement is a SELECT/UNION/WITH and contains no DML/DDL.
3. Every table is a governed gold table (or a CTE defined in the query).
4. Every column exists on a referenced table, or is an alias defined in the query.
5. A LIMIT is added if missing, so a bad query can't pull an entire table.

Failures come back as plain-English messages that are fed to the SQL writer
on the retry, which is how most first-attempt mistakes get fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

SOURCE_DIALECT = "trino"  # Athena engine v3 is Trino-based
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge, exp.Command, exp.TruncateTable)
DEFAULT_LIMIT = 1000


@dataclass
class ValidationResult:
    ok: bool
    sql: str = ""
    errors: list[str] = field(default_factory=list)
    tables: set[str] = field(default_factory=set)


def validate_sql(sql: str, allowed: dict[str, set[str]], dialect: str = SOURCE_DIALECT) -> ValidationResult:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return ValidationResult(False, errors=["Empty SQL."])
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as e:
        return ValidationResult(False, errors=[f"SQL does not parse: {str(e).splitlines()[0]}"])
    if len(statements) != 1:
        return ValidationResult(False, errors=["Exactly one SQL statement is allowed."])
    tree = statements[0]

    errors: list[str] = []
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        errors.append(f"Only SELECT queries are allowed (got {tree.key.upper()}).")
    for node in tree.find_all(*FORBIDDEN):
        errors.append(f"Forbidden operation: {node.key.upper()}.")
    if errors:
        return ValidationResult(False, errors=errors)

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    unknown = sorted(t for t in tables - cte_names if t not in allowed)
    if unknown:
        errors.append(f"Unknown table(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}.")

    real_tables = (tables - cte_names) & set(allowed)
    known_columns = set().union(*(allowed[t] for t in real_tables)) if real_tables else set()
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    cte_columns = {c.name.lower() for cte in tree.find_all(exp.CTE) for c in (cte.args.get("alias").columns if cte.args.get("alias") else [])}
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if not name or name == "*":
            continue
        if name not in known_columns and name not in aliases and name not in cte_columns:
            errors.append(f"Unknown column '{col.name}'. Use only columns from the schema.")
    if errors:
        return ValidationResult(False, errors=sorted(set(errors)), tables=tables)

    if isinstance(tree, exp.Select) and not tree.args.get("limit"):
        tree = tree.limit(DEFAULT_LIMIT)
    return ValidationResult(True, sql=tree.sql(dialect=dialect), tables=real_tables)


def transpile(sql: str, target: str, source: str = SOURCE_DIALECT) -> str:
    """Athena SQL -> another engine's dialect (Spark SQL for local mode)."""
    return sqlglot.transpile(sql, read=source, write=target)[0]
