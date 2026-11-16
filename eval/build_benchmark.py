"""Build the NL2SQL benchmark: ~150 questions with reference (gold) Athena SQL.

Questions are generated from 15 templates covering lookups, governed KPIs,
rankings, aggregations, window functions (QoQ / YoY), thresholds and counts,
with parameters (banks, states, quarters) drawn from the gold layer and wording
varied with KPI synonyms. Run after ``python -m bank_agent.cli build``::

    python eval/build_benchmark.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bank_agent.config import get_settings
from bank_agent.pipeline.medallion import get_spark, read_gold
from bank_agent.semantic.catalog import SemanticCatalog

OUT = Path(__file__).with_name("benchmark.jsonl")
J = "FROM fact_bank_quarter f JOIN dim_bank b ON f.cert = b.cert"
BUCKETS = {"under_1b": "community", "1b_to_10b": "regional", "over_10b": "large"}
LATEST = "(SELECT MAX(report_date) FROM fact_bank_quarter)"


def build(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    s = get_settings()
    spark = get_spark()
    banks = [r.asDict() for r in read_gold(spark, s.warehouse, "dim_bank").orderBy("cert").collect()]
    quarters = sorted({r.quarter for r in read_gold(spark, s.warehouse, "fact_bank_quarter").select("quarter").collect()})
    states = sorted({b["state"] for b in banks})
    kpis = SemanticCatalog.from_yaml().kpis
    kpi_names = list(kpis)

    def kpi_word(name: str) -> str:
        k = kpis[name]
        return rng.choice([k.label.split(" (")[0].lower(), *[x for x in k.synonyms if len(x) > 2][:2]])

    items: list[dict] = []

    def add(category: str, question: str, sql: str) -> None:
        items.append({"id": f"q{len(items) + 1:03d}", "category": category, "question": question, "gold_sql": " ".join(sql.split())})

    for _ in range(10):
        b, q = rng.choice(banks), rng.choice(quarters)
        col, word = rng.choice([("total_deposits", "total deposits"), ("total_assets", "total assets"), ("net_income", "net income"), ("total_loans", "total loans")])
        add("lookup", f"What were the {word} of {b['bank_name']} in {q}?",
            f"SELECT b.bank_name, f.quarter, f.{col} {J} WHERE b.bank_name = '{b['bank_name']}' AND f.quarter = '{q}'")

    for _ in range(32):
        b, q, k = rng.choice(banks), rng.choice(quarters), rng.choice(kpi_names)
        add("kpi_lookup", f"What was the {kpi_word(k)} of {b['bank_name']} in {q}?",
            f"SELECT b.bank_name, f.quarter, ROUND({kpis[k].sql}, 4) AS {k} {J} WHERE b.bank_name = '{b['bank_name']}' AND f.quarter = '{q}'")

    for _ in range(10):
        st, k, n = rng.choice(states), rng.choice(["net_interest_margin", "return_on_assets", "tier1_capital_ratio", "return_on_equity"]), rng.choice([3, 5])
        add("ranking", f"Which {n} banks in {st} had the highest {kpi_word(k)} in the latest quarter?",
            f"SELECT b.bank_name, ROUND({kpis[k].sql}, 4) AS {k} {J} WHERE b.state = '{st}' AND f.report_date = {LATEST} "
            f"ORDER BY {k} DESC, b.cert LIMIT {n}")

    for _ in range(10):
        q, n = rng.choice(quarters), rng.choice([5, 10])
        k = rng.choice(["efficiency_ratio", "npl_ratio"])
        add("ranking", f"Which {n} banks had the lowest {kpi_word(k)} in {q}?",
            f"SELECT b.bank_name, ROUND({kpis[k].sql}, 4) AS {k} {J} WHERE f.quarter = '{q}' ORDER BY {k} ASC, b.cert LIMIT {n}")

    for _ in range(10):
        st, q = rng.choice(states), rng.choice(quarters)
        col, word = rng.choice([("total_deposits", "total deposits"), ("total_loans", "total loans"), ("net_income", "total net income")])
        add("aggregate", f"What were {word} across all banks in {st} in {q}?",
            f"SELECT SUM(f.{col}) AS {col} {J} WHERE b.state = '{st}' AND f.quarter = '{q}'")

    for _ in range(10):
        st, bucket = rng.choice(states), rng.choice(list(BUCKETS))
        add("count", f"How many {BUCKETS[bucket]} banks are headquartered in {st}?",
            f"SELECT COUNT(*) AS bank_count FROM dim_bank b WHERE b.state = '{st}' AND b.asset_size_bucket = '{bucket}'")

    for _ in range(10):
        bucket, q, k = rng.choice(list(BUCKETS)), rng.choice(quarters), rng.choice(kpi_names)
        add("aggregate", f"What was the average {kpi_word(k)} for {BUCKETS[bucket]} banks in {q}?",
            f"SELECT ROUND(AVG({kpis[k].sql}), 4) AS avg_{k} {J} WHERE b.asset_size_bucket = '{bucket}' AND f.quarter = '{q}'")

    for _ in range(10):
        q, n = rng.choice(quarters[1:]), rng.choice([3, 5])
        add("window", f"Which {n} banks had the largest quarter-over-quarter deposit growth in {q}?",
            "WITH g AS (SELECT f.cert, f.quarter, f.total_deposits, LAG(f.total_deposits) OVER (PARTITION BY f.cert ORDER BY f.report_date) AS prev_deposits FROM fact_bank_quarter f) "
            f"SELECT b.bank_name, ROUND((g.total_deposits - g.prev_deposits) / g.prev_deposits, 4) AS deposit_growth FROM g JOIN dim_bank b ON g.cert = b.cert "
            f"WHERE g.quarter = '{q}' ORDER BY deposit_growth DESC, b.cert LIMIT {n}")

    for _ in range(10):
        b, year = rng.choice(banks), rng.choice(["2024", "2025"])
        col, word = rng.choice([("total_assets", "total assets"), ("total_deposits", "deposits"), ("net_income", "net income")])
        add("trend", f"Show the quarterly {word} of {b['bank_name']} in {year}.",
            f"SELECT f.quarter, f.{col} {J} WHERE b.bank_name = '{b['bank_name']}' AND f.quarter LIKE '{year}Q%' ORDER BY f.quarter")

    for _ in range(10):
        st, q, x = rng.choice(states), rng.choice(quarters), rng.choice([9, 10, 11])
        add("threshold", f"List banks in {st} with a tier 1 capital ratio below {x}% in {q}.",
            f"SELECT b.bank_name, ROUND({kpis['tier1_capital_ratio'].sql}, 4) AS tier1_capital_ratio {J} "
            f"WHERE b.state = '{st}' AND f.quarter = '{q}' AND {kpis['tier1_capital_ratio'].sql} < {x / 100}")

    for _ in range(10):
        q = rng.choice(quarters)
        col, word = rng.choice([("total_assets", "total assets"), ("total_deposits", "total deposits")])
        add("aggregate", f"Which state had the highest combined {word} in {q}?",
            f"SELECT b.state, SUM(f.{col}) AS {col} {J} WHERE f.quarter = '{q}' GROUP BY b.state ORDER BY {col} DESC LIMIT 1")

    for _ in range(10):
        b = rng.choice(banks)
        add("window", f"How did {b['bank_name']}'s net income in 2025Q4 compare with 2024Q4?",
            f"SELECT f.quarter, f.net_income {J} WHERE b.bank_name = '{b['bank_name']}' AND f.quarter IN ('2024Q4', '2025Q4') ORDER BY f.quarter")

    for _ in range(10):
        q, x = rng.choice(quarters), rng.choice([1, 1.5, 2])
        add("count", f"How many banks had a noncurrent loan ratio above {x}% in {q}?",
            f"SELECT COUNT(*) AS bank_count FROM fact_bank_quarter f WHERE f.quarter = '{q}' AND {kpis['npl_ratio'].sql} > {x / 100}")

    for _ in range(10):
        st = rng.choice(states)
        add("ranking", f"What is the largest bank by total assets in {st} as of the latest quarter?",
            f"SELECT b.bank_name, f.total_assets {J} WHERE b.state = '{st}' AND f.report_date = {LATEST} ORDER BY f.total_assets DESC LIMIT 1")

    # De-duplicate identical questions and drop any whose answer is empty in this dataset.
    from bank_agent.pipeline.medallion import register_gold_views
    from bank_agent.sql.validator import transpile

    register_gold_views(spark, s.warehouse)
    seen, unique = set(), []
    for it in items:
        if it["question"] not in seen and spark.sql(transpile(it["gold_sql"], "spark")).limit(1).count() > 0:
            seen.add(it["question"])
            unique.append(it)
    for i, it in enumerate(unique, 1):
        it["id"] = f"q{i:03d}"
    return unique


if __name__ == "__main__":
    items = build()
    OUT.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    print(f"wrote {len(items)} questions to {OUT}")
