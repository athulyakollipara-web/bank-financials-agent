"""Command line entry point.

    python -m bank_agent.cli build              # generate sample data + run the medallion pipeline + quality checks
    python -m bank_agent.cli ask "question"     # ask one question, print answer + SQL + trace
"""
from __future__ import annotations

import argparse
import json
import sys

from bank_agent.config import get_settings


def build(use_existing_raw: bool) -> int:
    from bank_agent.data.generate import generate
    from bank_agent.pipeline.medallion import GOLD_TABLES, get_spark, read_gold, run_all
    from bank_agent.pipeline.quality import validate_table

    s = get_settings()
    if not use_existing_raw:
        print("raw:", generate(s.raw_dir))
    spark = get_spark()
    print("pipeline:", run_all(spark, s.raw_dir, s.warehouse))
    ok = True
    for table in GOLD_TABLES:
        report = validate_table(table, read_gold(spark, s.warehouse, table).toPandas())
        print(f"quality {table}: {'PASS' if report.success else 'FAIL'} ({report.evaluated} checks) {report.failed or ''}")
        ok &= report.success
    return 0 if ok else 1


def ask(question: str, show_trace: bool) -> int:
    from bank_agent.runtime import build_runtime

    result = build_runtime().ask(question)
    print("\nANSWER\n" + str(result["answer"]))
    print("\nSQL\n" + str(result["sql"]))
    print(f"\nROWS ({len(result['rows'])})")
    for row in result["rows"][:10]:
        print("  ", row)
    print(f"\nattempts={result['attempts']} models={result['models']} grounding={result['grounding']}")
    if show_trace:
        print("\nTRACE\n" + json.dumps(result["trace"], indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bank-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--use-existing-raw", action="store_true", help="skip sample generation (e.g. after fetch_fdic)")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--trace", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "build":
        return build(args.use_existing_raw)
    return ask(args.question, args.trace)


if __name__ == "__main__":
    sys.exit(main())
