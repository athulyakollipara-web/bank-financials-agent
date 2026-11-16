"""Run the benchmark and log results to MLflow.

    python eval/run_eval.py                          # replay LLM: plumbing check, should be 100%
    BANK_AGENT_LLM=bedrock python eval/run_eval.py   # real accuracy with Bedrock models
    python eval/run_eval.py --min-accuracy 0.85      # exit non-zero below threshold (CI gate)

Writes ``eval/results/<run>.json`` and a Markdown summary, and logs metrics,
params and the per-question table to MLflow (``./mlruns`` by default).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlflow

from bank_agent.config import get_settings
from bank_agent.evaluation import execution_match
from bank_agent.runtime import build_runtime

HERE = Path(__file__).parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-accuracy", type=float, default=0.0)
    p.add_argument("--category", default="")
    args = p.parse_args()

    items = [json.loads(line) for line in (HERE / "benchmark.jsonl").read_text().splitlines() if line.strip()]
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.limit:
        items = items[: args.limit]

    settings = get_settings()
    rt = build_runtime(settings)
    results = []
    by_cat: dict[str, list[bool]] = defaultdict(list)
    started = time.time()
    for item in items:
        t = time.perf_counter()
        gold = rt.run_readonly_sql(item["gold_sql"], max_rows=1000)
        try:
            pred = rt.ask(item["question"])
            err = None
        except Exception as e:  # keep going; count as a miss
            pred, err = {"rows": [], "sql": None, "attempts": 0, "grounding": None, "models": []}, str(e)
        ordered = "ORDER BY" in item["gold_sql"].upper() and "LIMIT" in item["gold_sql"].upper()
        match = gold["ok"] and execution_match(gold["rows"], pred["rows"], ordered=ordered)
        by_cat[item["category"]].append(match)
        results.append({
            "id": item["id"], "category": item["category"], "question": item["question"], "match": match,
            "attempts": pred["attempts"], "pred_sql": pred["sql"], "gold_sql": item["gold_sql"],
            "groundedness": (pred.get("grounding") or {}).get("score"), "models": pred["models"],
            "latency_s": round(time.perf_counter() - t, 2), "error": err,
        })
        print(f"{item['id']} {'PASS' if match else 'FAIL'} [{item['category']}] {item['question'][:70]}")

    n = len(results)
    accuracy = sum(r["match"] for r in results) / max(1, n)
    grounded = [r["groundedness"] for r in results if r["groundedness"] is not None]
    first_try = sum(1 for r in results if r["match"] and r["attempts"] == 1) / max(1, n)
    summary = {
        "llm": settings.llm, "mode": settings.mode, "questions": n,
        "execution_accuracy": round(accuracy, 4),
        "first_attempt_accuracy": round(first_try, 4),
        "mean_numeric_groundedness": round(sum(grounded) / len(grounded), 4) if grounded else None,
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / max(1, n), 2),
        "by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())},
        "wall_time_s": round(time.time() - started, 1),
    }

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{settings.llm}-{stamp}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    md = [f"# Benchmark run ({settings.llm}, {stamp})", "", f"Execution accuracy: **{accuracy:.1%}** on {n} questions",
          f"First-attempt accuracy: {first_try:.1%}", "", "| Category | Accuracy |", "|---|---|"]
    md += [f"| {c} | {a:.1%} |" for c, a in summary["by_category"].items()]
    (out_dir / f"{settings.llm}-{stamp}.md").write_text("\n".join(md) + "\n")

    mlflow.set_experiment("bank-financials-nl2sql")
    with mlflow.start_run(run_name=f"{settings.llm}-{stamp}"):
        mlflow.log_params({"llm": settings.llm, "mode": settings.mode, "questions": n,
                           "primary_model": settings.primary_model, "fallback_model": settings.fallback_model})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None})
        mlflow.log_metrics({f"acc_{c}": a for c, a in summary["by_category"].items()})
        mlflow.log_table({k: [r[k] for r in results] for k in ("id", "category", "question", "match", "attempts", "latency_s")},
                         artifact_file="per_question.json")

    print(json.dumps(summary, indent=2))
    if accuracy < args.min_accuracy:
        print(f"FAILED: accuracy {accuracy:.1%} below threshold {args.min_accuracy:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
