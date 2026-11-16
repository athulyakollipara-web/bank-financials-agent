"""Offline stand-in for the LLM, used by tests and CI.

It does NOT understand questions. For SQL it replays the reference ("gold") SQL
from the benchmark file, so the rest of the system - retrieval, validation,
execution, retries, the API and the MCP server - can be tested end to end with
no AWS account and no cost. Accuracy numbers produced in replay mode only prove
that the plumbing works; real accuracy comes from ``BANK_AGENT_LLM=bedrock``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bank_agent.llm.base import LLMResponse

BENCHMARK = Path(__file__).resolve().parents[2] / "eval" / "benchmark.jsonl"


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


class ReplayLLM:
    def __init__(self, benchmark: Path = BENCHMARK, overrides: dict[str, list[str]] | None = None):
        self.gold: dict[str, str] = {}
        if benchmark.exists():
            for line in benchmark.read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.gold[_norm(item["question"])] = item["gold_sql"]
        # overrides: question -> list of SQL answers returned on successive attempts (for retry tests)
        self.overrides = {_norm(k): list(v) for k, v in (overrides or {}).items()}

    def complete(self, task: str, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        question = _extract(prompt, "QUESTION")
        if task == "plan":
            return LLMResponse(json.dumps({"intent": "analytics", "kpis": [], "search_terms": [question]}), "replay")
        if task == "sql":
            key = _norm(question)
            if key in self.overrides and self.overrides[key]:
                return LLMResponse(self.overrides[key].pop(0), "replay")
            if key not in self.gold:
                raise KeyError("Replay mode only knows benchmark questions. Set BANK_AGENT_LLM=bedrock for free-form questions.")
            return LLMResponse(self.gold[key], "replay")
        if task == "answer":
            rows = _extract(prompt, "RESULT")
            return LLMResponse(f"Here is what the data shows for: {question}\n{rows}", "replay")
        raise ValueError(f"Unknown task {task}")


def _extract(prompt: str, label: str) -> str:
    m = re.search(rf"<{label}>\n?(.*?)\n?</{label}>", prompt, re.S)
    return m.group(1).strip() if m else ""
