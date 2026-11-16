"""Numeric groundedness: does every figure in the answer come from the query result?

This is the hallucination check that matters most for financial answers. Each
number in the answer is compared with the result values, allowing for the
presentation conversions the answer agent is told to make (fraction -> percent,
$ thousands -> $M / $B) and for rounding.
"""
from __future__ import annotations

import re

NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text):
        s = m.replace("$", "").replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def _candidates(rows: list[list]) -> list[float]:
    vals = []
    for row in rows:
        for v in row:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            vals.extend([x, x * 100, x / 1_000, x / 1_000_000, x / 1_000_000_000])
    return vals


def _matches(n: float, candidates: list[float]) -> bool:
    for c in candidates:
        if c == 0:
            if n == 0:
                return True
            continue
        decimals = len(str(n).split(".")[1]) if "." in str(n) else 0
        if abs(n - c) <= max(0.5 * 10 ** -decimals, abs(c) * 0.005):
            return True
    return False


def numeric_groundedness(answer: str, rows: list[list]) -> dict:
    nums = [n for n in _numbers(answer) if not (abs(n) <= 10 and float(n).is_integer()) and not (2000 <= n <= 2100)]
    if not nums:
        return {"checked": 0, "grounded": 0, "score": 1.0, "ungrounded": []}
    cands = _candidates(rows)
    ungrounded = [n for n in nums if not _matches(n, cands)]
    grounded = len(nums) - len(ungrounded)
    return {"checked": len(nums), "grounded": grounded, "score": grounded / len(nums), "ungrounded": ungrounded}
