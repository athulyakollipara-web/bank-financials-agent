"""Execution-match scoring for NL2SQL.

Two result sets match when they have the same number of rows and every gold row
can be paired with a distinct predicted row that contains all of the gold row's
values. This is deliberately tolerant of things that don't change the answer:

* extra columns in the prediction (e.g. the model also returns ``state``),
* column order and (unless the question is about ranking order) row order,
* float rounding (relative tolerance 0.2%), and ratios expressed as percentages.
"""
from __future__ import annotations


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def values_equal(a, b) -> bool:
    na, nb = _num(a), _num(b)
    if na is None or nb is None:
        return str(a).strip().lower() == str(b).strip().lower()
    for scale in (1.0, 100.0, 0.01):
        x = nb * scale
        if abs(na - x) <= max(abs(na) * 0.002, 6e-5):
            return True
    return False


def _row_covers(pred_row: list, gold_row: list) -> bool:
    used: set[int] = set()
    for g in gold_row:
        for i, p in enumerate(pred_row):
            if i not in used and values_equal(g, p):
                used.add(i)
                break
        else:
            return False
    return True


def execution_match(gold_rows: list[list], pred_rows: list[list], ordered: bool = False) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    if ordered:
        return all(_row_covers(p, g) for g, p in zip(gold_rows, pred_rows))
    remaining = list(range(len(pred_rows)))
    for g in gold_rows:
        for j in remaining:
            if _row_covers(pred_rows[j], g):
                remaining.remove(j)
                break
        else:
            return False
    return True
