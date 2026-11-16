"""The agent workflow as a LangGraph state machine.

    screen -> plan -> retrieve -> write_sql -> validate -> execute -> verify -> answer
                                     ^            |           |          |
                                     +------------+-----------+----------+   (retry with error feedback)

Each node has one job and its own prompt, so failures are easy to localise in the
trace: the planning agent decides what is needed, the retrieval agent pulls the
governed KPI definitions and schema, the SQL agent writes Athena SQL, SQLGlot and
the executor check it, the verification agent sanity-checks the numbers, and the
answer agent explains them.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from bank_agent.agent.grounding import numeric_groundedness
from bank_agent.llm import prompts
from bank_agent.llm.base import LLM
from bank_agent.retrieval.hybrid import HybridRetriever
from bank_agent.safety import ScreenResult
from bank_agent.semantic.catalog import SemanticCatalog
from bank_agent.sql.executor import Executor, SqlError
from bank_agent.sql.validator import validate_sql


class AgentState(TypedDict, total=False):
    question: str
    blocked: bool
    plan: dict
    context: list[str]
    context_ids: list[str]
    sql: str
    validated_sql: str
    errors: list[str]
    attempts: int
    columns: list[str]
    rows: list[list]
    answer: str
    grounding: dict
    models: list[str]
    trace: list[dict]


def _log(state: AgentState, step: str, started: float, **info: Any) -> list[dict]:
    return state.get("trace", []) + [{"step": step, "ms": round((time.perf_counter() - started) * 1000, 1), **info}]


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def build_agent(
    llm: LLM,
    retriever: HybridRetriever,
    catalog: SemanticCatalog,
    executor: Executor,
    screen: Callable[[str], ScreenResult],
    max_attempts: int = 3,
):
    allowed = catalog.allowed_columns()
    sql_system = prompts.SQL_SYSTEM.format(schema=catalog.schema_text())

    def screen_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        result = screen(state["question"])
        return {
            "question": result.text, "blocked": not result.allowed, "attempts": 0, "errors": [], "models": [],
            "trace": _log(state, "screen", t, allowed=result.allowed, reasons=result.reasons),
        }

    def plan_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        resp = llm.complete("plan", prompts.PLANNER_SYSTEM, prompts.plan_prompt(state["question"]), max_tokens=300)
        try:
            plan = json.loads(_strip_fences(resp.text))
        except json.JSONDecodeError:
            plan = {"intent": "analytics", "kpis": [], "search_terms": [state["question"]]}
        return {"plan": plan, "models": state["models"] + [resp.model_id],
                "trace": _log(state, "plan", t, plan=plan, model=resp.model_id)}

    def retrieve_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        plan = state.get("plan", {})
        query = " ".join([state["question"], *plan.get("search_terms", []), *plan.get("kpis", [])])
        hits = retriever.search(query, k=8)
        context = [h.doc.text for h in hits]
        # Always include KPI definitions the planner named explicitly, even if retrieval missed them.
        for name in plan.get("kpis", []):
            kpi = catalog.kpis.get(name)
            if kpi and not any(f"({name})" in c for c in context):
                context.append(f"KPI {kpi.label} ({kpi.name}). SQL: {kpi.sql}")
        return {"context": context, "context_ids": [h.doc.doc_id for h in hits],
                "trace": _log(state, "retrieve", t, hits=[(h.doc.doc_id, round(h.score, 3), h.sources) for h in hits])}

    def write_sql_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        prompt = prompts.sql_prompt(state["question"], state.get("plan", {}), state["context"], state.get("errors", []), state.get("sql", ""))
        resp = llm.complete("sql", sql_system, prompt, max_tokens=800)
        sql = _strip_fences(resp.text)
        return {"sql": sql, "attempts": state["attempts"] + 1, "models": state["models"] + [resp.model_id],
                "trace": _log(state, "write_sql", t, attempt=state["attempts"] + 1, model=resp.model_id, sql=sql)}

    def validate_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        result = validate_sql(state["sql"], allowed)
        return {"validated_sql": result.sql, "errors": result.errors,
                "trace": _log(state, "validate", t, ok=result.ok, errors=result.errors)}

    def execute_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        try:
            res = executor.run(state["validated_sql"])
        except SqlError as e:
            return {"errors": [f"Execution failed: {e}"], "trace": _log(state, "execute", t, ok=False, error=str(e))}
        return {"columns": res.columns, "rows": res.rows, "errors": [],
                "trace": _log(state, "execute", t, ok=True, row_count=len(res.rows))}

    def verify_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        problems = []
        if not state.get("rows"):
            problems.append("The query returned no rows. Check filter values (state codes like 'MA', quarter labels like '2025Q4', bank names).")
        else:
            for ci, col in enumerate(state["columns"]):
                if any(k in col.lower() for k in ("ratio", "margin", "return", "roa", "roe", "nim")):
                    vals = [r[ci] for r in state["rows"] if isinstance(r[ci], (int, float))]
                    if vals and (max(vals) > 5 or min(vals) < -1):
                        problems.append(f"Column {col} has implausible ratio values (max {max(vals):.2f}). Ratios should be fractions; check the KPI formula.")
        return {"errors": problems, "trace": _log(state, "verify", t, ok=not problems, problems=problems)}

    def answer_node(state: AgentState) -> AgentState:
        t = time.perf_counter()
        if state.get("blocked"):
            return {"answer": "This request was blocked by the safety screen.", "trace": _log(state, "answer", t, blocked=True)}
        if state.get("plan", {}).get("intent") == "out_of_scope":
            return {"answer": "I can only answer questions about the bank financial data in this dataset.",
                    "trace": _log(state, "answer", t, out_of_scope=True)}
        if "rows" not in state:
            msg = "I couldn't produce a valid query for that question. Last errors: " + "; ".join(state.get("errors", []))
            return {"answer": msg, "trace": _log(state, "answer", t, failed=True)}
        resp = llm.complete("answer", prompts.ANSWER_SYSTEM, prompts.answer_prompt(state["question"], state["columns"], state["rows"]), max_tokens=400)
        answer = resp.text.strip()
        if state.get("errors"):
            answer += "\n\nNote: automatic checks flagged this result: " + "; ".join(state["errors"])
        grounding = numeric_groundedness(answer, state["rows"])
        return {"answer": answer, "grounding": grounding, "models": state["models"] + [resp.model_id],
                "trace": _log(state, "answer", t, model=resp.model_id, grounding=grounding)}

    def after_screen(state: AgentState) -> str:
        return "answer" if state.get("blocked") else "plan"

    def after_plan(state: AgentState) -> str:
        return "answer" if state.get("plan", {}).get("intent") == "out_of_scope" else "retrieve"

    def retry_or(next_node: str) -> Callable[[AgentState], str]:
        def route(state: AgentState) -> str:
            if state.get("errors"):
                return "write_sql" if state["attempts"] < max_attempts else ("answer" if next_node == "answer" else "give_up")
            return next_node
        return route

    g = StateGraph(AgentState)
    for name, fn in [("screen", screen_node), ("plan", plan_node), ("retrieve", retrieve_node), ("write_sql", write_sql_node),
                     ("validate", validate_node), ("execute", execute_node), ("verify", verify_node), ("answer", answer_node)]:
        g.add_node(name, fn)
    g.set_entry_point("screen")
    g.add_conditional_edges("screen", after_screen, {"plan": "plan", "answer": "answer"})
    g.add_conditional_edges("plan", after_plan, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("retrieve", "write_sql")
    g.add_edge("write_sql", "validate")
    g.add_conditional_edges("validate", retry_or("execute"), {"write_sql": "write_sql", "execute": "execute", "give_up": "answer"})
    g.add_conditional_edges("execute", retry_or("verify"), {"write_sql": "write_sql", "verify": "verify", "give_up": "answer"})
    g.add_conditional_edges("verify", retry_or("answer"), {"write_sql": "write_sql", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()
