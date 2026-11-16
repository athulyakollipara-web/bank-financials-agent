"""FastAPI service. Interactive docs at http://localhost:8000/docs once running."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bank_agent.runtime import default_runtime

app = FastAPI(
    title="Bank Financials Analyst Agent",
    description="Ask plain-English questions about bank call report data. Answers come from validated SQL over governed KPIs.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Which 5 banks in MA had the highest net interest margin in 2025Q4?"])
    include_trace: bool = False


class SqlRequest(BaseModel):
    sql: str = Field(..., max_length=5000)
    max_rows: int = Field(100, ge=1, le=1000)


@app.get("/health")
def health() -> dict:
    rt = default_runtime()
    return {"status": "ok", "mode": rt.settings.mode, "llm": rt.settings.llm}


@app.get("/kpis")
def kpis() -> list[dict]:
    return [
        {"name": k.name, "label": k.label, "description": k.description, "sql": k.sql, "synonyms": list(k.synonyms)}
        for k in default_runtime().catalog.kpis.values()
    ]


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        result = default_runtime().ask(req.question)
    except KeyError as e:  # replay mode, unknown question
        raise HTTPException(status_code=422, detail=str(e)) from e
    if not req.include_trace:
        result.pop("trace", None)
    return result


@app.post("/sql")
def sql(req: SqlRequest) -> dict:
    result = default_runtime().run_readonly_sql(req.sql, req.max_rows)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["errors"])
    return result
