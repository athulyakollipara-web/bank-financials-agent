import asyncio
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(runtime):
    import bank_agent.runtime as rt_module
    from bank_agent.api.app import app

    rt_module.default_runtime.cache_clear()
    return TestClient(app)


def test_health_and_kpis(client):
    assert client.get("/health").json()["status"] == "ok"
    names = {k["name"] for k in client.get("/kpis").json()}
    assert {"net_interest_margin", "efficiency_ratio", "tier1_capital_ratio"} <= names


def test_ask_endpoint(client):
    from tests.test_agent import BENCH

    body = client.post("/ask", json={"question": BENCH[3]["question"], "include_trace": True}).json()
    assert body["rows"] and body["sql"].upper().startswith("SELECT") and body["trace"]


def test_sql_endpoint_rejects_writes(client):
    r = client.post("/sql", json={"sql": "DELETE FROM dim_bank"})
    assert r.status_code == 400


def test_mcp_server_lists_and_calls_tools(runtime):
    from bank_agent import mcp_server

    tools = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert {"ask_bank_question", "list_kpis", "describe_schema", "run_readonly_sql"} <= tools
    out = mcp_server.run_readonly_sql("SELECT COUNT(*) AS n FROM dim_bank")
    assert out["ok"] and out["rows"][0][0] == 120


def test_lambda_handler(runtime):
    from bank_agent import lambda_handler
    from tests.test_agent import BENCH

    resp = lambda_handler.handler({"routeKey": "POST /ask", "body": json.dumps({"question": BENCH[5]["question"]})}, None)
    assert resp["statusCode"] == 200 and json.loads(resp["body"])["rows"]
    assert lambda_handler.handler({"routeKey": "GET /nope"}, None)["statusCode"] == 404
