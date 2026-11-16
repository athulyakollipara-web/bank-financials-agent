"""AWS Lambda entry point (API Gateway HTTP API, payload v2). Runs in AWS mode: Athena + Bedrock."""
from __future__ import annotations

import json

from bank_agent.runtime import default_runtime


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body, default=str)}


def handler(event: dict, context) -> dict:
    route = event.get("routeKey", "")
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "Body must be JSON"})
    rt = default_runtime()
    if route == "GET /health":
        return _response(200, {"status": "ok", "mode": rt.settings.mode})
    if route == "POST /ask":
        question = (body.get("question") or "").strip()
        if not 3 <= len(question) <= 500:
            return _response(400, {"error": "question must be 3-500 characters"})
        result = rt.ask(question)
        result.pop("trace", None)
        return _response(200, result)
    return _response(404, {"error": f"Unknown route {route}"})
