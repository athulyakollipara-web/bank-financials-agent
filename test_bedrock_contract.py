"""Contract tests for the Bedrock client, without calling AWS.

botocore's Stubber validates every request against the official Bedrock Runtime
API model, so a wrong parameter name or shape fails here instead of in a paid run.
"""
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from bank_agent.llm.bedrock import BedrockLLM

PRIMARY, FALLBACK = "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "us.amazon.nova-pro-v1:0"


def _ok(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150},
        "metrics": {"latencyMs": 900},
    }


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    return BedrockLLM(PRIMARY, FALLBACK, "us-east-1")


def _expected(model: str, cache: bool = True) -> dict:
    system = [{"text": "SYS"}] + ([{"cachePoint": {"type": "default"}}] if cache else [])
    return {"modelId": model, "system": system, "messages": [{"role": "user", "content": [{"text": "PROMPT"}]}],
            "inferenceConfig": {"maxTokens": 800, "temperature": 0}}


def test_sql_task_routes_to_primary_with_prompt_cache(llm):
    with Stubber(llm.client) as stub:
        stub.add_response("converse", _ok("SELECT 1"), _expected(PRIMARY))
        r = llm.complete("sql", "SYS", "PROMPT", max_tokens=800)
    assert (r.text, r.model_id, r.input_tokens) == ("SELECT 1", PRIMARY, 120)


def test_falls_back_when_primary_is_throttled(llm):
    with Stubber(llm.client) as stub:
        stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429,
                              expected_params=_expected(PRIMARY))
        stub.add_response("converse", _ok("SELECT 2"), _expected(FALLBACK))
        r = llm.complete("sql", "SYS", "PROMPT", max_tokens=800)
    assert r.model_id == FALLBACK


def test_planning_uses_the_cheaper_model_first(llm):
    with Stubber(llm.client) as stub:
        stub.add_response("converse", _ok("{}"), _expected(FALLBACK))
        assert llm.complete("plan", "SYS", "PROMPT", max_tokens=800).model_id == FALLBACK


def test_non_retryable_errors_are_raised(llm):
    with Stubber(llm.client) as stub:
        stub.add_client_error("converse", service_error_code="AccessDeniedException", http_status_code=403)
        with pytest.raises(ClientError):
            llm.complete("sql", "SYS", "PROMPT", max_tokens=800)


def test_prompt_cache_can_be_disabled(monkeypatch, llm):
    monkeypatch.setenv("BEDROCK_PROMPT_CACHE", "0")
    plain = BedrockLLM(PRIMARY, FALLBACK, "us-east-1")
    with Stubber(plain.client) as stub:
        stub.add_response("converse", _ok("x"), _expected(PRIMARY, cache=False))
        plain.complete("sql", "SYS", "PROMPT", max_tokens=800)
