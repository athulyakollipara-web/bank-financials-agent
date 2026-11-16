"""Input screening before a question reaches any model.

AWS mode calls Amazon Bedrock Guardrails (``ApplyGuardrail``) with the guardrail
created by the CDK stack (prompt-attack filter + PII handling). Local mode uses a
small rule-based screen so the behaviour can be tested offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BLOCKED_SENTINEL = "BLOCKED_BY_GUARDRAIL"

INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) prompt",
    r"you are now",
    r"reveal (your|the) (system )?prompt",
    r"\b(drop|delete|truncate|insert|update|alter)\s+(table|from|into)\b",
]
PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "ACCOUNT_NUMBER": r"\b\d{10,17}\b",
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
}


@dataclass
class ScreenResult:
    allowed: bool
    text: str
    reasons: list[str]


def local_screen(text: str) -> ScreenResult:
    reasons = [f"prompt_attack:{p}" for p in INJECTION_PATTERNS if re.search(p, text, re.I)]
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, redacted):
            reasons.append(f"pii_redacted:{label}")
            redacted = re.sub(pattern, f"[{label}]", redacted)
    blocked = any(r.startswith("prompt_attack") for r in reasons)
    return ScreenResult(not blocked, redacted, reasons)


def bedrock_screen(text: str, guardrail_id: str, version: str, region: str) -> ScreenResult:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.apply_guardrail(
        guardrailIdentifier=guardrail_id, guardrailVersion=version, source="INPUT",
        content=[{"text": {"text": text}}],
    )
    if resp["action"] != "GUARDRAIL_INTERVENED":
        return ScreenResult(True, text, [])
    output = (resp.get("outputs") or [{"text": ""}])[0].get("text", "")
    # The CDK stack sets the guardrail's blocked-input message to BLOCKED_SENTINEL.
    # Any other intervention is PII masking: continue with the masked text.
    if output.strip() == BLOCKED_SENTINEL:
        return ScreenResult(False, text, ["bedrock_guardrail:blocked"])
    return ScreenResult(True, output, ["bedrock_guardrail:pii_masked"])
