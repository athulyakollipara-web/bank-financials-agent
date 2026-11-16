import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

INFRA = Path(__file__).resolve().parents[1] / "infra"


@pytest.fixture(scope="module")
def template():
    pytest.importorskip("aws_cdk")
    if not shutil.which("node"):
        pytest.skip("CDK synth needs Node.js")
    sys.path.insert(0, str(INFRA))
    import aws_cdk as cdk
    from aws_cdk.assertions import Template
    from stack import BankAgentStack

    app = cdk.App()
    stack = BankAgentStack(app, "Test")
    return Template.from_stack(stack)


def test_core_resources_exist(template):
    for resource, count in [
        ("AWS::Glue::Job", 1), ("AWS::Glue::Crawler", 1), ("AWS::Athena::WorkGroup", 1),
        ("AWS::DynamoDB::Table", 1), ("AWS::Bedrock::Guardrail", 1), ("AWS::Cognito::UserPool", 1),
        ("AWS::ApiGatewayV2::Api", 1),
    ]:
        template.resource_count_is(resource, count)


def test_ask_route_requires_jwt(template):

    template.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "POST /ask", "AuthorizationType": "JWT"})
    template.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "GET /health", "AuthorizationType": "NONE"})


def test_guardrail_blocks_prompt_attacks_and_masks_pii(template):
    from aws_cdk.assertions import Match

    template.has_resource_properties("AWS::Bedrock::Guardrail", {
        "BlockedInputMessaging": "BLOCKED_BY_GUARDRAIL",
        "ContentPolicyConfig": {"FiltersConfig": Match.array_with([Match.object_like({"Type": "PROMPT_ATTACK", "InputStrength": "HIGH"})])},
    })


def test_athena_has_scan_limit(template):
    from aws_cdk.assertions import Match

    template.has_resource_properties("AWS::Athena::WorkGroup", {
        "WorkGroupConfiguration": Match.object_like({"BytesScannedCutoffPerQuery": 1_000_000_000}),
    })
