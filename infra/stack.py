"""AWS CDK stack for AWS mode.

Creates: S3 data lake, Glue database + PySpark job + crawler, Athena workgroup,
DynamoDB KPI table, Bedrock Guardrail, Cognito user pool, and a container
Lambda behind an HTTP API with a Cognito JWT authorizer. IAM is scoped to the
specific resources the agent needs.

OpenSearch is intentionally not provisioned by default: the semantic layer is
~40 documents, which in-memory hybrid search handles well. Set
``OPENSEARCH_HOST`` once the corpus grows (policies, runbooks, filings).
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_athena as athena
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_glue as glue
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SENTINEL = "BLOCKED_BY_GUARDRAIL"  # must match bank_agent/safety.py
GLUE_DB = "bank_gold"


class BankAgentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------- data lake + pipeline ----------------
        lake = s3.Bucket(
            self, "DataLake",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        s3deploy.BucketDeployment(
            self, "PipelineCode",
            sources=[s3deploy.Source.asset(str(ROOT), exclude=["*", "!bank_agent/pipeline/medallion.py", "!infra/glue_entry.py"])],
            destination_bucket=lake,
            destination_key_prefix="code",
        )

        glue_role = iam.Role(self, "GlueRole", assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
                             managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole")])
        lake.grant_read_write(glue_role)

        glue.CfnDatabase(self, "GoldDb", catalog_id=self.account, database_input=glue.CfnDatabase.DatabaseInputProperty(name=GLUE_DB))
        glue.CfnJob(
            self, "MedallionJob",
            name="bank-agent-medallion",
            role=glue_role.role_arn,
            glue_version="5.0",
            worker_type="G.1X",
            number_of_workers=2,
            command=glue.CfnJob.JobCommandProperty(name="glueetl", python_version="3",
                                                   script_location=f"s3://{lake.bucket_name}/code/infra/glue_entry.py"),
            default_arguments={
                "--extra-py-files": f"s3://{lake.bucket_name}/code/bank_agent/pipeline/medallion.py",
                "--RAW_PATH": f"s3://{lake.bucket_name}/raw/",
                "--WAREHOUSE": f"s3://{lake.bucket_name}/",
                "--enable-metrics": "true",
                "--enable-continuous-cloudwatch-log": "true",
            },
        )
        glue.CfnCrawler(
            self, "GoldCrawler",
            name="bank-agent-gold",
            role=glue_role.role_arn,
            database_name=GLUE_DB,
            targets=glue.CfnCrawler.TargetsProperty(s3_targets=[
                glue.CfnCrawler.S3TargetProperty(path=f"s3://{lake.bucket_name}/gold/dim_bank/"),
                glue.CfnCrawler.S3TargetProperty(path=f"s3://{lake.bucket_name}/gold/fact_bank_quarter/"),
            ]),
        )
        workgroup = athena.CfnWorkGroup(
            self, "Workgroup",
            name="bank-agent",
            recursive_delete_option=True,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                enforce_work_group_configuration=True,
                bytes_scanned_cutoff_per_query=1_000_000_000,  # 1 GB guard against runaway queries
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(output_location=f"s3://{lake.bucket_name}/athena-results/"),
            ),
        )

        # ---------------- semantic layer ----------------
        kpi_table = dynamodb.Table(
            self, "KpiTable",
            partition_key=dynamodb.Attribute(name="name", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=True),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---------------- guardrail ----------------
        guardrail = bedrock.CfnGuardrail(
            self, "Guardrail",
            name="bank-agent-input",
            blocked_input_messaging=BLOCKED_SENTINEL,
            blocked_outputs_messaging=BLOCKED_SENTINEL,
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(filters_config=[
                bedrock.CfnGuardrail.ContentFilterConfigProperty(type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE"),
            ]),
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(pii_entities_config=[
                bedrock.CfnGuardrail.PiiEntityConfigProperty(type=t, action="ANONYMIZE")
                for t in ("US_SOCIAL_SECURITY_NUMBER", "US_BANK_ACCOUNT_NUMBER", "EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER")
            ]),
        )

        # ---------------- auth ----------------
        pool = cognito.UserPool(self, "Users", self_sign_up_enabled=False, sign_in_aliases=cognito.SignInAliases(email=True),
                                removal_policy=RemovalPolicy.DESTROY)
        client = pool.add_client("ApiClient", auth_flows=cognito.AuthFlow(user_password=True, user_srp=True))

        # ---------------- API ----------------
        fn = lambda_.DockerImageFunction(
            self, "AgentFn",
            code=lambda_.DockerImageCode.from_image_asset(str(ROOT), file="Dockerfile.lambda",
                                                          exclude=["warehouse", "mlruns", "cdk.out", ".venv", "eval/results", ".git"]),
            memory_size=1024,
            timeout=Duration.seconds(60),
            environment={
                "BANK_AGENT_MODE": "aws",
                "BANK_AGENT_LLM": "bedrock",
                "ATHENA_DATABASE": GLUE_DB,
                "ATHENA_WORKGROUP": workgroup.name,
                "KPI_TABLE": kpi_table.table_name,
                "BEDROCK_GUARDRAIL_ID": guardrail.attr_guardrail_id,
                "BEDROCK_GUARDRAIL_VERSION": "DRAFT",
            },
        )
        kpi_table.grant_read_data(fn)
        lake.grant_read(fn, "gold/*")
        lake.grant_read_write(fn, "athena-results/*")
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults", "athena:StopQueryExecution"],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/{workgroup.name}"],
        ))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["glue:GetDatabase", "glue:GetTable", "glue:GetTables", "glue:GetPartitions"],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog",
                       f"arn:aws:glue:{self.region}:{self.account}:database/{GLUE_DB}",
                       f"arn:aws:glue:{self.region}:{self.account}:table/{GLUE_DB}/*"],
        ))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:Converse"],
            resources=["arn:aws:bedrock:*::foundation-model/*", f"arn:aws:bedrock:*:{self.account}:inference-profile/*"],
        ))
        fn.add_to_role_policy(iam.PolicyStatement(actions=["bedrock:ApplyGuardrail"], resources=[guardrail.attr_guardrail_arn]))

        api = apigw.HttpApi(self, "Api", api_name="bank-agent")
        auth = authorizers.HttpUserPoolAuthorizer("CognitoAuth", pool, user_pool_clients=[client])
        integration = integrations.HttpLambdaIntegration("AgentIntegration", fn)
        api.add_routes(path="/ask", methods=[apigw.HttpMethod.POST], integration=integration, authorizer=auth)
        api.add_routes(path="/health", methods=[apigw.HttpMethod.GET], integration=integration)

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "DataLakeBucket", value=lake.bucket_name)
        CfnOutput(self, "KpiTableName", value=kpi_table.table_name)
        CfnOutput(self, "UserPoolId", value=pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=client.user_pool_client_id)
        CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
