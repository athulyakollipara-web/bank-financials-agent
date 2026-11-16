# Bank Financials Analyst Agent

[![ci](https://github.com/athulyakollipara-web/bank-financials-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/athulyakollipara-web/bank-financials-agent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue) ![aws](https://img.shields.io/badge/AWS-Bedrock%20%7C%20Athena%20%7C%20Glue-orange)

Ask plain-English questions about US bank financials and get answers backed by validated SQL:

> *"Which 5 banks in MA had the highest net interest margin last quarter?"*
> *"Which banks had the largest quarter-over-quarter deposit growth in 2025Q2?"*
> *"List banks in TX with a tier 1 capital ratio below 10%."*

The agent never guesses formulas. KPIs like NIM, efficiency ratio and Tier 1 ratio come from a **governed semantic layer**, every query is **structurally validated with SQLGlot** before it runs, and every number in the answer is **checked against the query result**. It runs as a REST API, as an **MCP server** you can plug into Claude Desktop, and deploys to AWS with one CDK command.

![demo](docs/images/demo.png)

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data pipeline (PySpark: local or AWS Glue)"]
        RAW[Call report extract<br/>FDIC BankFind API] --> B[Bronze<br/>raw + lineage]
        B --> S[Silver<br/>typed, de-duplicated]
        S --> G[Gold<br/>dim_bank, fact_bank_quarter]
        G --> GX{{Great Expectations<br/>quality gate}}
    end

    subgraph Agent["LangGraph agent"]
        Q[Question] --> SC[Screen<br/>Bedrock Guardrails]
        SC --> P[Planning agent]
        P --> R[Retrieval agent<br/>BM25 + k-NN + RRF + rerank]
        R --> W[SQL agent<br/>Claude on Bedrock]
        W --> V[SQLGlot validator]
        V -- errors --> W
        V --> X[Execute<br/>Athena / Spark SQL]
        X -- errors --> W
        X --> VF[Verification agent]
        VF -- implausible --> W
        VF --> A[Answer agent<br/>+ numeric groundedness]
    end

    KPI[(Semantic layer<br/>KPIs in DynamoDB)] --> R
    GX --> X
    A --> API[FastAPI / Lambda]
    A --> MCP[MCP server]
```

| Layer | What it does | Tech |
|---|---|---|
| Ingestion | Pulls call report data (or generates a seeded sample with the same layout) | Python, FDIC BankFind API |
| Medallion pipeline | Bronze → Silver → Gold. Silver fixes formatting, casts types and keeps only the latest amended filing | PySpark (runs locally and as a Glue 5.0 job) |
| Data quality | 13 checks on gold tables, e.g. unique (cert, date), deposits ≤ assets | Great Expectations |
| Semantic layer | 8 governed KPI formulas with synonyms, schema docs and a glossary | YAML → DynamoDB |
| Retrieval | Hybrid BM25 + vector k-NN, Reciprocal Rank Fusion, second-stage reranker | OpenSearch (or in-memory), Titan Embeddings V2 |
| NL2SQL | Writes Athena SQL; SQLGlot checks it's one read-only SELECT over approved tables and columns, adds a LIMIT, and transpiles to Spark SQL locally | Claude Sonnet on Bedrock, SQLGlot |
| Orchestration | Plan → retrieve → write → validate → execute → verify → answer, with error feedback loops (max 3 attempts) | LangGraph |
| Model routing | Per-step model choice, fallback on throttling, prompt caching | Bedrock Converse API (Claude + Nova) |
| Safety | Prompt-attack blocking and PII masking before any model call | Bedrock Guardrails |
| Serving | REST API, MCP tools, Lambda behind a Cognito-protected HTTP API | FastAPI, MCP, API Gateway, Cognito |
| Evaluation | 150-question NL2SQL benchmark, retrieval recall, numeric groundedness, all logged to MLflow; CI gate | MLflow, pytest, GitHub Actions |
| Infrastructure | S3, Glue job + crawler, Athena workgroup with scan limit, DynamoDB, Guardrail, Cognito, Lambda, HTTP API | AWS CDK (Python) |

## Quickstart (local, no AWS account needed)

Requires Python 3.11+ and Java 17+ (for PySpark).

```bash
git clone https://github.com/athulyakollipara-web/bank-financials-agent.git
cd bank-financials-agent
make setup && source .venv/bin/activate

make build     # generate sample data, run bronze/silver/gold, run quality checks
make test      # 36 tests: pipeline, validator, retrieval, agent loop, API, MCP, CDK
make serve     # http://localhost:8000/docs
```

```bash
python -m bank_agent.cli ask "Which 3 banks had the largest quarter-over-quarter deposit growth in 2025Q2?" --trace
```

Local mode uses a **replay LLM** that returns the benchmark's reference SQL. This tests every other part of the system offline and for free, but it only knows the 150 benchmark questions. For free-form questions, switch to Bedrock:

```bash
export BANK_AGENT_LLM=bedrock AWS_REGION=us-east-1   # needs Bedrock model access for Claude + Nova
python -m bank_agent.cli ask "Which community banks in Ohio have the weakest capital ratios?"
```

### Use it from Claude Desktop (MCP)

```json
{
  "mcpServers": {
    "bank-financials": {
      "command": "/absolute/path/to/bank-financials-agent/.venv/bin/python",
      "args": ["-m", "bank_agent.mcp_server"],
      "env": { "BANK_AGENT_LLM": "bedrock", "AWS_REGION": "us-east-1" }
    }
  }
}
```

Tools exposed: `ask_bank_question`, `list_kpis`, `describe_schema`, `run_readonly_sql`.

### Use real FDIC data

```bash
python -m bank_agent.data.fetch_fdic --states MA,NY,CA --since 20240331
python -m bank_agent.cli build --use-existing-raw
```

`fetch_fdic.py` maps FDIC field codes to the pipeline's raw columns. Fields marked `verify` in `FIELD_MAP` should be checked against the API's field definitions before use.

## Evaluation

### Retrieval: does the right KPI definition reach the SQL agent?

Measured with `make retrieval-eval` (local hashing embedder; Titan embeddings in AWS mode).

| Suite | Questions | BM25 recall@1 | k-NN recall@1 | **Hybrid recall@1** | Hybrid recall@3 |
|---|---|---|---|---|---|
| Benchmark KPI questions (use catalog terms) | 80 | 100% | 85% | **100%** | 100% |
| Held-out paraphrases (avoid KPI names and synonyms) | 20 | 80% | 80% | **95%** | 100% |

The paraphrase set is the honest test: questions like *"How much does each bank spend to generate a dollar of revenue?"* never mention "efficiency ratio". Each retriever alone misses different questions; fusing them gets 95% top-1. It's a 20-question set, so treat it as directional.

### NL2SQL: execution accuracy on 150 questions

`eval/benchmark.jsonl` has 150 questions with reference SQL across 8 categories: lookups, governed KPIs, rankings, aggregations, counts, thresholds, trends, and window functions (QoQ / YoY). A prediction counts as correct when its result set matches the reference. Extra columns, column order and rounding are tolerated (see `bank_agent/evaluation.py`).

```bash
make eval            # replay LLM: must be 100%. Proves every reference query validates and runs through the agent
make eval-bedrock    # real accuracy with Bedrock; fails below 85%
```

<!-- BEDROCK_RESULTS: replace this comment with the table from eval/results/bedrock-*.md after running `make eval-bedrock` -->

Each run logs accuracy, first-attempt accuracy, accuracy per category, numeric groundedness and latency to MLflow (`mlflow ui`).

## Design decisions

- **Governed KPIs instead of letting the model write formulas.** "Efficiency ratio" has one definition here, and it lives in data (DynamoDB), not in a prompt. Changing it doesn't need a deploy, and every answer uses the same formula.
- **Validate before executing.** SQLGlot rejects writes, multiple statements, unknown tables and invented columns, and its error messages are fed back to the SQL agent. Most first-attempt mistakes are fixed on the retry (`tests/test_agent.py::test_sql_agent_retries_with_validator_feedback`).
- **Write SQL once, run it anywhere.** The model always writes Athena (Trino) SQL. Locally, SQLGlot transpiles it to Spark SQL, so tests run the same query text that production runs.
- **Check numbers, not just text.** The numeric groundedness check flags any figure in the answer that isn't in the result set, after allowing for % and $M/$B conversions.
- **OpenSearch is optional.** The semantic layer is about 40 documents, and in-memory hybrid search handles that in about 1 ms. The OpenSearch backend (`retrieval/opensearch_backend.py`) is there for when the corpus grows to include policy documents or filings. Leaving it out keeps the default deployment cheap.
- **Cost guards.** The Athena workgroup has a 1 GB per-query scan limit, the validator forces a LIMIT, and cheaper models handle planning and answer wording while the strongest model writes SQL.

## Deploy to AWS

```bash
npm install -g aws-cdk && make deploy
```

Then upload raw files to `s3://<DataLakeBucket>/raw/`, run the `bank-agent-medallion` Glue job and the `bank-agent-gold` crawler, and seed the KPI table:

```python
from bank_agent.semantic.catalog import SemanticCatalog
SemanticCatalog.from_yaml().sync_to_dynamodb("<KpiTableName>", "us-east-1")
```

Remove everything with `cdk destroy`.

## Project structure

```
bank_agent/
  data/          sample generator + FDIC API client
  pipeline/      PySpark medallion + Great Expectations suite
  semantic/      KPI definitions, schema docs, glossary (+ DynamoDB sync)
  retrieval/     BM25, embeddings, hybrid fusion + reranker, OpenSearch backend
  sql/           SQLGlot validator/transpiler, Athena + Spark executors
  llm/           Bedrock router with fallback + prompt caching, replay LLM, prompts
  agent/         LangGraph workflow, numeric groundedness
  api/           FastAPI app
  mcp_server.py  MCP tools
eval/            benchmark builder, NL2SQL eval (MLflow), retrieval eval
infra/           CDK stack + Glue entry point
tests/           36 tests
```

## Data note

The repository ships with a **seeded synthetic dataset** in the FDIC call report layout so everything runs offline and in CI. Bank names are invented and figures are generated. Use `fetch_fdic.py` to load real public call report data.
