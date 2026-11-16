.PHONY: setup build benchmark test eval eval-bedrock retrieval-eval serve mcp docker deploy

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[local,eval,dev]"

build:            ## sample data -> bronze/silver/gold + quality checks
	python -m bank_agent.cli build

benchmark:        ## regenerate eval/benchmark.jsonl from the gold layer
	python eval/build_benchmark.py

test:
	python -m pytest

eval:             ## replay LLM: plumbing check (expect 100%)
	python eval/run_eval.py

eval-bedrock:     ## real accuracy with Bedrock (needs AWS credentials + model access)
	BANK_AGENT_LLM=bedrock python eval/run_eval.py --min-accuracy 0.85

retrieval-eval:
	python eval/retrieval_eval.py

serve:
	uvicorn bank_agent.api.app:app --reload --port 8000

mcp:
	python -m bank_agent.mcp_server

docker:
	docker build -t bank-financials-agent . && docker run -p 8000:8000 bank-financials-agent

deploy:           ## needs AWS credentials, Docker and the CDK CLI (npm i -g aws-cdk)
	cd infra && pip install -e "..[infra]" && cdk deploy
