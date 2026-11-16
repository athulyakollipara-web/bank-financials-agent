# Local-mode demo image: Spark SQL over the sample gold layer + FastAPI.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bank_agent ./bank_agent
COPY eval ./eval
RUN pip install --no-cache-dir -e ".[local]"

# Build the sample warehouse at image build time so the container starts ready.
RUN python -m bank_agent.cli build

ENV BANK_AGENT_MODE=local BANK_AGENT_LLM=replay
EXPOSE 8000
CMD ["uvicorn", "bank_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
