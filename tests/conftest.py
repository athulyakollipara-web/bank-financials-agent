import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory) -> Path:
    """Generate the seeded sample data and run the full pipeline once per test session."""
    wh = tmp_path_factory.mktemp("warehouse")
    os.environ["BANK_AGENT_WAREHOUSE"] = str(wh)
    os.environ["BANK_AGENT_MODE"] = "local"
    os.environ["BANK_AGENT_LLM"] = "replay"
    os.environ.pop("BEDROCK_GUARDRAIL_ID", None)
    from bank_agent.config import get_settings
    from bank_agent.data.generate import generate
    from bank_agent.pipeline.medallion import get_spark, run_all

    s = get_settings()
    generate(s.raw_dir)
    run_all(get_spark(), s.raw_dir, s.warehouse)
    return wh


@pytest.fixture(scope="session")
def runtime(warehouse):
    from bank_agent.runtime import build_runtime

    return build_runtime()
