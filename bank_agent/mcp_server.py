"""MCP server: lets Claude Desktop (or any MCP client) use the agent as a tool.

Add to Claude Desktop's config (``claude_desktop_config.json``)::

    {
      "mcpServers": {
        "bank-financials": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["-m", "bank_agent.mcp_server"],
          "env": {"BANK_AGENT_LLM": "bedrock", "AWS_REGION": "us-east-1"}
        }
      }
    }
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from bank_agent.runtime import default_runtime

server = MCPServer(
    name="bank-financials",
    instructions=(
        "Answers questions about US bank call report data (assets, deposits, loans, income, capital) "
        "using governed KPI definitions. Use list_kpis to see available metrics, ask_bank_question for "
        "plain-English questions, and run_readonly_sql for exact queries against dim_bank and fact_bank_quarter."
    ),
)


@server.tool(description="Ask a plain-English question about bank financials. Returns the answer, the SQL used and the result rows.")
def ask_bank_question(question: str) -> dict:
    r = default_runtime().ask(question)
    return {k: r[k] for k in ("answer", "sql", "columns", "rows", "grounding")}


@server.tool(description="List the governed KPI definitions (name, label, formula, synonyms).")
def list_kpis() -> list[dict]:
    return [{"name": k.name, "label": k.label, "sql": k.sql, "synonyms": list(k.synonyms)}
            for k in default_runtime().catalog.kpis.values()]


@server.tool(description="Describe the tables and columns that can be queried.")
def describe_schema() -> str:
    return default_runtime().catalog.schema_text()


@server.tool(description="Run a read-only Athena-dialect SELECT against dim_bank / fact_bank_quarter. The query is validated first.")
def run_readonly_sql(sql: str, max_rows: int = 100) -> dict:
    return default_runtime().run_readonly_sql(sql, max_rows)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
