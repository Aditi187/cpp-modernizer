# Air-Gapped Codebase Modernization Engine

Modernizes legacy C++ code using a LangGraph workflow with an OpenAI-backed LLM and deterministic fallback.

## Project Structure

```text
agents/   # Workflow orchestration and modernization agents
core/     # Parser, graph, LLM bridge, tester, and modernization engines
tools/    # MCP server and integration tools
tests/    # Unit and integration tests
docs/     # Refactoring and technical documentation
cache/    # Runtime cache data (not required for source control)
```

## Overview

The engine parses C++ source, builds dependency order, modernizes function-by-function, compiles, and verifies behavior. The workflow path is Analyzer -> Modernizer -> Verifier (and tester routing), with automatic no-LLM fallback when provider health or quota is unavailable.

## Features

- C++ parsing with tree-sitter.
- Dependency-aware modernization ordering.
- OpenAI provider bridge for model calls.
- Deterministic rule-based fallback when LLM is unavailable.
- Differential compile/run parity checks.
- LangFuse trace and generation observability.
- MCP server for tool-based integration.

## Requirements

- Python 3.12 or 3.13.
- C++ compiler on PATH (g++ or clang++).
- OpenAI API key.

## Installation

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment

Configure [.env](.env) with OpenAI settings.

```bash
WORKFLOW_MODEL_PROVIDER=openai
API_KEY=...
OPENAI_MODELS=gpt-5.3-codex-xhigh
OPENAI_ENABLE_CACHE=1
OPENAI_MAX_CALLS_PER_MINUTE=35
OPENAI_INTER_REQUEST_DELAY_SECONDS=1.5
OPENAI_429_BASE_DELAY_SECONDS=60
OPENAI_429_MAX_DELAY_SECONDS=900
WORKFLOW_BATCH_SIZE=3
```

## Run Workflow

```bash
python agents/workflow.py
```

Default sample input is [test.cpp](test.cpp), and output is written to [test_modernized.cpp](test_modernized.cpp).

## Run MCP Server

```bash
python tools/mcp_server.py
```

## Notes

- If OpenAI health fails (for example quota/rate-limit), the workflow can continue in deterministic fallback mode.
- LangFuse is optional but enabled when LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST are set.

## Git Hygiene

- Runtime artifacts are ignored (for example cache snapshots, token usage, generated reports, and logs).
- Keep only source, tests, and docs in commits.
- Before push, run:

```bash
pytest -q
```

## License

MIT
