# Air-Gapped Codebase Modernization Engine

A tool for analyzing and modernizing C++ codebases with a LangGraph-powered multi-agent workflow and hosted LLM access through Gemini.

## Overview

This engine parses legacy C++ source files, builds a dependency graph, and runs a three-node LangGraph workflow — **Analyzer → Modernizer → Verifier** — to automatically suggest and verify modernized code. Source analysis and compilation remain local, while model inference can be routed through Gemini instead of a self-hosted local model.

## Features

- **C++ Parsing** — Extracts functions and structure from `.cpp` files using `tree-sitter`.
- **Dependency Graph** — Builds a call graph to determine safe modernization order.
- **LangGraph Workflow** — Multi-step agent pipeline with an Analyzer, Modernizer, and Verifier node (with feedback loop).
- **Differential Testing** — Compiles and compares original vs. modernized code to catch regressions.
- **MCP Tool Server** — Exposes project tools via a [FastMCP](https://github.com/jlowin/fastmcp) server for agent use.
- **Hosted Model Support** — Uses Gemini with an API key, model fallback, exponential backoff, and full-response safeguards to avoid truncated rewrites.
- **Observability (LangFuse)** — Captures modernization traces, tool spans, Gemini generations, and token usage/cost-related metadata.

## Project Structure

```
├── agents/
│   └── workflow.py          # LangGraph workflow (Analyzer → Modernizer → Verifier)
├── core/
│   ├── parser.py            # C++ source parser (tree-sitter)
│   ├── graph.py             # Dependency graph & modernization ordering
│   ├── differential_tester.py  # Compile & diff-test original vs. modernized code
│   ├── gemini_bridge.py     # Gemini client + LangFuse tracing bridge
│   └── inspect_parser.py    # Parser inspection utilities
├── tools/
│   └── mcp_server.py        # FastMCP tool server
├── requirements.txt
├── pyproject.toml
└── test.cpp                 # Sample C++ file for testing
```

## Requirements

- **Python 3.12 or 3.13** (Python 3.14 triggers Pydantic V1 compatibility warnings in `langchain-core`; use 3.12/3.13 for a clean run)
- A `GEMINI_API_KEY` (or reuse `OPENROUTER_API_KEY` temporarily as a fallback env name)
- A C++ compiler (`g++` / `clang++`) available on `PATH`

## Installation

```bash
# Clone the repository
git clone <https://github.com/Aditi187/Air-Gapped-Codebase-Modernization-Engine/tree/main>
cd air-gapped-codebase-modernization-engine

# Create and activate a virtual environment (use Python 3.12 or 3.13)
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install all dependencies (networkx and python-dotenv are now included)
pip install -r requirements.txt
```

## Usage

1. Configure Gemini:

```bash
# Required
GEMINI_API_KEY=your_key_here

# Strong defaults for large context and full-file rewrites
GEMINI_MODELS=gemini-1.5-pro,gemini-1.5-flash
GEMINI_MAX_OUTPUT_TOKENS=8192

# LangFuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

`GEMINI_MODELS` is tried in order with automatic fallback if a model is unavailable.
Gemini Pro is first by default because it is better suited for precise, full-file C++ modernization and larger rewrites than the flash tier.
The shared Gemini bridge also retries suspiciously short code outputs with a stronger full-response reminder so file rewrites do not collapse into tiny partial responses.
2. Run the modernization workflow against a C++ file:

```bash
python agents/workflow.py
```

3. To start the MCP tool server:

```bash
python tools/mcp_server.py
```

## LangFuse Expectations

- Each modernization run creates a trace named `CPP-Modernization`.
- The `get_context_for_function` and `run_compiler` MCP tools are captured as spans.
- Gemini calls are captured as generations with `model`, `prompt_tokens`, and `completion_tokens` metadata.
- If `run_compiler` returns an error status, the trace is marked as `Error`.

## License

MIT
