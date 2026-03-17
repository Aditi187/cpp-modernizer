# Air-Gapped Codebase Modernization Engine

A tool for analyzing and modernizing C++ codebases with a LangGraph-powered multi-agent workflow and local LLM inference through Ollama.

## Overview

This engine parses legacy C++ source files, builds a dependency graph, and runs a three-node LangGraph workflow — **Analyzer → Modernizer → Verifier** — to automatically suggest and verify modernized code. Source analysis, compilation, and model inference all run locally using Ollama.

## Features

- **C++ Parsing** — Extracts functions and structure from `.cpp` files using `tree-sitter`.
- **Dependency Graph** — Builds a call graph to determine safe modernization order.
- **LangGraph Workflow** — Multi-step agent pipeline with an Analyzer, Modernizer, and Verifier node (with feedback loop).
- **Differential Testing** — Compiles and compares original vs. modernized code to catch regressions.
- **MCP Tool Server** — Exposes project tools via a [FastMCP](https://github.com/jlowin/fastmcp) server for agent use.
- **Local Model Support** — Uses Ollama with a local code model (default: `deepseek-coder:6.7b`) to avoid API quotas and rate limits.
- **Observability (LangFuse)** — Captures modernization traces, tool spans, Gemini generations, and token usage/cost-related metadata.

## Project Structure

```
├── agents/
│   └── workflow.py          # LangGraph workflow (Analyzer → Modernizer → Verifier)
├── core/
│   ├── parser.py            # C++ source parser (tree-sitter)
│   ├── graph.py             # Dependency graph & modernization ordering
│   ├── differential_tester.py  # Compile & diff-test original vs. modernized code
│   ├── local_ollama_bridge.py # Local Ollama client + LangFuse tracing bridge
│   └── inspect_parser.py    # Parser inspection utilities
├── tools/
│   └── mcp_server.py        # FastMCP tool server
├── requirements.txt
├── pyproject.toml
└── test.cpp                 # Sample C++ file for testing
```

## Requirements

- **Python 3.12 or 3.13** (Python 3.14 triggers Pydantic V1 compatibility warnings in `langchain-core`; use 3.12/3.13 for a clean run)
- [Ollama](https://ollama.com) installed and running locally
- A local model pulled (default: `deepseek-coder:6.7b`)
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

1. Configure Ollama:

```bash
# Required local runtime
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-coder:6.7b
OLLAMA_TIMEOUT_SECONDS=300

# LangFuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Pull the model once before running:

```bash
ollama pull deepseek-coder:6.7b
```
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
- Ollama calls are captured as generations with `model` metadata.
- If `run_compiler` returns an error status, the trace is marked as `Error`.

## License

MIT
