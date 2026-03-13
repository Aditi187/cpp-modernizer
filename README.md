# Air-Gapped Codebase Modernization Engine

A tool for analyzing and modernizing C++ codebases in fully air-gapped (offline) environments using local LLMs via [Ollama](https://ollama.com) and a LangGraph-powered multi-agent workflow.

## Overview

This engine parses legacy C++ source files, builds a dependency graph, and runs a three-node LangGraph workflow — **Analyzer → Modernizer → Verifier** — to automatically suggest and verify modernized code. All processing happens locally; no internet connection is required.

## Features

- **C++ Parsing** — Extracts functions and structure from `.cpp` files using `tree-sitter`.
- **Dependency Graph** — Builds a call graph to determine safe modernization order.
- **LangGraph Workflow** — Multi-step agent pipeline with an Analyzer, Modernizer, and Verifier node (with feedback loop).
- **Differential Testing** — Compiles and compares original vs. modernized code to catch regressions.
- **MCP Tool Server** — Exposes project tools via a [FastMCP](https://github.com/jlowin/fastmcp) server for agent use.
- **Air-Gapped** — Runs entirely offline using a local Ollama LLM instance.

## Project Structure

```
├── agents/
│   └── workflow.py          # LangGraph workflow (Analyzer → Modernizer → Verifier)
├── core/
│   ├── parser.py            # C++ source parser (tree-sitter)
│   ├── graph.py             # Dependency graph & modernization ordering
│   ├── differential_tester.py  # Compile & diff-test original vs. modernized code
│   └── inspect_parser.py    # Parser inspection utilities
├── tools/
│   └── mcp_server.py        # FastMCP tool server
├── requirements.txt
├── pyproject.toml
└── test.cpp                 # Sample C++ file for testing
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) running locally (default: `http://localhost:11434`)
- A C++ compiler (`g++` / `clang++`) available on `PATH`

## Installation

```bash
# Clone the repository
git clone <https://github.com/Aditi187/Air-Gapped-Codebase-Modernization-Engine/tree/main>
cd air-gapped-codebase-modernization-engine

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Start Ollama** and pull a model (e.g. `ollama pull codellama`).
2. Run the modernization workflow against a C++ file:

```bash
python agents/workflow.py
```

3. To start the MCP tool server:

```bash
python tools/mcp_server.py
```

## License

MIT
