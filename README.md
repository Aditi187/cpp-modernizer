# 🛠️ Air-Gapped C++ Modernization Engine

> **Transform legacy C/C++ codebases into verified, idiomatic C++17 — fully offline, LLM-powered, compiler-confirmed.**

[![CI](https://github.com/Aditi187/cpp-modernizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditi187/cpp-modernizer/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml) [![C++17](https://img.shields.io/badge/target-C%2B%2B17-orange)]() [![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Most AI code tools rewrite and hope. This engine **proves it works** — by actually compiling the output.

---

## ✨ Key Features

| Feature | What it does |
|---|---|
| **3-Layer Modernization Pipeline** | Deterministic rules → LLM semantic rewrite → deterministic enforcement |
| **Semantic Guard** | Audits function signatures and structure after every LLM call to detect logic drift |
| **Compiler Verification Loop** | Actually compiles output with GCC/Clang; automatically repairs errors with a dedicated Fixer agent |
| **DAG-Aware Batch Processing** | Resolves `#include` dependency order across a codebase before processing — headers before implementations |
| **Interactive Developer Mode** | Review every file's diff, approve/reject/retry with custom LLM feedback before writing to disk |
| **AST-Based Rules** | Uses libclang for token-level precision when available; falls back to regex automatically |
| **compile_commands.json Support** | Reads your project's compilation database to apply correct include paths and compiler flags |
| **Air-Gap Ready** | Works with any OpenAI-compatible API (NVIDIA NIM, local Ollama, LM Studio, etc.) — no cloud dependency |
| **FastAPI Service** | Optional REST API endpoint for CI/CD pipeline integration |

---

## 🏗️ Architecture

```
Input Legacy C++
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                  LangGraph State Machine                │
│                                                         │
│  ┌──────────┐   ┌─────────┐   ┌───────────────────┐   │
│  │ ANALYZER │──▶│ PLANNER │──▶│    MODERNIZER     │   │
│  │ (AST +   │   │ (Risk + │   │  Layer 1: Rules   │   │
│  │  regex)  │   │  Plan)  │   │  Layer 2: LLM     │   │
│  └──────────┘   └─────────┘   │  Layer 3: Rules   │   │
│                                └────────┬──────────┘   │
│                                         │               │
│  ┌──────────────┐   ┌──────────┐        ▼               │
│  │    FIXER     │◀──│ VERIFIER │◀── SEMANTIC_GUARD      │
│  │ (LLM repair) │   │ (g++/    │   (signature audit)    │
│  └──────┬───────┘   │  clang)  │                        │
│         │           └──────────┘                        │
│         └──────────────────────────────────▶ END        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
Verified C++17 Output + JSON Report + Terminal Diff
```

## 🛡️ Security Model

This tool assumes a trusted user running it locally or on an internal team intranet against their own codebase.

| Deployment Scenario | Isolation | Rate Limit | Recommendation |
|---|---|---|---|
| **Local CLI** (single user) | OS-level rlimits | N/A | ✓ Safe, use as-is |
| **Intranet API** (trusted team) | OS-level rlimits | IP-based (in-memory) | ✓ Safe with `--workers 1` |
| **Public SaaS** (untrusted input) | ❌ Insufficient | ❌ Not suitable | ⚠️ Must add container sandbox |

### Isolation & Limits

- **By default:** Compiled user code is isolated using OS-level limits:
  - Linux: `rlimits` (100 MB memory per process, 1 subprocess max)
  - Windows: Job Objects (same memory + process limits)
  - Timeout: 30 seconds (configurable)
- **For public/untrusted deployments:** Wrap in ephemeral Docker containers (future `SANDBOX_MODE=container` feature)

### Rate Limiting

- **Current implementation:** Redis-backed, consistent across all workers via `slowapi`
- Gracefully falls back to in-memory, per-IP counters if Redis is unreachable (with a logged warning)
- By default, API endpoints are limited to 60 requests/minute per client

### API Authentication & Audit

- API endpoints require a `Bearer` token (API_AUTH_TOKEN)
- Supports multi-key attribution (`API_AUTH_TOKENS`) for cross-team deployment
- Automatically logs `submitted_by` (team attribution) into project audit records
- **Security Note**: The web interface stores the token in `sessionStorage` (cleared when the browser closes). For higher-assurance deployments, we strongly recommend implementing a backend session service using `HttpOnly; SameSite=Strict` cookies instead of client-side token storage.

### Summary

- ✓ **Air-gapped intranet:** Safe. CLI mode or single-worker API with trusted users.
- ⚠️ **Public SaaS:** Must implement container-based sandboxing + Redis rate limiting + per-API-key audit trails before exposing to untrusted input.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- **A C++ compiler** — see [Compiler Setup](#compiler-setup) below
- **For air-gapped operation:** [Ollama](https://ollama.ai) with a model pre-downloaded (e.g., `ollama pull qwen2.5-coder:7b`)
- **Optional (for non-air-gapped):** NVIDIA NIM API key or OpenAI API key

### Installation

```bash
git clone https://github.com/Aditi187/cpp-modernizer.git
cd cpp-modernizer
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
```

### Configuration (Local Ollama — Fully Air-Gapped)

**Step 1:** Download Ollama and pull a model:
```bash
# Download from https://ollama.ai
# Then:
ollama pull qwen2.5-coder:7b
# Or try a newer model:
ollama pull qwen3-coder:7b  # if available
```

**Step 2:** Copy `.env.example` to `.env` and configure for local Ollama:

```env
# Air-Gapped: Local Ollama (default, no cloud calls)
WORKFLOW_MODEL_PROVIDER=openai
OPENAI_ENDPOINT_BASE=http://localhost:11434/v1
OPENAI_MODELS=qwen2.5-coder:7b
API_KEY=ollama

# Compiler settings
COMPILER_PATH=g++  # or your custom compiler path
SKIP_VERIFICATION=0

# Optional: C++ standard (default: c++17)
CPP_STANDARD=c++17  # Options: c++14, c++17, c++20, c++23
```

Then start Ollama and the app:
```bash
# Terminal 1: Start Ollama (runs on localhost:11434)
ollama serve

# Terminal 2: Verify Ollama is up
curl http://localhost:11434/api/tags

# Terminal 3: Run modernizer
python cli.py legacy_code.cpp
```

### Compiler Setup

**⚠️ Important:** The engine requires a working C++ compiler for verification.

#### Option A: GCC/Clang via system package manager (Linux/macOS)

```bash
# Ubuntu/Debian
sudo apt-get install g++ build-essential

# macOS (with Homebrew)
brew install gcc

# Then test
g++ --version
```

#### Option B: MSYS2/MinGW64 (Windows) — **Recommended**

1. **Open MSYS2 terminal** (if not installed, download from https://www.msys2.org):
   ```bash
   pacman -Sy
   pacman -S mingw-w64-x86_64-toolchain
   ```

2. **Add compiler to `.env`**:
   ```env
   COMPILER_PATH=C:\msys64\mingw64\bin\g++.exe
   ```

#### Option C: Portable GCC (best for air-gapped)

For fully offline deployment, download a portable compiler once and commit to your repo:

1. Download from: https://github.com/niXman/mingw-builds-binaries/releases (e.g., `x86_64-13.2.0-release-posix-seh-rt_v11-rev1.7z`)
2. Extract to: `./.cpp-compiler/` (in your project root)
3. Add to `.env`:
   ```env
   COMPILER_PATH=./.cpp-compiler/mingw64/bin/g++.exe
   ```
4. **For air-gapped deployment:** Include `.cpp-compiler/` in your project package

#### Option D: MSVC (Visual Studio)

If Visual Studio is installed:
```env
COMPILER_PATH=cl.exe
```



### Configuration (Cloud APIs — Optional)

**To use NVIDIA NIM or OpenAI instead**, update `.env`:

```env
# NVIDIA NIM (free tier available)
WORKFLOW_MODEL_PROVIDER=openai
OPENAI_ENDPOINT_BASE=https://integrate.api.nvidia.com/v1
OPENAI_MODELS=meta/llama3-70b-instruct  # See https://catalog.ngc.nvidia.com for available models
API_KEY=nvapi-xxxx

# OR: OpenAI
WORKFLOW_MODEL_PROVIDER=openai
OPENAI_ENDPOINT_BASE=https://api.openai.com/v1
OPENAI_MODELS=gpt-4o
API_KEY=sk-...
```

⚠️ **Note:** Using cloud APIs means code leaves your network. For truly air-gapped operation, use Ollama as described above.

---

## 🌐 Air-Gap Checklist

Use this checklist to ensure **zero outbound network calls** during modernization:

- [ ] **Ollama installed locally** and verified running on `localhost:11434`
- [ ] **Model pre-downloaded** before disconnecting from the internet:
  ```bash
  ollama pull qwen2.5-coder:7b
  # (or your chosen model)
  ```
- [ ] **`.env` configured for local Ollama only:**
  ```env
  OPENAI_ENDPOINT_BASE=http://localhost:11434/v1
  OPENAI_MODELS=qwen2.5-coder:7b
  # No NVIDIA_API_KEY or OPENAI_API_KEY set
  ```
- [ ] **Offline C++ compiler** available on PATH (GCC, Clang, MSVC)
- [ ] **Test run with `--dry-run`** to ensure zero network calls:
  ```bash
  python cli.py test_input/ --dry-run
  ```
- [ ] **Enable log redaction** for audited deployments: set `LOG_REDACT_SOURCE=1` to redact long source-like lines from modernization.log (recommended for multi-user/public deployments).
- [ ] **Network disconnected** before production runs (optional, but safest)

**Verification:** Monitor `localhost:11434` with `curl http://localhost:11434/api/tags` — if Ollama is up and responding, all traffic is local.

---

## 🎨 Web Dashboard (New!)

The engine now includes a beautiful, fully interactive **C++ Modernization Studio** web UI! 
Instead of staring at terminal output, you can visually track the modernization pipeline and compare the before/after C++ code using the built-in diff viewer.

1. Start the API server:
   ```bash
   uvicorn api:app --port 8000
   ```
2. Open your browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
3. Paste your legacy C++ code into the left panel and click **Modernize**. Watch the pipeline execute in real-time and review the results in Split View or Unified View!

---

## 💻 Usage

### Modernize a single file (defaults to C++17)
```bash
python cli.py legacy_code.cpp
```

### Modernize with a different C++ standard
```bash
python cli.py legacy_code.cpp --cpp-standard c++20
```

### Modernize an entire directory (DAG-ordered, parallel)
```bash
python cli.py ./my_legacy_project/ --workers 4
```

### Interactive mode — review every diff before accepting
```bash
python cli.py ./my_legacy_project/ --interactive
```

### Skip compiler verification (LLM-only mode)
```bash
python cli.py legacy_code.cpp --skip-verify
```

### Dry-run (show changes without writing)
```bash
python cli.py legacy_code.cpp --dry-run
```

### Resume interrupted batch processing
```bash
python cli.py ./my_legacy_project/ --resume
```

### Try it now on the included demo project
```bash
python cli.py test_input/
```

### All CLI options
```
usage: cli.py [-h] [-o OUTPUT] [-v] [--skip-verify] [--cpp-standard {c++14,c++17,c++20,c++23}] 
              [--workers N] [-i] [--dry-run] [--resume] [--version] input

positional arguments:
  input                     Path to a C++ source file OR a directory to batch-process

options:
  -o, --output              Custom output path (single-file mode only)
  -v, --verbose             Enable detailed debug logging
  --skip-verify             Skip compiler verification
  --cpp-standard STANDARD   Target C++ standard (default: c++17)
                            Options: c++14, c++17, c++20, c++23
  --workers N               Parallel workers for batch mode (default: 4)
  -i, --interactive         Human-in-the-loop review mode
  --dry-run                 Show changes without writing to disk
  --resume                  Resume interrupted batch job from project state
  --version                 Show version and exit
```

---

## 🖥️ Example Output

**Sample output from a real test run** using `python cli.py test_input/`. Actual results will vary based on input complexity, LLM model, and hardware. Your modernization score and fix iterations depend on code patterns and LLM latency.

```
╭─────────────── INDUSTRIAL MODERNIZATION BENCHMARK REPORT ───────────────╮
│  Source File          │  SessionManager.cpp                              │
│  Target Standard      │  C++17                                           │
│  Status               │  SUCCESS                                         │
│  Processed Functions  │  6                                               │
│  Legacy Patterns Fixed│  7                                               │
│  Fix Iterations       │  1                                               │
│  Semantic Guard       │  PASSED                                          │
│  Modernization Score  │  0.95                                            │
│  Safety Rating        │  EXCEPTIONAL                                     │
│  Compilation Status   │  STABLE                                          │
│  Attribution          │  LLM: qwen2.5-coder:7b + verified               │
╰──────────────────────────────────────────────────────────────────────────╯

╭──────── TRANSFORMATION DIFF (before → after) ────────╮
│ - #include <stdio.h>           + #include <cstdio>   │
│ - char userName[64];           + std::string userName │
│ - NULL                         + nullptr             │
│ - (char*)malloc(...)           + std::vector<...>    │
╰──────────────────────────────────────────────────────╯
```

To reproduce this output:
```bash
ollama serve &  # Terminal 1
python cli.py test_input/  # Terminal 2
```

**What gets modernized automatically:**
- `NULL` → `nullptr`
- `<stdio.h>` → `<cstdio>` (and all C headers)
- `malloc/free` → `std::vector` / smart pointers
- `char*` → `std::string` / `std::string_view`
- `typedef` → `using`
- `#define` constants → `constexpr`
- C-style casts → `static_cast`
- Manual linked lists → `std::vector` with RAII

---

## 🧪 Tests

Tests are currently pending implementation.

### Model Benchmarking

To evaluate different LLM models for C++ modernization:

```bash
# Run the benchmark suite (requires Ollama running locally)
python tests/benchmark_models.py
```

This compares models on:
- **Modernization Score** — rule reduction + modern idiom adoption
- **Semantic Guard Pass Rate** — function signature consistency
- **Compiler Success Rate** — compilation verification
- **Latency & Token Usage** — performance metrics

Results are saved to `benchmark_results.json`. If a newer model (e.g., `qwen3-coder:7b`) outperforms the default (`qwen2.5-coder:7b`), update `.env.example` and `.env` accordingly.

---

## 🌐 REST API

Start the FastAPI server:
```bash
uvicorn api:app --reload
```
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🤖 AI Assistant Integration (MCP Server) — EXPERIMENTAL

**Status:** This feature is experimental and not recommended for production use.

The modernizer can be exposed as tools for AI assistants (Claude, ChatGPT, etc.) via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

### Running the MCP Server

```bash
python tools/mcp_server.py
```

This starts an MCP server on stdio transport, exposing:
- `read_file(path)` — read C++ files from your project
- `write_file(path, content)` — write modernized code back
- `list_directory(path)` — browse project structure
- `search_code(query, pattern)` — regex search for patterns
- `modernize_cpp_file(path)` — full 3-layer modernization pipeline
- `run_compiler(command, timeout)` — compile and test output

### Using with Claude (or other MCP clients)

The MCP server is designed to be connected to AI assistants that support MCP (e.g., Claude Desktop).

**⚠️ Warning:** This feature is **experimental and still under development**. Prefer the REST API (`/modernize` endpoint) for production integrations. MCP server stability and API may change without notice.

---

## 🔧 Supported LLM Backends

The engine is **model-agnostic** and works with any OpenAI-compatible API. For air-gapped operation, use local Ollama (default).

| Backend | Configuration | Notes |
|---|---|---|
| **🛡️ Local Ollama (Recommended for Air-Gap)** | `OPENAI_ENDPOINT_BASE=http://localhost:11434/v1`<br/>`OPENAI_MODELS=qwen2.5-coder:7b` | Fully offline, zero network calls. [Download Ollama](https://ollama.ai) and `ollama pull qwen2.5-coder:7b` |
| NVIDIA NIM | `OPENAI_ENDPOINT_BASE=https://integrate.api.nvidia.com/v1`<br/>`API_KEY=nvapi-...` | Free tier available; cloud-based, not air-gapped |
| OpenAI (GPT-4/4o) | `OPENAI_ENDPOINT_BASE=https://api.openai.com/v1`<br/>`API_KEY=sk-...` | Cloud-based, requires internet; highest quality models |
| Any OpenAI-compatible | `OPENAI_ENDPOINT_BASE=<your-endpoint>`<br/>`API_KEY=<your-key>` | LM Studio, vLLM, etc. (self-hosted or cloud) |

For a quick Ollama setup:
```bash
# 1. Download and install Ollama from https://ollama.ai
# 2. In a terminal, start Ollama:
ollama serve

# 3. In another terminal, pull a model:
ollama pull qwen2.5-coder:7b

# 4. Configure .env and run the engine
```

---

## 📄 License

MIT
