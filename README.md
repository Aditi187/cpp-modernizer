# 🛠️ Air-Gapped C++ Modernization Engine

> **Transform legacy C/C++ codebases into verified, idiomatic C++17 — fully offline, LLM-powered, compiler-confirmed.**

Most AI code tools rewrite and hope. This engine **proves it works** — by actually compiling the output.

---

## ✨ Key Features

- **3-Layer Modernization Pipeline**: Deterministic rules → LLM semantic rewrite → deterministic enforcement.
- **Semantic Guard**: Audits function signatures and structure after every LLM call to detect logic drift.
- **Compiler Verification Loop**: Actually compiles output with GCC/Clang; automatically repairs errors with a dedicated Fixer agent.
- **Interactive Developer Mode**: Review every file's diff, approve/reject/retry with custom LLM feedback before writing to disk.
- **Air-Gap Ready**: Works with any OpenAI-compatible API (NVIDIA NIM, local Ollama, LM Studio, etc.) — no cloud dependency.
- **Web Dashboard**: Interactive C++ Modernization Studio web UI for visual diffs and pipeline tracking.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- **A C++ compiler** (GCC, Clang, or MSVC)
- **For air-gapped operation:** [Ollama](https://ollama.ai) with a model pre-downloaded.

### Installation

```bash
git clone https://github.com/Aditi187/cpp-modernizer.git
cd cpp-modernizer
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e .
```

### Configuration

Copy `.env.example` to `.env` and configure your preferences.

**Local Ollama (Fully Air-Gapped)**
```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

### Usage

**Web Dashboard (Recommended)**
```bash
uvicorn api:app --port 8000
```
Navigate to `http://127.0.0.1:8000` to interact with the visual diff and pipeline UI.

**CLI Mode**
```bash
# Modernize a single file
python cli.py legacy_code.cpp

# Modernize a whole project interactively
python cli.py ./my_legacy_project/ --interactive
```

