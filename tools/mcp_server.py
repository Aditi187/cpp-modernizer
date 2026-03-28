# mcp_server.py

from __future__ import annotations

import os
import re
import sys
import json
import glob
import time
import shlex
import hashlib
import threading
import subprocess

from pathlib import Path
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

# ---------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=True,
)

from core.logger import get_logger
from agents.workflow.context import WorkflowContext
from agents.workflow.infra.model_provider import ModelClient

logger = get_logger(__name__)

CPP_MODERNIZATION_SYSTEM_PROMPT = """
You are a C++ Modernization Expert. Your mission is to transform legacy C++ code (C++98/03/11) into high-quality, efficient, and safe modern C++ (C++17/20).

Core Principles:
1.  **Safety First**: Prefer smart pointers (std::unique_ptr, std::shared_ptr) over raw pointers and manual memory management.
2.  **Modern Standard Library**: Use std::string_view, std::optional, std::variant, and modern algorithms.
3.  **RAII**: Ensure everything is Exception-Safe and Resource-Safe.
4.  **Auto & Const**: Use 'auto' where it improves readability and 'const' wherever possible.
5.  **Clean Code**: Maintain the original logic while improving the structure and readability.

Your output should be ONLY the modernized C++ code, wrapped in a single Markdown code block.
"""

print("[WORKING] System initialized: Logging and Propts Ready.")


# ---------------------------------------------------------------------
# parser availability
# ---------------------------------------------------------------------

try:

    from core.parser import CppParser

    PARSER_AVAILABLE = True

except Exception:

    CppParser = None

    PARSER_AVAILABLE = False


def new_cpp_parser():

    if CppParser is None:

        raise RuntimeError(
            "CppParser unavailable"
        )

    return CppParser()


# ---------------------------------------------------------------------
# mcp server
# ---------------------------------------------------------------------

mcp_server = FastMCP(
    "air-gapped-code-tools"
)

ALLOWED_ROOT = str(
    Path.cwd().resolve()
)

print(f"[WORKING] MCP Server Root: {ALLOWED_ROOT}")


IGNORED_DIRS = {

    "node_modules",
    ".git",
    "build",
    ".venv",
    "__pycache__",

}


ALLOWED_COMPILERS = {

    "g++",
    "clang++",
    "gcc",
    "clang",
    "c++",
    "cc",

}


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

def env_int(
    name: str,
    default: int,
    minimum: int,
):

    try:

        return max(
            minimum,
            int(
                os.getenv(
                    name,
                    str(default),
                )
            ),
        )

    except Exception:

        return default


COMPILER_TIMEOUT = env_int(

    "MCP_RUN_COMPILER_TIMEOUT_SECONDS",

    30,

    1,

)

MAX_BINARY_OUTPUT = env_int(

    "MCP_MAX_BINARY_OUTPUT_CHARS",

    10_000,

    256,

)


SYSTEM_PROMPT = (

    CPP_MODERNIZATION_SYSTEM_PROMPT

)

# ---------------------------------------------------------------------
# Workflow Context & Model Bridge (Phase 4)
# ---------------------------------------------------------------------

try:
    CONTEXT = WorkflowContext()
    MODEL_BRIDGE = ModelClient(CONTEXT)
    print("[WORKING] Infrastructure initialized: Context and ModelClient Ready.")
except Exception as e:
    print(f"[ERROR] Failed to initialize project infrastructure: {e}")
    MODEL_BRIDGE = None


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def ensure_safe_path(

    path: str,

) -> str:

    root = Path(ALLOWED_ROOT)

    candidate = Path(path)

    resolved = (

        candidate

        if candidate.is_absolute()

        else root / candidate

    ).resolve()

    if root not in resolved.parents and resolved != root:

        raise ValueError(

            f"Sandbox violation: {path}"

        )

    return str(resolved)


def make_result(

    status: str,

    **data,

) -> str:

    payload = dict(

        status=status,

        **data,

    )

    return json.dumps(

        payload,

        indent=2,

        default=str,

    )


def truncate(

    text: str,

    limit: int,

):

    if len(text) <= limit:

        return text, False

    suffix = "\n...<truncated>"

    return (

        text[: limit - len(suffix)]

        + suffix,

        True,

    )


def run_process(

    command_parts: list[str],

    cwd: str | None,

):

    try:

        result = subprocess.run(

            command_parts,

            cwd=cwd,

            capture_output=True,

            text=True,

            timeout=COMPILER_TIMEOUT,

        )

        return dict(

            ok=True,

            returncode=result.returncode,

            stdout=result.stdout,

            stderr=result.stderr,

        )

    except subprocess.TimeoutExpired:

        return dict(

            ok=False,

            message="timeout",

        )

    except Exception as e:

        return dict(

            ok=False,

            message=str(e),

        )


# ---------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------

@mcp_server.tool()

def read_code(

    file_path: str,

) -> str:

    try:
        path = ensure_safe_path(
            file_path
        )
    except Exception as e:
        print(f"[ERROR] read_code failed safe path check: {e}")
        return make_result(
            "error",
            message=str(e),
        )

    if not os.path.isfile(path):
        print(f"[ERROR] read_code: file not found: {path}")
        return make_result(
            "error",
            message="file not found",
        )

    try:
        text = Path(path).read_text(
            encoding="utf-8"
        )
        print(f"[WORKING] read_code successful: {path} ({len(text)} chars)")
        return make_result(
            "success",
            content=text,
        )

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )


@mcp_server.tool()

def write_code(

    file_path: str,

    content: str,

) -> str:

    try:
        path = ensure_safe_path(
            file_path
        )
    except Exception as e:
        print(f"[ERROR] write_code failed safe path check: {e}")
        return make_result(
            "error",
            message=str(e),
        )

    try:
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(path).write_text(
            content,
            encoding="utf-8",
        )
        print(f"[WORKING] write_code successful: {path} ({len(content)} bytes)")
        return make_result(
            "success",
            bytes_written=len(content),
        )

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )


@mcp_server.tool()

def run_compiler(

    command: str,

    working_directory: str | None = None,

) -> str:

    try:

        parts = shlex.split(command)

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )

    binary = os.path.basename(

        parts[0]

    ).lower()

    if binary not in ALLOWED_COMPILERS:

        return make_result(

            "error",

            message="compiler not allowed",

        )

    cwd = None

    if working_directory:

        try:

            cwd = ensure_safe_path(

                working_directory

            )

        except Exception as e:

            return make_result(

                "error",

                message=str(e),

            )

    result = run_process(
        parts,
        cwd,
    )

    if not result["ok"]:
        print(f"[ERROR] run_compiler failed: {result['message']}")
        return make_result(
            "error",
            message=result["message"],
        )

    status = "success" if result["returncode"] == 0 else "error"
    print(f"[{'WORKING' if status == 'success' else 'ERROR'}] run_compiler finished with code {result['returncode']}")

    return make_result(
        status,
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["returncode"],
    )


@mcp_server.tool()

def run_binary(

    path: str,

    timeout: int = 5,

) -> str:

    try:

        binary_path = ensure_safe_path(

            path

        )

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )

    if not os.path.isfile(

        binary_path

    ):

        return make_result(

            "error",

            message="binary not found",

        )

    try:

        result = subprocess.run(

            [binary_path],

            capture_output=True,

            text=True,

            timeout=max(1, timeout),

        )

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )

    stdout, t1 = truncate(
        result.stdout or "",
        MAX_BINARY_OUTPUT,
    )
    stderr, t2 = truncate(
        result.stderr or "",
        MAX_BINARY_OUTPUT,
    )

    status = "success" if result.returncode == 0 else "error"
    print(f"[{'WORKING' if status == 'success' else 'ERROR'}] run_binary finished with code {result.returncode}")

    return make_result(
        status,
        stdout=stdout,
        stderr=stderr,
        truncated_stdout=t1,
        truncated_stderr=t2,
    )


@mcp_server.tool()

def list_directory(

    path: str,

) -> str:

    try:

        p = ensure_safe_path(

            path or "."

        )

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )

    if not os.path.isdir(p):

        return make_result(

            "error",

            message="not directory",

        )

    entries = []

    for name in sorted(

        os.listdir(p)

    ):

        full = os.path.join(

            p,

            name,

        )

        entries.append(

            dict(

                name=name,

                type="dir"

                if os.path.isdir(full)

                else "file",

            )

        )

    return make_result(

        "success",

        entries=entries,

    )


@mcp_server.tool()

def search_code(

    query: str,

    pattern: str = "*",

) -> str:

    try:

        regex = re.compile(

            query

        )

    except Exception as e:

        return make_result(

            "error",

            message=str(e),

        )

    matches = []

    for path in glob.glob(

        os.path.join(

            ALLOWED_ROOT,

            "**",

            pattern,

        ),

        recursive=True,

    ):

        if not os.path.isfile(

            path

        ):

            continue

        if any(

            d in path

            for d in IGNORED_DIRS

        ):

            continue

        try:

            text = Path(path).read_text(

                encoding="utf-8"

            )

        except Exception:

            continue

        for i, line in enumerate(

            text.splitlines(),

            1,

        ):

            if regex.search(line):

                matches.append(

                    dict(

                        file=path,

                        line=i,

                        text=line,

                    )

                )

        if len(matches) > 500:

            break

    return make_result(

        "success",

        matches=matches,

    )


@mcp_server.tool()
def modernize_cpp_file(
    file_path: str,
    output_path: str | None = None
) -> str:
    """
    Modernizes a C++ file using the full Industrial Modernization Pipeline.
    This includes AST analysis, 3-layer transformation, and verification.
    """
    try:
        abs_path = ensure_safe_path(file_path)
        if not os.path.isfile(abs_path):
            return make_result("error", message=f"File not found: {file_path}")
        
        from agents.workflow.orchestrator import run_modernization_workflow
        
        with open(abs_path, "r", encoding="utf-8") as f:
            code = f.read()

        print(f"[WORKING] MCP: Initiating full modernization for {file_path}")
        final_state = run_modernization_workflow(
            code=code,
            source_file=abs_path,
            output_path=output_path
        )
        
        semantic_ok = final_state.get("semantic_ok", False)
        status = "success" if semantic_ok else "partial_success"
        
        return make_result(
            status,
            output_file=final_state.get("output_file_path"),
            legacy_findings=final_state.get("legacy_findings", []),
            metrics=final_state.get("metrics", {}),
            semantic_ok=semantic_ok
        )
    except Exception as e:
        logger.exception(f"MCP modernization failed: {e}")
        return make_result("error", message=str(e))


# ---------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------

def warmup():
    logger.info("MCP warmup")
    print("[WORKING] Warming up server nodes...")

    if PARSER_AVAILABLE:
        try:
            new_cpp_parser().parse_string("int main(){}")
            print("[WORKING] CppParser verified.")
        except Exception as e:
            print(f"[ERROR] CppParser check failed: {e}")

    if MODEL_BRIDGE:
        try:
            status, msg = MODEL_BRIDGE.check_health()
            if status:
                print(f"[WORKING] Model bridge healthy: {msg}")
            else:
                print(f"[ERROR] Model bridge unhealthy: {msg}")
        except Exception as e:
            print(f"[ERROR] Model health check crashed: {e}")


if __name__ == "__main__":
    print("[WORKING] Starting Air-Gapped MCP Server...")
    warmup()
    mcp_server.run()