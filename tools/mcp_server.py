import os
import shlex
import subprocess
import re
import glob
import json
import sys
# Force UTF-8 output on Windows to prevent UnicodeEncodeError with emoji characters.
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(encoding="utf-8", errors="replace")

import threading
import time
from datetime import datetime

from dotenv import load_dotenv
from typing import Optional
from fastmcp import FastMCP

# Make the project root importable so we can reach core.parser.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(
    dotenv_path=os.path.join(_PROJECT_ROOT, ".env"),
    override=True,
)  # Load .env from project root explicitly and override inherited empty values.

from core.gemini_bridge import (
    CPP_MODERNIZATION_SYSTEM_PROMPT,
    GeminiBridge,
)

try:
    from core.parser import CppParser
    _PARSER_AVAILABLE = True
except Exception:
    _PARSER_AVAILABLE = False


mcp_server = FastMCP("air-gapped-code-tools")


ALLOWED_ROOT = os.path.abspath(os.getcwd())

# ---------------------------------------------------------------------------
# Directories to skip during recursive scans (performance / noise).
# ---------------------------------------------------------------------------
_IGNORED_DIRS: set[str] = {
    "node_modules", ".git", "build", "vcpkg_installed",
    "__pycache__", ".venv", "venv", ".vs", ".cache", ".mypy_cache",
}

# ---------------------------------------------------------------------------
# Whitelist of allowed compiler / build-tool binaries for run_compiler.
# ---------------------------------------------------------------------------
_ALLOWED_COMPILERS: set[str] = {
    "g++", "gcc", "c++", "cc",
    "clang++", "clang",
    "make", "cmake", "ninja",
}

_MODEL_SYSTEM_PROMPT = CPP_MODERNIZATION_SYSTEM_PROMPT
_MODEL_BRIDGE = GeminiBridge.from_env(
    log_fn=lambda message: print(message, file=sys.stderr)
)


def call_model(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini with shared retry and full-response safeguards."""
    return _MODEL_BRIDGE.chat_completion(
        system_prompt,
        user_prompt,
        start_new_trace=True,
    )


def _read_text_if_exists(path: str) -> str:
    """Read UTF-8 text from a path, returning an empty string on failure."""
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
    except Exception:
        return ""
    return ""


def _extract_cpp_inputs_from_command(
    command_parts: list[str],
    working_directory_resolved: Optional[str],
) -> list[dict[str, str]]:
    """Extract C++ source files from a compiler command and return their content."""
    sources: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    cpp_exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
    base_dir = working_directory_resolved or ALLOWED_ROOT

    for token in command_parts[1:]:
        if not token or token.startswith("-"):
            continue

        candidate = token.strip().strip('"').strip("'")
        if not candidate:
            continue

        resolved_path = candidate
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.normpath(os.path.join(base_dir, resolved_path))
        else:
            resolved_path = os.path.normpath(resolved_path)

        if os.path.splitext(resolved_path)[1].lower() not in cpp_exts:
            continue
        if resolved_path in seen_paths:
            continue

        code = _read_text_if_exists(resolved_path)
        if not code:
            continue

        seen_paths.add(resolved_path)
        rel_path = (
            os.path.relpath(resolved_path, ALLOWED_ROOT)
            if resolved_path.startswith(ALLOWED_ROOT)
            else resolved_path
        )
        sources.append({"path": rel_path, "code": code})

    return sources


# ===================================================================
# Standardised JSON result helper
# ===================================================================

def _make_result(status: str, **kwargs) -> str:
    """Build a JSON-ready result dict for uniform tool output.

    Every MCP tool returns this format so LLM / workflow consumers can
    rely on a predictable schema::

        {"status": "success"|"error", ...extra_fields}
    """
    payload: dict = {"status": status}
    payload.update(kwargs)
    return json.dumps(payload, indent=2, default=str)


# ===================================================================
# Sandbox helpers
# ===================================================================

def _contains_parent_traversal(path: str) -> bool:
    """Return True if the path tries to go up a directory using '..'."""
    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    return ".." in parts


def _ensure_within_allowed_root(path: str) -> str:
    """Normalise *path*, resolve symlinks, and verify it stays inside ALLOWED_ROOT.

    Symlinks are resolved with ``os.path.realpath`` so that a symlink
    pointing outside the sandbox cannot bypass the check.
    """
    if _contains_parent_traversal(path):
        raise ValueError(
            f"Sandbox violation: path uses '..' to traverse upwards: {path}"
        )

    if os.path.isabs(path):
        absolute_path = os.path.abspath(path)
    else:
        absolute_path = os.path.abspath(os.path.join(ALLOWED_ROOT, path))

    # Resolve symlinks *after* computing the absolute path.
    resolved_path = os.path.realpath(absolute_path)
    resolved_root = os.path.realpath(ALLOWED_ROOT)

    common_root = os.path.commonpath([resolved_root, resolved_path])
    if common_root != resolved_root:
        raise ValueError(
            "Sandbox violation: path resolves outside the allowed project root.\n"
            f"Requested path: {resolved_path}\n"
            f"Allowed root:   {resolved_root}"
        )

    return resolved_path


def _validate_compiler_binary(command_parts: list[str]) -> str | None:
    """Return ``None`` if the binary is allowed, or an error message."""
    if not command_parts:
        return "Empty command."
    binary = os.path.basename(command_parts[0]).lower()
    # Strip version suffixes (g++-12) and .exe on Windows.
    base = re.sub(r'-\d+(\.\d+)*$', '', binary)
    base = re.sub(r'\.exe$', '', base)
    if base not in _ALLOWED_COMPILERS:
        return (
            f"Binary '{command_parts[0]}' is not in the allowed compiler whitelist: "
            f"{sorted(_ALLOWED_COMPILERS)}"
        )
    return None


def _path_has_ignored_component(filepath: str) -> bool:
    """Return True if any path segment belongs to ``_IGNORED_DIRS``."""
    parts = filepath.replace("\\", "/").split("/")
    return any(p in _IGNORED_DIRS for p in parts)


def _tool_log(tool_name: str, detail: str) -> None:
    """Write a concise tool-invocation log line to stderr."""
    print(f"Tool '{tool_name}' called: {detail}", file=sys.stderr, flush=True)


# ===================================================================
# MCP tools
# ===================================================================

@mcp_server.tool()
def read_code(file_path: str) -> str:
    """Read the contents of a text file from the local filesystem.

    Returns a JSON object with ``status``, ``content``, and ``file_path``
    on success, or ``status`` and ``message`` on error.
    """
    _tool_log("read_code", f"file_path='{file_path}'")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isfile(absolute_path):
        return _make_result("error", message=f"File not found or not a regular file: {absolute_path}")

    try:
        with open(absolute_path, "r", encoding="utf-8") as fh:
            file_contents = fh.read()
        return _make_result(
            "success",
            content=file_contents,
            file_path=os.path.relpath(absolute_path, ALLOWED_ROOT),
        )
    except UnicodeDecodeError:
        return _make_result(
            "error",
            message="Could not decode the file as UTF-8 text. The file may be binary or use an unsupported encoding.",
        )
    except Exception as error:
        return _make_result("error", message=f"Unexpected problem while reading the file: {error!r}")


@mcp_server.tool()
def write_code(file_path: str, content: str) -> str:
    """Write text content to a file inside the sandboxed project root.

    Parent directories are created automatically when they do not exist.

    Truncation guard: if the target file already exists and the incoming content
    is less than 50 % of the existing file’s length, the write is refused with
    an error so that partial/truncated LLM responses cannot silently corrupt
    source files.
    """
    _tool_log("write_code", f"file_path='{file_path}', content_len={len(content)}")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    # ------------------------------------------------------------------
    # Truncation safety check
    # ------------------------------------------------------------------
    if os.path.isfile(absolute_path):
        try:
            with open(absolute_path, "r", encoding="utf-8") as _fh:
                existing_content = _fh.read()
            existing_len = len(existing_content)
            incoming_len = len(content)
            if existing_len > 0 and incoming_len < existing_len * 0.5:
                pct = incoming_len * 100 // existing_len
                return _make_result(
                    "warning",
                    message="ERROR: Potential truncation detected. Please provide the full file content.",
                    existing_chars=existing_len,
                    incoming_chars=incoming_len,
                    truncation_percent=pct,
                )
        except UnicodeDecodeError:
            pass  # Binary file — skip length check.

    directory = os.path.dirname(absolute_path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as error:
            return _make_result("error", message=f"Could not create parent directories: {error!r}")

    try:
        with open(absolute_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return _make_result(
            "success",
            message=f"Wrote {len(content)} characters to {absolute_path}",
            bytes_written=len(content),
        )
    except Exception as error:
        return _make_result("error", message=f"Unexpected problem while writing the file: {error!r}")


# ---------------------------------------------------------------------------
# Compiler output helpers
# ---------------------------------------------------------------------------

def _extract_source_locations(text: str) -> set[tuple[str, str, Optional[str]]]:
    """Extract file/line(/column) locations from compiler output text."""
    locations: set[tuple[str, str, Optional[str]]] = set()
    if not text:
        return locations

    pattern_with_column = re.compile(r"([^\s:]+):(\d+):(\d+)")
    pattern_without_column = re.compile(r"([^\s:]+):(\d+)(?!:)")

    for match in pattern_with_column.finditer(text):
        file_name, line, column = match.groups()
        locations.add((file_name, line, column))

    for match in pattern_without_column.finditer(text):
        file_name, line = match.groups()
        if not any(f == file_name and l == line for (f, l, _) in locations):
            locations.add((file_name, line, None))

    return locations


def _build_location_hints(standard_output: str, error_output: str) -> list[dict]:
    """Return structured location hints extracted from compiler output."""
    locations = set()
    locations.update(_extract_source_locations(error_output))
    locations.update(_extract_source_locations(standard_output))

    hints: list[dict] = []
    for file_name, line, column in sorted(locations):
        entry: dict = {"file": file_name, "line": int(line)}
        if column is not None:
            entry["column"] = int(column)
        hints.append(entry)
    return hints


@mcp_server.tool()
def run_compiler(command: str, working_directory: Optional[str] = None) -> str:
    """Run a local compiler command and return its result as JSON.

    Only whitelisted compiler binaries (g++, clang++, make, cmake, …) are
    permitted.  Returns ``{status, exit_code, stdout, stderr, location_hints}``.
    """
    _tool_log(
        "run_compiler",
        f"command='{command}', working_directory='{working_directory or '.'}'",
    )

    span = _MODEL_BRIDGE.start_span(
        "run_compiler",
        input_payload={
            "command": command,
            "working_directory": working_directory or ".",
        },
    )

    try:
        command_parts = shlex.split(command)
    except ValueError as error:
        result = _make_result("error", message=f"Could not parse the command string: {error!r}")
        parsed = json.loads(result)
        _MODEL_BRIDGE.mark_trace_error("run_compiler parse error", details=parsed)
        _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
        return result

    # --- Compiler whitelist enforcement ---
    rejection = _validate_compiler_binary(command_parts)
    if rejection is not None:
        result = _make_result("error", message=rejection)
        parsed = json.loads(result)
        _MODEL_BRIDGE.mark_trace_error("run_compiler validation error", details=parsed)
        _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
        return result

    if working_directory is not None:
        try:
            working_directory_resolved = _ensure_within_allowed_root(working_directory)
        except ValueError as sandbox_error:
            result = _make_result("error", message=str(sandbox_error))
            parsed = json.loads(result)
            _MODEL_BRIDGE.mark_trace_error("run_compiler sandbox error", details=parsed)
            _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
            return result

        if not os.path.isdir(working_directory_resolved):
            result = _make_result(
                "error",
                message=f"The working directory does not exist or is not a directory: {working_directory_resolved}",
            )
            parsed = json.loads(result)
            _MODEL_BRIDGE.mark_trace_error("run_compiler invalid working directory", details=parsed)
            _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
            return result
    else:
        working_directory_resolved = None

    cpp_inputs = _extract_cpp_inputs_from_command(command_parts, working_directory_resolved)
    _MODEL_BRIDGE.end_span(
        span,
        output_payload={
            "stage": "compiler-input",
            "cpp_sources": cpp_inputs,
        },
    )
    span = _MODEL_BRIDGE.start_span(
        "run_compiler.execution",
        input_payload={
            "command": command,
            "working_directory": working_directory_resolved or ".",
            "cpp_sources": cpp_inputs,
        },
    )

    try:
        completed_process = subprocess.run(
            command_parts,
            cwd=working_directory_resolved or None,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result = _make_result(
            "error",
            message="The compiler program could not be found. "
                    "Confirm the command name is installed and on your PATH.",
        )
        parsed = json.loads(result)
        _MODEL_BRIDGE.mark_trace_error("run_compiler executable missing", details=parsed)
        _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
        return result
    except Exception as error:
        result = _make_result("error", message=f"Unexpected problem while running the compiler: {error!r}")
        parsed = json.loads(result)
        _MODEL_BRIDGE.mark_trace_error("run_compiler execution failed", details=parsed)
        _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
        return result

    exit_code = completed_process.returncode
    stdout = (completed_process.stdout or "").strip()
    stderr = (completed_process.stderr or "").strip()
    hints = _build_location_hints(stdout, stderr)

    result = _make_result(
        "success" if exit_code == 0 else "error",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        location_hints=hints,
    )
    parsed = json.loads(result)

    if parsed.get("status") == "error":
        _MODEL_BRIDGE.mark_trace_error("run_compiler returned error", details=parsed)
        _MODEL_BRIDGE.end_span(span, output_payload=parsed, level="ERROR")
    else:
        _MODEL_BRIDGE.end_span(span, output_payload=parsed)
    return result


@mcp_server.tool()
def run_binary(path: str, timeout: int = 5) -> str:
    """Execute a compiled binary inside the sandbox and capture its output.

    The binary must reside under ALLOWED_ROOT.  A timeout (seconds) prevents
    infinite loops from hanging the server.
    """
    _tool_log("run_binary", f"path='{path}', timeout={timeout}")

    try:
        absolute_path = _ensure_within_allowed_root(path)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isfile(absolute_path):
        return _make_result("error", message=f"Binary not found or not a regular file: {absolute_path}")

    working_directory = os.path.dirname(absolute_path) or ALLOWED_ROOT
    normalized_timeout = max(1, int(timeout))
    completed_process = None
    permission_retry_attempted = False

    while True:
        try:
            completed_process = subprocess.run(
                [absolute_path],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=normalized_timeout,
            )
            break
        except subprocess.TimeoutExpired:
            return _make_result(
                "error",
                message=f"Binary execution exceeded timeout of {timeout} seconds and was terminated.",
            )
        except PermissionError as perm_err:
            if permission_retry_attempted:
                return _make_result(
                    "error",
                    message=f"Permission denied even after chmod 755: {perm_err!r}",
                )
            permission_retry_attempted = True
            _tool_log("run_binary", f"permission denied for '{absolute_path}', attempting chmod 755")
            try:
                os.chmod(absolute_path, 0o755)
            except Exception as chmod_err:
                return _make_result(
                    "error",
                    message=f"Permission denied and chmod 755 failed: {chmod_err!r}",
                )
        except Exception as error:
            return _make_result("error", message=f"Unexpected problem while running the binary: {error!r}")

    exit_code = completed_process.returncode
    stdout = (completed_process.stdout or "").strip()
    stderr = (completed_process.stderr or "").strip()

    return _make_result(
        "success" if exit_code == 0 else "error",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


@mcp_server.tool()
def list_directory(path: str) -> str:
    """List the contents of a directory inside the sandboxed project root."""
    _tool_log("list_directory", f"path='{path}'")

    target = path or "."
    try:
        absolute_path = _ensure_within_allowed_root(target)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isdir(absolute_path):
        return _make_result("error", message=f"Not a directory or does not exist: {absolute_path}")

    try:
        entries = sorted(os.listdir(absolute_path))
    except Exception as error:
        return _make_result("error", message=f"Could not list directory contents: {error!r}")

    items: list[dict] = []
    for name in entries:
        full = os.path.join(absolute_path, name)
        items.append({"name": name, "type": "directory" if os.path.isdir(full) else "file"})

    return _make_result(
        "success",
        directory=os.path.relpath(absolute_path, ALLOWED_ROOT),
        entries=items,
    )


@mcp_server.tool()
def list_tree(path: str, depth: int = 2) -> str:
    """Render a visual directory tree from a path inside the sandbox.

    Directories in ``_IGNORED_DIRS`` (node_modules, .git, build,
    vcpkg_installed, …) are skipped automatically to avoid noise.
    """
    _tool_log("list_tree", f"path='{path}', depth={depth}")

    target = path or "."
    try:
        absolute_path = _ensure_within_allowed_root(target)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isdir(absolute_path):
        return _make_result("error", message=f"Not a directory or does not exist: {absolute_path}")

    max_depth = max(0, int(depth))
    root_display = os.path.relpath(absolute_path, ALLOWED_ROOT)
    if root_display == ".":
        root_display = os.path.basename(ALLOWED_ROOT) or ALLOWED_ROOT

    lines: list[str] = [f"Tree for {root_display} (depth={max_depth}):"]

    def _walk(current_path: str, prefix: str, current_depth: int) -> None:
        if current_depth >= max_depth:
            return
        try:
            entries = sorted(os.listdir(current_path))
        except Exception as error:
            lines.append(f"{prefix}[ERROR] {error!r}")
            return

        # Filter ignored directories.
        entries = [
            e for e in entries
            if not (os.path.isdir(os.path.join(current_path, e)) and e in _IGNORED_DIRS)
        ]

        for index, entry in enumerate(entries):
            full_path = os.path.join(current_path, entry)
            is_last = index == len(entries) - 1
            connector = "`-- " if is_last else "|-- "
            child_prefix = "    " if is_last else "|   "

            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}{entry}/")
                _walk(full_path, prefix + child_prefix, current_depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry}")

    _walk(absolute_path, prefix="", current_depth=0)

    return _make_result("success", tree="\n".join(lines))


@mcp_server.tool()
def search_code(query: str, file_pattern: str = "*") -> str:
    """Search text across project files using glob + regex.

    Directories in ``_IGNORED_DIRS`` are automatically excluded.
    Returns a JSON object with ``matches`` and metadata.
    """
    _tool_log("search_code", f"query='{query}', file_pattern='{file_pattern}'")

    if not query:
        return _make_result("error", message="Query must be a non-empty regex string.")

    try:
        regex = re.compile(query)
    except re.error as regex_error:
        return _make_result("error", message=f"Invalid regex query: {regex_error}")

    pattern = file_pattern or "*"
    glob_pattern = os.path.join(ALLOWED_ROOT, "**", pattern)
    candidate_paths = glob.glob(glob_pattern, recursive=True)

    files: list[str] = []
    for matched in candidate_paths:
        if not os.path.isfile(matched):
            continue
        if _path_has_ignored_component(matched):
            continue
        try:
            _ensure_within_allowed_root(matched)
        except ValueError:
            continue
        files.append(matched)

    files = sorted(set(files))
    if not files:
        return _make_result("success", matches=[], files_scanned=0, message=f"No files matched pattern '{pattern}'.")

    matches: list[dict] = []
    files_scanned = 0
    files_with_decode_errors = 0
    max_results = 500

    for file_path in files:
        files_scanned += 1
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line_number, line_text in enumerate(handle, start=1):
                    if regex.search(line_text):
                        rel_path = os.path.relpath(file_path, ALLOWED_ROOT)
                        matches.append({
                            "file": rel_path,
                            "line": line_number,
                            "text": line_text.rstrip(),
                        })
                        if len(matches) >= max_results:
                            break
        except UnicodeDecodeError:
            files_with_decode_errors += 1
        except Exception:
            continue

        if len(matches) >= max_results:
            break

    return _make_result(
        "success",
        matches=matches,
        total_matches=len(matches),
        files_scanned=files_scanned,
        skipped_decode_errors=files_with_decode_errors,
        truncated=len(matches) >= max_results,
    )


@mcp_server.tool()
def get_file_info(file_path: str) -> str:
    """Return metadata for a file: size, modified time, and line count."""
    _tool_log("get_file_info", f"file_path='{file_path}'")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isfile(absolute_path):
        return _make_result("error", message=f"File not found or not a regular file: {absolute_path}")

    try:
        stat_info = os.stat(absolute_path)
        size_bytes = stat_info.st_size
        modified_iso = datetime.fromtimestamp(stat_info.st_mtime).isoformat()

        line_count = 0
        with open(absolute_path, "r", encoding="utf-8") as handle:
            for _ in handle:
                line_count += 1

        return _make_result(
            "success",
            file_path=os.path.relpath(absolute_path, ALLOWED_ROOT),
            size_bytes=size_bytes,
            last_modified=modified_iso,
            line_count=line_count,
        )
    except UnicodeDecodeError:
        return _make_result(
            "error",
            message="Could not decode the file as UTF-8 for line counting. File may be binary.",
        )
    except Exception as error:
        return _make_result("error", message=f"Unexpected problem while gathering file info: {error!r}")


# ===================================================================
# Helpers for get_context_for_function
# ===================================================================

_PRIMITIVE_TYPES: set[str] = {
    "int", "float", "double", "char", "bool", "void", "long", "short",
    "unsigned", "signed", "auto", "size_t", "ptrdiff_t", "nullptr_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t",  "int16_t",  "int32_t",  "int64_t",
    "wchar_t", "char8_t", "char16_t", "char32_t",
    "const", "static", "inline", "virtual", "explicit", "constexpr",
    "override", "final", "typename", "class", "struct", "enum",
    "template", "namespace", "return", "new", "delete", "nullptr",
    "true", "false", "using", "typedef", "public", "private", "protected",
}


def _extract_template_arguments(text: str) -> list[str]:
    """Extract the top-level comma-separated fragments from inside ``<…>``.

    Handles nesting so ``std::map<std::string, MyRecord>`` yields
    ``["std::string", "MyRecord"]`` rather than splitting mid-token.
    """
    args: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in text:
        if ch == '<':
            depth += 1
            if depth == 1:
                current = []
                continue
        elif ch == '>':
            depth -= 1
            if depth == 0 and current:
                args.append(''.join(current).strip())
                current = []
                continue
        if depth >= 1:
            current.append(ch)

    return args


def _extract_candidate_type_names(text: str) -> list[str]:
    """Return identifier tokens from *text* that might be custom C++ types.

    Improvements over a naive word-boundary scanner:
    - recursively unpacks template arguments (``std::vector<MyType>`` yields MyType)
    - extracts both fully-qualified names (``Foo::Bar``) and their leaf identifiers
    """
    # Expand template arguments into the search text.
    queue = list(_extract_template_arguments(text))
    all_inner: list[str] = []
    while queue:
        fragment = queue.pop()
        all_inner.append(fragment)
        queue.extend(_extract_template_arguments(fragment))

    combined = text + " " + " ".join(all_inner)

    # Extract qualified identifiers (Foo::Bar::Baz).
    qualified = re.findall(r'\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\b', combined)
    # Extract simple identifiers.
    simple = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', combined)

    seen: set[str] = set()
    result: list[str] = []

    for tok in qualified + simple:
        if tok in seen:
            continue
        if tok in _PRIMITIVE_TYPES:
            seen.add(tok)
            continue
        # Skip pure std:: names.
        if tok.startswith("std::") or tok == "std":
            seen.add(tok)
            continue
        seen.add(tok)
        result.append(tok)

    # Also extract the leaf component of qualified user names for lookup.
    for qn in qualified:
        if qn.startswith("std::"):
            continue
        leaf = qn.split("::")[-1]
        if leaf and leaf not in seen and leaf not in _PRIMITIVE_TYPES:
            seen.add(leaf)
            result.append(leaf)

    return result


def _classify_ownership(source_code: str) -> list[str]:
    """Infer ownership / lifecycle notes from a type's source code.

    Detects Rule of Five compliance, move semantics, swap idiom,
    C++20 non-owning views, concepts, and coroutine types.
    """
    hints: list[str] = []

    # --- Rule of Five analysis ---
    has_dtor = bool(re.search(r'~\w+\s*\(', source_code))
    has_copy_ctor = bool(re.search(r'\w+\s*\(\s*const\s+\w+\s*&\s*[),]', source_code))
    has_copy_assign = bool(re.search(r'operator\s*=\s*\(\s*const\s+\w+\s*&', source_code))
    has_move_ctor = bool(re.search(r'\w+\s*\(\s*\w+\s*&&\s*[),]', source_code))
    has_move_assign = bool(re.search(r'operator\s*=\s*\(\s*\w+\s*&&', source_code))
    has_deleted = bool(re.search(r'=\s*delete', source_code))

    r5_count = sum([has_dtor, has_copy_ctor, has_copy_assign, has_move_ctor, has_move_assign])

    if has_deleted:
        if re.search(r'\b(operator\s*=|\w+\s*\(\s*const\s+\w+\s*&)', source_code):
            hints.append("non-copyable (copy constructor or assignment deleted)")
        else:
            hints.append("has explicitly deleted special member(s)")

    if r5_count >= 5:
        hints.append("Rule of Five compliant (all five special members defined)")
    elif r5_count >= 3:
        hints.append(f"Rule of Five partially implemented ({r5_count}/5 special members)")

    if has_move_ctor or has_move_assign:
        hints.append("supports move semantics")

    # Swap idiom.
    if re.search(r'\bswap\s*\(', source_code):
        hints.append("implements swap idiom")

    # Exclusive ownership.
    if re.search(r'\bstd::unique_ptr\b', source_code):
        hints.append("owns exclusive resource via std::unique_ptr — not copyable by default")

    # Shared ownership.
    if re.search(r'\bstd::shared_ptr\b', source_code):
        hints.append("shares ownership via std::shared_ptr")

    # Polymorphic base.
    if re.search(r'\bvirtual\b', source_code):
        hints.append("polymorphic (has virtual method(s)); prefer pointer/reference semantics")

    # Explicit destructor without move (resource managed manually).
    if has_dtor and not has_move_ctor and not has_move_assign:
        hints.append("manages resources (explicit destructor, no move support)")

    # C++20: Non-owning views.
    if re.search(r'\bstd::span\b', source_code):
        hints.append("uses std::span (non-owning contiguous view)")
    if re.search(r'\bstd::string_view\b', source_code):
        hints.append("uses std::string_view (non-owning string reference)")

    # C++20: Concepts and constraints.
    if re.search(r'\brequires\b', source_code):
        hints.append("C++20 constrained (uses requires clause)")
    if re.search(r'\bconcept\b', source_code):
        hints.append("defines or uses C++20 concept")

    # C++20: Coroutines.
    if re.search(r'\bco_await\b|\bco_yield\b|\bco_return\b', source_code):
        hints.append("C++20 coroutine type (uses co_await/co_yield/co_return)")

    return hints if hints else ["standard value semantics (copyable and movable)"]


def _collect_type_bundle(
    seed_type_names: list[str],
    type_definitions: dict[str, str],
    types_meta: list[dict],
) -> dict[str, dict]:
    """Recursively resolve *seed_type_names* and all their base classes.

    Returns ``{type_name: {source_code, bases, ownership_hints}}``.
    """
    name_to_meta: dict[str, dict] = {}
    for t in types_meta:
        n = str(t.get("name") or "")
        if n:
            name_to_meta[n] = t

    bundle: dict[str, dict] = {}
    queue = list(seed_type_names)
    visited: set[str] = set()

    while queue:
        type_name = queue.pop(0)
        if type_name in visited:
            continue
        visited.add(type_name)

        source_code = type_definitions.get(type_name)
        if source_code is None:
            continue

        meta = name_to_meta.get(type_name, {})
        bases = [str(b) for b in (meta.get("bases") or [])]
        ownership_hints = _classify_ownership(source_code)

        bundle[type_name] = {
            "source_code": source_code,
            "bases": bases,
            "ownership_hints": ownership_hints,
        }

        queue.extend(b for b in bases if b not in visited)

    return bundle


def _format_type_bundle(
    bundle: dict[str, dict],
    header_paths: dict[str, str] | None = None,
) -> str:
    """Render the type bundle as a structured, LLM-readable section.

    When *header_paths* is supplied, each type entry includes the relative
    file path where the type was defined.
    """
    if not bundle:
        return ""

    header_paths = header_paths or {}

    lines: list[str] = ["=== TYPE BUNDLE (memory layout & ownership model) ==="]
    for type_name in sorted(bundle):
        info = bundle[type_name]
        lines.append(f"\n--- Type: {type_name} ---")
        hpath = header_paths.get(type_name, "")
        if hpath:
            lines.append(f"Header Path: {hpath}")
        bases = info.get("bases") or []
        if bases:
            lines.append(f"Inherits from: {', '.join(bases)}")
        hints = info.get("ownership_hints") or []
        lines.append(f"Ownership/Lifecycle: {'; '.join(hints)}")
        lines.append("Definition:")
        lines.append(str(info.get("source_code", "")))

    return "\n".join(lines)


# ===================================================================
# Global cross-file type resolution cache (incremental + threaded)
# ===================================================================

class _GlobalProjectMapCache:
    """Lazily scans all C++ source / header files under ALLOWED_ROOT.

    Features:
    - **Incremental indexing** — tracks ``mtime`` per file and only re-parses
      files that changed since the last scan.
    - **Background build** — ``build_in_background()`` spawns a daemon thread
      so the first tool call need not block for large codebases.
    - **Header-path tracking** — records which file each type was defined in
      so ``get_context_for_function`` can report it in the Type Bundle.
    - **``_IGNORED_DIRS`` filtering** — skips node_modules, build, etc.
    """

    _instance: "_GlobalProjectMapCache | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._type_definitions: dict[str, str] = {}
        self._types_meta: list[dict] = []
        self._file_mtimes: dict[str, float] = {}
        self._type_to_file: dict[str, str] = {}
        self._include_graph: dict[str, list[str]] = {}
        self._built: bool = False
        self._building: bool = False

    @classmethod
    def get(cls) -> "_GlobalProjectMapCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # --- Incremental helpers ------------------------------------------------

    def _should_reindex_file(self, filepath: str) -> bool:
        try:
            current_mtime = os.path.getmtime(filepath)
        except OSError:
            return False
        return self._file_mtimes.get(filepath) != current_mtime

    def _index_file(self, parser: "CppParser", filepath: str) -> None:
        try:
            pm = parser.parse_file(filepath, workspace_root=ALLOWED_ROOT)
        except Exception:
            return

        try:
            self._file_mtimes[filepath] = os.path.getmtime(filepath)
        except OSError:
            pass

        # Evict stale entries belonging to this file.
        stale_names = [n for n, f in self._type_to_file.items() if f == filepath]
        for n in stale_names:
            self._type_definitions.pop(n, None)
            self._type_to_file.pop(n, None)
        self._types_meta = [
            t for t in self._types_meta
            if self._type_to_file.get(str(t.get("name", ""))) != filepath
        ]

        # Insert fresh data.
        for name, src in (pm.get("type_definitions") or {}).items():
            self._type_definitions[name] = src
            self._type_to_file[name] = filepath
        self._types_meta.extend(pm.get("types") or [])

        # Track the include graph for this file.
        headers: list[str] = pm.get("headers") or []
        self._include_graph[filepath] = headers

    # --- Build methods ------------------------------------------------------

    def _build(self) -> None:
        if self._building:
            return
        self._building = True
        try:
            if not _PARSER_AVAILABLE:
                self._built = True
                return

            parser = CppParser()
            extensions = ("*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp", "*.hxx")
            for ext in extensions:
                pattern = os.path.join(ALLOWED_ROOT, "**", ext)
                for filepath in glob.glob(pattern, recursive=True):
                    if _path_has_ignored_component(filepath):
                        continue
                    if not self._built or self._should_reindex_file(filepath):
                        self._index_file(parser, filepath)
            self._built = True
        finally:
            self._building = False

    def build_in_background(self) -> threading.Thread:
        """Launch a daemon thread that builds the cache without blocking."""
        thread = threading.Thread(target=self._build, daemon=True)
        thread.start()
        return thread

    def ensure_built(self) -> None:
        if not self._built:
            self._build()

    # --- Query methods ------------------------------------------------------

    def lookup_types(
        self, type_names: list[str],
    ) -> tuple[dict[str, str], list[dict]]:
        """Return (type_definitions, types_meta) for the requested names."""
        self.ensure_built()
        matched_defs = {
            n: self._type_definitions[n]
            for n in type_names
            if n in self._type_definitions
        }
        name_set = set(type_names)
        matched_meta = [
            t for t in self._types_meta
            if str(t.get("name", "")) in name_set
        ]
        return matched_defs, matched_meta

    def get_header_path(self, type_name: str) -> str:
        """Return the workspace-relative path where *type_name* is defined."""
        self.ensure_built()
        filepath = self._type_to_file.get(type_name, "")
        if filepath:
            return os.path.relpath(filepath, ALLOWED_ROOT)
        return ""

    def get_include_graph_for_file(self, filepath: str) -> dict[str, str | None]:
        """Return ``{include_directive: resolved_path | None}`` for *filepath*."""
        self.ensure_built()
        raw_headers = self._include_graph.get(filepath, [])
        result: dict[str, str | None] = {}
        file_dir = os.path.dirname(filepath)
        for hdr in raw_headers:
            match = re.search(r'#\s*include\s*([<"].*[>"])', hdr)
            if not match:
                continue
            header_ref = match.group(1).strip()
            resolved = self._resolve_header(header_ref, file_dir)
            result[hdr.strip()] = resolved
        return result

    def _resolve_header(self, header_ref: str, file_dir: str) -> str | None:
        """Try to resolve a header ref like ``"foo.h"`` or ``<bar.hpp>`` to a real path."""
        if header_ref.startswith('"') and header_ref.endswith('"'):
            bare = header_ref.strip('"')
            candidate = os.path.normpath(os.path.join(file_dir, bare))
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, ALLOWED_ROOT)
            candidate = os.path.normpath(os.path.join(ALLOWED_ROOT, bare))
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, ALLOWED_ROOT)
        elif header_ref.startswith('<') and header_ref.endswith('>'):
            bare = header_ref.strip('<>')
            candidate = os.path.normpath(os.path.join(ALLOWED_ROOT, bare))
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, ALLOWED_ROOT)
        return None

    def invalidate(self) -> None:
        """Force a full rescan on next access."""
        with self._lock:
            self._type_definitions.clear()
            self._types_meta.clear()
            self._file_mtimes.clear()
            self._type_to_file.clear()
            self._include_graph.clear()
            self._built = False


# ===================================================================
# MCP tools — semantic / contextual
# ===================================================================

@mcp_server.tool()
def get_context_for_function(file_path: str, function_fqn: str) -> str:
    """Return a rich modernisation context bundle for a single C++ function.

    The JSON result contains:
    1. **function_context** — signature, body, parameters.
    2. **called_signatures** — signatures of internally called functions.
    3. **type_bundle** — definitions + ownership hints + *header_path* for
       every custom type used in the function (resolved cross-file).
    4. **required_headers** — inferred ``#include`` directives.
    """
    _tool_log(
        "get_context_for_function",
        f"file_path='{file_path}', function_fqn='{function_fqn}'",
    )

    span = _MODEL_BRIDGE.start_span(
        "get_context_for_function",
        input_payload={
            "file_path": file_path,
            "function_fqn": function_fqn,
            "cpp_code": _read_text_if_exists(file_path),
        },
    )

    if not _PARSER_AVAILABLE:
        result = _make_result(
            "error",
            message="CppParser is unavailable. Ensure 'tree-sitter-cpp' is installed.",
        )
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    if not function_fqn or not function_fqn.strip():
        result = _make_result("error", message="function_fqn must be a non-empty fully-qualified function name.")
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        result = _make_result("error", message=str(sandbox_error))
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    if not os.path.isfile(absolute_path):
        result = _make_result("error", message=f"File not found or not a regular file: {absolute_path}")
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    source_text = _read_text_if_exists(absolute_path)

    try:
        parser = CppParser()
        project_map = parser.parse_file(absolute_path, workspace_root=ALLOWED_ROOT)
    except Exception as exc:
        result = _make_result("error", message=f"Failed to parse C++ file: {exc!r}")
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    functions: dict = project_map.get("functions") or {}
    if not isinstance(functions, dict) or function_fqn not in functions:
        available = sorted(functions.keys()) if isinstance(functions, dict) else []
        result = _make_result(
            "error",
            message=f"Function '{function_fqn}' not found in {os.path.basename(absolute_path)}.",
            available_fqns=available,
        )
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    try:
        base_context = parser.get_context_for_function(function_fqn)
    except Exception as exc:
        result = _make_result("error", message=f"Context extraction failed: {exc!r}")
        _MODEL_BRIDGE.end_span(span, output_payload=json.loads(result), level="ERROR")
        return result

    fn_meta = functions[function_fqn]
    signature = str(fn_meta.get("signature") or "")
    body = str(base_context.get("body") or "")
    called_signatures: dict = base_context.get("called_function_signatures") or {}

    # Collect seed type names from parameters + signature + body.
    parameters: list = fn_meta.get("parameters") or []
    param_type_text = " ".join(str(p.get("type") or "") for p in parameters)
    seed_candidates = _extract_candidate_type_names(
        param_type_text + " " + signature + " " + body
    )

    type_definitions: dict = project_map.get("type_definitions") or {}
    types_meta: list = project_map.get("types") or []

    seed_types = [t for t in seed_candidates if t in type_definitions]

    # Cross-file resolution: look up types not found locally.
    unresolved = [t for t in seed_candidates if t not in type_definitions]
    if unresolved:
        global_cache = _GlobalProjectMapCache.get()
        global_defs, global_meta = global_cache.lookup_types(unresolved)
        merged_defs = {**global_defs, **type_definitions}
        local_names = {str(t.get("name", "")) for t in types_meta}
        merged_meta = types_meta + [
            m for m in global_meta if str(m.get("name", "")) not in local_names
        ]
        seed_types.extend(t for t in unresolved if t in global_defs)
    else:
        merged_defs = type_definitions
        merged_meta = types_meta

    type_bundle = _collect_type_bundle(seed_types, merged_defs, merged_meta)

    # Build header_path map for every type in the bundle.
    global_cache = _GlobalProjectMapCache.get()
    header_paths: dict[str, str] = {}
    for type_name in type_bundle:
        hp = global_cache.get_header_path(type_name)
        if hp:
            header_paths[type_name] = hp

    # --- Format human-readable output sections ---
    sections: list[str] = []
    sections.append(f"=== FUNCTION CONTEXT: {function_fqn} ===")
    sections.append(f"\nSignature:\n{signature}")
    sections.append(f"\nBody:\n{body}")

    if called_signatures:
        sections.append("\n=== CALLED FUNCTION SIGNATURES ===")
        for callee_fqn, callee_sig in sorted(called_signatures.items()):
            sections.append(f"  {callee_fqn}:\n    {callee_sig}")

    type_bundle_text = _format_type_bundle(type_bundle, header_paths=header_paths)
    if type_bundle_text:
        sections.append("\n" + type_bundle_text)

    include_reqs: list = (project_map.get("include_requirements") or {}).get(function_fqn, [])
    if include_reqs:
        sections.append(f"\n=== REQUIRED HEADERS ===\n{', '.join(include_reqs)}")

    context_text = "\n".join(sections)

    result = _make_result(
        "success",
        context=context_text,
        function_fqn=function_fqn,
        header_paths=header_paths,
        required_headers=include_reqs,
    )
    _MODEL_BRIDGE.end_span(
        span,
        output_payload={
            "status": "success",
            "function_fqn": function_fqn,
            "cpp_code": source_text,
            "context": json.loads(result),
        },
    )
    return result


@mcp_server.tool()
def get_include_graph(file_path: str) -> str:
    """Return the ``#include`` dependency graph for a C++ source/header file.

    The result maps each ``#include`` directive to its resolved workspace-
    relative path (or ``null`` if the header could not be located).  This
    helps the LLM understand and manage header dependencies.
    """
    _tool_log("get_include_graph", f"file_path='{file_path}'")

    if not _PARSER_AVAILABLE:
        return _make_result(
            "error",
            message="CppParser is unavailable. Ensure 'tree-sitter-cpp' is installed.",
        )

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isfile(absolute_path):
        return _make_result("error", message=f"File not found: {absolute_path}")

    # Ensure the global cache is built so header resolution can use it.
    global_cache = _GlobalProjectMapCache.get()
    global_cache.ensure_built()

    graph = global_cache.get_include_graph_for_file(absolute_path)

    # If the file was not yet indexed (e.g. just created), parse it ad-hoc.
    if not graph:
        try:
            parser = CppParser()
            pm = parser.parse_file(absolute_path, workspace_root=ALLOWED_ROOT)
            headers = pm.get("headers") or []
            file_dir = os.path.dirname(absolute_path)
            graph = {}
            for hdr in headers:
                match = re.search(r'#\s*include\s*([<"].*[>"])', hdr)
                if match:
                    ref = match.group(1).strip()
                    graph[hdr.strip()] = global_cache._resolve_header(ref, file_dir)
        except Exception as exc:
            return _make_result("error", message=f"Failed to parse file: {exc!r}")

    return _make_result(
        "success",
        file_path=os.path.relpath(absolute_path, ALLOWED_ROOT),
        include_graph=graph,
    )


@mcp_server.tool()
def add_header_to_file(file_path: str, header_name: str) -> str:
    """Add an ``#include`` directive to a C++ file if not already present.

    The header is inserted immediately after the last existing ``#include``
    line.  If no includes exist, the directive is prepended at the top.
    """
    _tool_log("add_header_to_file", f"file_path='{file_path}', header_name='{header_name}'")

    if not header_name or not header_name.strip():
        return _make_result("error", message="header_name must be a non-empty string.")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return _make_result("error", message=str(sandbox_error))

    if not os.path.isfile(absolute_path):
        return _make_result("error", message=f"File not found or not a regular file: {absolute_path}")

    header_name = header_name.strip()
    if not (header_name.startswith("<") or header_name.startswith('"')):
        header_name = f"<{header_name}>"
    include_directive = f"#include {header_name}"

    try:
        with open(absolute_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception as exc:
        return _make_result("error", message=f"Could not read file: {exc!r}")

    for line in lines:
        if line.strip() == include_directive:
            return _make_result(
                "success",
                message=f"{include_directive} already exists in {os.path.basename(absolute_path)}.",
                already_present=True,
            )

    last_include_idx = -1
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#include"):
            last_include_idx = idx

    if last_include_idx >= 0:
        lines.insert(last_include_idx + 1, include_directive + "\n")
    else:
        lines.insert(0, include_directive + "\n")

    try:
        with open(absolute_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    except Exception as exc:
        return _make_result("error", message=f"Could not write file: {exc!r}")

    return _make_result(
        "success",
        message=f"Added '{include_directive}' to {os.path.basename(absolute_path)}.",
        already_present=False,
    )


@mcp_server.tool()
def get_compilation_errors(
    command: str, working_directory: Optional[str] = None,
) -> str:
    """Run a compiler command and return a JSON object with structured errors.

    Each entry has the shape ``{file, line, error_message}``.
    On success (exit code 0) the ``errors`` list is empty.
    """
    _tool_log(
        "get_compilation_errors",
        f"command='{command}', working_directory='{working_directory or '.'}'",
    )

    try:
        command_parts = shlex.split(command)
    except ValueError as error:
        return _make_result(
            "error",
            errors=[{"file": "<command>", "line": 0, "error_message": f"Bad command string: {error}"}],
        )

    # Compiler whitelist applies here too.
    rejection = _validate_compiler_binary(command_parts)
    if rejection is not None:
        return _make_result(
            "error",
            errors=[{"file": "<whitelist>", "line": 0, "error_message": rejection}],
        )

    cwd_resolved: str | None = None
    if working_directory is not None:
        try:
            cwd_resolved = _ensure_within_allowed_root(working_directory)
        except ValueError as sandbox_error:
            return _make_result(
                "error",
                errors=[{"file": "<sandbox>", "line": 0, "error_message": str(sandbox_error)}],
            )
        if not os.path.isdir(cwd_resolved):
            return _make_result(
                "error",
                errors=[{"file": "<cwd>", "line": 0, "error_message": f"Directory not found: {cwd_resolved}"}],
            )

    try:
        proc = subprocess.run(
            command_parts,
            cwd=cwd_resolved,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return _make_result(
            "error",
            errors=[{"file": "<compiler>", "line": 0, "error_message": "Compiler not found on PATH."}],
        )
    except Exception as exc:
        return _make_result(
            "error",
            errors=[{"file": "<runtime>", "line": 0, "error_message": repr(exc)}],
        )

    if proc.returncode == 0:
        return _make_result("success", errors=[], exit_code=0)

    # Parse gcc/clang style diagnostics.
    error_pattern = re.compile(
        r'^(?P<file>[^:\s]+):(?P<line>\d+):\d*:?\s*(?:error|fatal error):\s*(?P<msg>.+)$',
        re.MULTILINE,
    )

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    errors: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for m in error_pattern.finditer(combined):
        entry = (m.group("file"), int(m.group("line")), m.group("msg").strip())
        if entry not in seen:
            seen.add(entry)
            errors.append({"file": entry[0], "line": entry[1], "error_message": entry[2]})

    if not errors:
        errors.append({
            "file": "<unknown>",
            "line": 0,
            "error_message": (proc.stderr or proc.stdout or "compilation failed with no output").strip()[:2000],
        })

    return _make_result("error", errors=errors, exit_code=proc.returncode)


if __name__ == "__main__":
    mcp_server.run()
