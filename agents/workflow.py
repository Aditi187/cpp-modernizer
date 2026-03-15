"""
LangGraph Workflow for Code Modernization Engine  # This top-level string briefly explains that this file defines the multi-step modernization workflow.
Implements a 3-node workflow: Analyzer -> Modernizer -> Verifier (with feedback loop)  # This line describes the three main nodes and the feedback loop structure.
"""  # This closing triple quote ends the module-level documentation string.

import sys
import os as os_module
# Force UTF-8 output on Windows to prevent UnicodeEncodeError with emoji characters.
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os_module.path.join(_PROJECT_ROOT, ".env"),
    override=True,
)  # Load .env from project root explicitly and override inherited empty values.

from core.parser import extract_functions_from_cpp_file
from core.parser import detect_legacy_patterns
from core.parser import CppParser
from core.graph import (
    DependencyGraph,
    build_analysis_report,
    get_modernization_order,
)
from core.differential_tester import (
    run_differential_test,
    compile_cpp_source,
)
from core.gemini_bridge import (
    CPP_MODERNIZATION_SYSTEM_PROMPT,
    GeminiBridge,
)
from core.openrouter_bridge import OpenRouterBridge
from core.inspect_parser import score_cpp23_compliance
from agents.function_modernizer import FunctionModernizer, similarity


_MODEL_SYSTEM_PROMPT = CPP_MODERNIZATION_SYSTEM_PROMPT
_MODEL_BRIDGE = GeminiBridge.from_env(log_fn=print)


def _build_openrouter_fallback_bridge() -> OpenRouterBridge | None:
    fallback_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not fallback_key:
        return None
    try:
        return OpenRouterBridge.from_env(log_fn=print)
    except Exception as exc:
        print(f"OpenRouter fallback initialization failed: {exc}")
        return None


_FALLBACK_MODEL_BRIDGE = _build_openrouter_fallback_bridge()
_MIN_CPP23_COMPLIANCE_PERCENT = 40
MAX_WORKERS = 2


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _read_bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


_STRICT_CPP23_MODE = (
    _read_bool_env("WORKFLOW_STRICT_MODE", False)
    or _read_bool_env("CPP23_STRICT_MODE", False)
)
_STRICT_CPP23_TARGET_PERCENT = _read_bounded_int_env(
    "CPP23_STRICT_TARGET_PERCENT",
    70,
    0,
    100,
)


def call_model(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini with shared retry and full-response safeguards."""
    return _MODEL_BRIDGE.chat_completion(system_prompt, user_prompt)


def check_model_health() -> bool:
    """Verify the Gemini API key is set and the endpoint is reachable."""
    is_healthy, message = _MODEL_BRIDGE.check_health()
    print(("✅ " if is_healthy else "❌ ") + message)
    return is_healthy


def _parse_functions_from_source(source_code: str) -> list[dict[str, Any]]:
    """Parse C++ source from text and return function metadata."""

    temp_file_path: Optional[str] = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".cpp",
            mode="w",
            encoding="utf-8",
        )
        temp_file_path = temp_file.name
        temp_file.write(source_code)
        temp_file.flush()
        temp_file.close()
        return extract_functions_from_cpp_file(temp_file_path)
    except Exception:
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _build_callers_map(dependency_map: dict[str, list[str]]) -> dict[str, list[str]]:
    callers_map: dict[str, list[str]] = {name: [] for name in dependency_map.keys()}
    for caller_name, callees in dependency_map.items():
        for callee_name in callees:
            if callee_name in callers_map and caller_name not in callers_map[callee_name]:
                callers_map[callee_name].append(caller_name)
    return callers_map


# Regex to match DeepSeek special tokens like <｜begin▁of▁sentence｜> that leak
# into generated code.  Handles both fullwidth (U+FF5C ｜) and ASCII pipe (|).
_DEEPSEEK_SPECIAL_TOKEN_RE = re.compile(r"<[｜|][^>]{1,40}[｜|]>")


def _clean_model_code_block(text: str) -> str:
    """Extract code from the model's response, stripping markdown fences and prose.

    If the response contains a fenced code block (```), only the content inside
    the *first* such block is returned.  This prevents explanatory text that the
    model sometimes emits before or after the code from polluting the source.
    """
    # Try to extract the first fenced code block.
    fence_match = re.search(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Fallback: strip any stray triple-backtick lines.
        cleaned = re.sub(r"```(?:[^\n]*)\n?", "", text)
        cleaned = cleaned.strip()
    # Strip DeepSeek special tokens (e.g. <｜begin▁of▁sentence｜>) that the
    # model sometimes injects into its output.
    cleaned = _DEEPSEEK_SPECIAL_TOKEN_RE.sub("", cleaned)
    return cleaned


def _split_function_signature_and_body(function_source: str) -> tuple[str, str]:
    """Split a full function definition into signature prefix and body text."""
    first_brace = function_source.find("{")
    last_brace = function_source.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return function_source, ""
    signature = function_source[: first_brace + 1]
    body = function_source[first_brace + 1:last_brace]
    return signature, body


def _transform_char_ptr_literal_declarations(body: str) -> tuple[str, int]:
    """Convert local `char* name = \"...\";` into `std::string name = \"...\";`."""
    pattern = re.compile(
        r"(^|\n)([ \t]*)char\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"[^\"\\]*(?:\\.[^\"\\]*)*\")\s*;"
    )

    replacements = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        prefix, indent, name, literal = match.groups()
        return f"{prefix}{indent}std::string {name} = {literal};"

    return pattern.sub(_repl, body), replacements


def _transform_null_macro_to_nullptr(body: str) -> tuple[str, int]:
    """Convert legacy NULL macro tokens to nullptr in function bodies."""
    pattern = re.compile(r"\bNULL\b")
    replacements = len(pattern.findall(body))
    return pattern.sub("nullptr", body), replacements


def _transform_malloc_to_unique_ptr(body: str) -> tuple[str, int]:
    """Convert local malloc + free pairs to std::make_unique for simple local buffers."""
    pattern = re.compile(
        r"(^|\n)([ \t]*)([A-Za-z_][A-Za-z0-9_:<>]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\([^)]+\)\s*)?malloc\(([^\)]*)\)\s*;"
    )

    replacements = 0
    freed_vars: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        prefix, indent, base_type, var_name, malloc_args = match.groups()
        freed_vars.append(var_name)

        size_pattern = re.compile(
            r"^\s*sizeof\s*\(\s*" + re.escape(base_type) + r"\s*\)\s*\*\s*(.+)$"
        )
        size_match = size_pattern.match(malloc_args.strip())
        if size_match:
            count_expr = size_match.group(1).strip()
            replacement = f"{indent}auto {var_name} = std::make_unique<{base_type}[]>({count_expr});"
        else:
            replacement = f"{indent}auto {var_name} = std::make_unique<{base_type}>();"
        return f"{prefix}{replacement}"

    updated = pattern.sub(_repl, body)
    for var_name in freed_vars:
        updated = re.sub(
            r"(^|\n)[ \t]*free\s*\(\s*" + re.escape(var_name) + r"\s*\)\s*;",
            r"\1",
            updated,
        )
    return updated, replacements


def _transform_index_for_to_range_loop(body: str) -> tuple[str, int]:
    """Convert simple index-based loops over one container into range-based loops."""
    loop_pattern = re.compile(
        r"for\s*\(\s*int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\s*;\s*\1\s*<\s*([^;\)]+)\s*;\s*(?:\+\+\1|\1\+\+)\s*\)\s*\{([\s\S]*?)\}",
        re.MULTILINE,
    )

    replacements = 0

    def _loop_repl(match: re.Match[str]) -> str:
        nonlocal replacements
        index_var = match.group(1)
        limit_expr = match.group(2).strip()
        loop_body = match.group(3)

        # Support common patterns: i < count or i < container.size().
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", limit_expr):
            candidates = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*" + re.escape(index_var) + r"\s*\]", loop_body)
            if len(set(candidates)) != 1:
                return match.group(0)
            container_name = candidates[0]
        else:
            size_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*size\s*\(\s*\)", limit_expr)
            if not size_match:
                return match.group(0)
            container_name = size_match.group(1)

        loop_body_rewritten = re.sub(
            r"\b" + re.escape(container_name) + r"\s*\[\s*" + re.escape(index_var) + r"\s*\]",
            "item",
            loop_body,
        )
        replacements += 1
        return f"for (auto& item : {container_name}) {{{loop_body_rewritten}}}"

    return loop_pattern.sub(_loop_repl, body), replacements


def _apply_rule_based_function_transforms(function_source: str) -> tuple[str, list[str]]:
    """Apply deterministic C++ modernization rewrites before calling the LLM."""
    signature, body = _split_function_signature_and_body(function_source)
    if not body:
        return function_source, []

    notes: list[str] = []

    body, malloc_count = _transform_malloc_to_unique_ptr(body)
    if malloc_count:
        notes.append(f"malloc->unique_ptr: {malloc_count}")

    body, loop_count = _transform_index_for_to_range_loop(body)
    if loop_count:
        notes.append(f"index-loop->range-loop: {loop_count}")

    body, char_ptr_count = _transform_char_ptr_literal_declarations(body)
    if char_ptr_count:
        notes.append(f"char*-literal->std::string: {char_ptr_count}")

    body, null_count = _transform_null_macro_to_nullptr(body)
    if null_count:
        notes.append(f"NULL->nullptr: {null_count}")

    first_brace = signature
    transformed = first_brace + body + "}"
    return transformed, notes


def _extract_error_line_numbers(compiler_text: str) -> list[int]:
    """Extract likely source line numbers from compiler stderr/stdout text."""

    line_numbers: set[int] = set()
    for match in re.finditer(r":(\d+):(\d+)?:", compiler_text):
        try:
            line_numbers.add(int(match.group(1)))
        except ValueError:
            continue
    return sorted(line_numbers)


def _get_code_snippet_by_line(code_text: str, line_number: int, radius: int = 2) -> str:
    lines = code_text.splitlines()
    if line_number <= 0 or line_number > len(lines):
        return ""

    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    snippet_lines: list[str] = []
    for idx in range(start, end + 1):
        snippet_lines.append(f"{idx:4d}: {lines[idx - 1]}")
    return "\n".join(snippet_lines)


def _replace_function_by_span(
    source_code: str,
    start_byte: int,
    end_byte: int,
    replacement: str,
) -> str:
    source_bytes = source_code.encode("utf-8")
    prefix = source_bytes[:start_byte]
    suffix = source_bytes[end_byte:]
    replacement_bytes = replacement.encode("utf-8")
    updated_bytes = prefix + replacement_bytes + suffix
    return updated_bytes.decode("utf-8", errors="replace")


def _extract_function_text_from_code(source_code: str, function_name: str) -> str:
    functions_info = _parse_functions_from_source(source_code)
    for function_info in functions_info:
        if str(function_info.get("name") or "") != function_name:
            continue
        start_byte = function_info.get("start_byte")
        end_byte = function_info.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int):
            code_bytes = source_code.encode("utf-8")
            return code_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
    return ""


def _remove_functions_by_name(source_code: str, function_names: set[str]) -> str:
    """Remove full function definitions by name using parser byte spans."""
    if not function_names:
        return source_code

    source_bytes = source_code.encode("utf-8")
    functions_info = _parse_functions_from_source(source_code)
    spans_to_remove: list[tuple[int, int]] = []

    for fn in functions_info:
        fn_name = str(fn.get("name") or "")
        if fn_name not in function_names:
            continue
        start_byte = fn.get("start_byte")
        end_byte = fn.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int) and 0 <= start_byte <= end_byte <= len(source_bytes):
            spans_to_remove.append((start_byte, end_byte))

    if not spans_to_remove:
        return source_code

    spans_to_remove.sort(key=lambda pair: pair[0])
    new_chunks: list[bytes] = []
    cursor = 0
    for start_byte, end_byte in spans_to_remove:
        if start_byte < cursor:
            continue
        if start_byte > cursor:
            new_chunks.append(source_bytes[cursor:start_byte])
        cursor = end_byte

    if cursor < len(source_bytes):
        new_chunks.append(source_bytes[cursor:])

    return b"".join(new_chunks).decode("utf-8", errors="replace")


class IncludeManager:
    """Maps modern C++ feature patterns to the headers they require and injects
    missing ``#include`` directives into source files.

    Usage::

        manager = IncludeManager()
        updated_file = manager.update_file_includes(full_file_content, new_function_code)
    """

    # (regex_pattern, header) — each pattern is tested independently so one
    # header can be triggered by multiple patterns (e.g. both std::variant and
    # std::visit map to <variant>).
    _FEATURE_HEADERS: list[tuple[str, str]] = [
        # C++23
        (r"\bstd::print\b",                "<print>"),
        (r"\bstd::println\b",              "<print>"),
        # C++20 — formatting
        (r"\bstd::format\b",               "<format>"),
        (r"\bstd::vformat\b",              "<format>"),
        # C++20 — ranges / views
        (r"\bstd::ranges::",               "<ranges>"),
        (r"\bstd::views::",                "<ranges>"),
        # C++20 — span
        (r"\bstd::span\b",                 "<span>"),
        # C++20 — bit operations
        (r"\bstd::popcount\b",             "<bit>"),
        (r"\bstd::countl_zero\b",          "<bit>"),
        (r"\bstd::bit_cast\b",             "<bit>"),
        # C++20 — math constants
        (r"\bstd::numbers::",              "<numbers>"),
        # C++20 — concepts
        (r"\bstd::integral\b",             "<concepts>"),
        (r"\bstd::floating_point\b",       "<concepts>"),
        (r"\bstd::same_as\b",              "<concepts>"),
        (r"\bstd::convertible_to\b",       "<concepts>"),
        (r"\bstd::invocable\b",            "<concepts>"),
        # C++17 — string_view, optional, variant, any, filesystem, execution
        (r"\bstd::string_view\b",          "<string_view>"),
        (r"\bstd::optional\b",             "<optional>"),
        (r"\bstd::variant\b",              "<variant>"),
        (r"\bstd::visit\b",                "<variant>"),
        (r"\bstd::any\b",                  "<any>"),
        (r"\bstd::filesystem::",           "<filesystem>"),
        (r"\bstd::execution::",            "<execution>"),
        # Smart pointers / memory management
        (r"\bstd::unique_ptr\b",           "<memory>"),
        (r"\bstd::shared_ptr\b",           "<memory>"),
        (r"\bstd::weak_ptr\b",             "<memory>"),
        (r"\bstd::make_unique\b",          "<memory>"),
        (r"\bstd::make_shared\b",          "<memory>"),
        # Containers
        (r"\bstd::vector\b",               "<vector>"),
        (r"\bstd::array\b",                "<array>"),
        (r"\bstd::map\b",                  "<map>"),
        (r"\bstd::unordered_map\b",        "<unordered_map>"),
        (r"\bstd::set\b",                  "<set>"),
        (r"\bstd::unordered_set\b",        "<unordered_set>"),
        (r"\bstd::deque\b",                "<deque>"),
        (r"\bstd::list\b",                 "<list>"),
        (r"\bstd::forward_list\b",         "<forward_list>"),
        (r"\bstd::stack\b",                "<stack>"),
        (r"\bstd::queue\b",                "<queue>"),
        (r"\bstd::priority_queue\b",       "<queue>"),
        # Strings / streams
        (r"\bstd::string\b",               "<string>"),
        (r"\bstd::stringstream\b",         "<sstream>"),
        (r"\bstd::ostringstream\b",        "<sstream>"),
        (r"\bstd::istringstream\b",        "<sstream>"),
        # I/O
        (r"\bstd::cout\b",                 "<iostream>"),
        (r"\bstd::cin\b",                  "<iostream>"),
        (r"\bstd::cerr\b",                 "<iostream>"),
        # Algorithms
        (r"\bstd::sort\b",                 "<algorithm>"),
        (r"\bstd::find\b",                 "<algorithm>"),
        (r"\bstd::transform\b",            "<algorithm>"),
        (r"\bstd::for_each\b",             "<algorithm>"),
        (r"\bstd::copy\b",                 "<algorithm>"),
        (r"\bstd::fill\b",                 "<algorithm>"),
        (r"\bstd::count_if\b",             "<algorithm>"),
        # Numeric
        (r"\bstd::accumulate\b",           "<numeric>"),
        (r"\bstd::iota\b",                 "<numeric>"),
        (r"\bstd::reduce\b",               "<numeric>"),
        # Utility
        (r"\bstd::move\b",                 "<utility>"),
        (r"\bstd::forward\b",              "<utility>"),
        (r"\bstd::pair\b",                 "<utility>"),
        (r"\bstd::swap\b",                 "<utility>"),
        (r"\bstd::exchange\b",             "<utility>"),
        # Tuple
        (r"\bstd::tuple\b",                "<tuple>"),
        (r"\bstd::tie\b",                  "<tuple>"),
        # Concurrency
        (r"\bstd::thread\b",               "<thread>"),
        (r"\bstd::mutex\b",                "<mutex>"),
        (r"\bstd::lock_guard\b",           "<mutex>"),
        (r"\bstd::unique_lock\b",          "<mutex>"),
        (r"\bstd::condition_variable\b",   "<condition_variable>"),
        (r"\bstd::atomic\b",               "<atomic>"),
        (r"\bstd::chrono::",               "<chrono>"),
        # Functional
        (r"\bstd::function\b",             "<functional>"),
        (r"\bstd::bind\b",                 "<functional>"),
        # Exceptions
        (r"\bstd::exception\b",            "<exception>"),
        (r"\bstd::runtime_error\b",        "<stdexcept>"),
        (r"\bstd::logic_error\b",          "<stdexcept>"),
        (r"\bstd::out_of_range\b",         "<stdexcept>"),
        (r"\bstd::invalid_argument\b",     "<stdexcept>"),
        # Type traits
        (r"\bstd::is_same\b",              "<type_traits>"),
        (r"\bstd::is_same_v\b",            "<type_traits>"),
        (r"\bstd::enable_if\b",            "<type_traits>"),
        (r"\bstd::decay\b",                "<type_traits>"),
        (r"\bstd::decay_t\b",              "<type_traits>"),
        (r"\bstd::remove_reference\b",     "<type_traits>"),
        # Numeric limits
        (r"\bstd::numeric_limits\b",       "<limits>"),
    ]

    def _get_existing_includes(self, file_content: str) -> set[str]:
        """Return the set of header strings already present in file_content."""
        return {
            m.group(1).strip()
            for m in re.finditer(
                r'^\s*#\s*include\s*([<"][^>"]+[>"])',
                file_content,
                re.MULTILINE,
            )
        }

    def required_headers(self, code: str) -> set[str]:
        """Return the set of headers required by feature patterns found in code."""
        needed: set[str] = set()
        for pattern, header in self._FEATURE_HEADERS:
            if re.search(pattern, code):
                needed.add(header)
        return needed

    def update_file_includes(self, file_content: str, modernized_code: str) -> str:
        """Return file_content with missing headers injected after the last #include.

        Scans *modernized_code* for known C++ feature patterns to determine which
        headers are needed.  Any header not already present in *file_content* is
        injected right after the last existing ``#include`` line in alphabetical
        order.  If no ``#include`` exists, the directives are prepended at the top.
        """
        existing = self._get_existing_includes(file_content)
        needed = self.required_headers(modernized_code)
        missing = needed - existing
        if not missing:
            return file_content

        new_directives = sorted(f"#include {h}" for h in missing)
        lines = file_content.splitlines()

        # Locate the last #include within the first 100 lines (safe header-block limit).
        last_include_idx = -1
        scan_limit = min(len(lines), 100)
        for i in range(scan_limit):
            if lines[i].strip().startswith("#include"):
                last_include_idx = i

        if last_include_idx >= 0:
            insert_at = last_include_idx + 1
            result_lines = lines[:insert_at] + new_directives + lines[insert_at:]
        else:
            result_lines = new_directives + lines

        # Preserve trailing newline if the original had one.
        joined = "\n".join(result_lines)
        if file_content.endswith("\n") and not joined.endswith("\n"):
            joined += "\n"
        return joined


class ModernizationState(TypedDict):
    """State object passed through the workflow"""
    code: str  # This field holds either the raw source code text or a path to a source file, depending on how the workflow is invoked.
    language: str  # This field records the language name (for example "cpp"), so nodes can adjust behavior based on the language.
    analysis: str  # This field stores a JSON-formatted string describing analysis results such as discovered functions and graph metrics.
    dependency_map: dict[str, list[str]]  # This field stores a call-graph style dependency map: function name -> list of functions it calls.
    call_graph_data: dict[str, Any]  # This field stores a JSON-serializable view of the call graph (nodes and edges) for visualization and analysis.
    impact_map: dict[str, list[str]]  # This field stores which functions call which (alias of dependency_map for clarity in prompts).
    orphans: list[str]  # This field stores a list of functions that have no callers in the local call graph.
    analysis_report: str  # This field stores a concise human-readable summary of the codebase / translation unit structure.
    modernized_code: str  # This field holds the most recent modernized version of the code produced by the modernizer node.
    verification_result: dict  # This field records the outcome of compiling the modernized code, including success flag and any errors.
    error_log: str  # This field accumulates human-readable error messages that should be fed back into the model for retries.
    attempt_count: int  # This field tracks how many times the modernizer node has been run, which controls the retry loop.
    is_parity_passed: bool  # This field records whether the most recent differential test confirmed functional parity.
    is_functionally_equivalent: bool  # This field records whether the latest tester run declared the modernized code functionally equivalent.
    diff_output: str  # This field stores the unified diff from the tester when outputs differ.
    feedback_loop_count: int  # This field counts how many times we have routed back into the modernizer for self-healing.
    modernization_order: list[str]  # This field records the order in which functions should be modernized.
    modernized_functions: dict[str, str]  # This field tracks which functions have already been modernized.
    current_function_index: int  # This field tracks which function in the modernization_order is currently being modernized.
    partial_success: bool  # This field records whether the workflow finished with only a subset of functions modernized.
    last_working_code: str  # This field stores the last code snapshot that successfully compiled.
    current_target_function: str  # This field stores the function being surgically modernized in the current iteration.
    source_file: str  # Path to the original source file; used to derive the output file path.
    output_file_path: str  # Explicit output path; when non-empty it overrides the auto-derived path.
    legacy_findings: list[dict[str, Any]]  # Static-analysis tags for legacy C/C++ regions needing C++23 overhaul.
    compliance_report: dict[str, Any]  # C++23 compliance score details for the current modernized candidate.
    functions_info: list[dict[str, Any]]  # Cached parser output for the current working code to avoid repeated full re-parsing.
    current_function_name: str  # Name of the function currently being modernized.
    current_function_span: tuple[int, int]  # Byte span (start, end) of the current target function in the current working code.
    project_map: dict[str, Any]  # Cached Tree-sitter semantic map for the current working source.


def analyzer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 1: Analyzer - Graph-aware analysis of legacy code.

    This node:
      - Uses CppParser to extract function definitions and their calls.
      - Builds a DependencyGraph of functions using networkx.
      - Identifies orphan functions (no callers) and circular recursions (cycles).
      - Produces both a machine-friendly JSON analysis and a human-readable analysis_report.
    """
    print("\n🔍 ANALYZER NODE")  # This print helps you see in the console when the analyzer node is running.
    print(f"Language: {state['language']}")  # This print shows which language label the workflow thinks it is processing.
    print(f"Input Code (first 200 chars):\n{state['code'][:200]}...")  # This print gives you a quick peek at the beginning of the input code for context.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20", "c++23"}  # This line checks whether the language label indicates C++, which is when we want to use the Tree-sitter parser.

    functions_info = []  # type: ignore[var-annotated]  # This variable will hold the list of functions discovered by the parser, if any.
    dependency_map: dict[str, list[str]] = {}  # This dictionary will map each function name to the list of functions it calls.
    parser_error: str = ""  # This string will store any error message encountered while trying to parse the code.
    orphans: list[str] = []  # This list will store functions with no callers.
    cycles: list[list[str]] = []  # This list will store any detected circular recursion cycles.
    analysis_report: str = ""  # This string will hold a concise human-readable summary of the code structure.
    call_graph_data: dict[str, Any] = {}  # This dictionary will store a JSON-serializable view of the call graph.
    legacy_findings: list[dict[str, Any]] = []
    project_map: dict[str, Any] = {}

    if is_cpp:  # This condition ensures we only run the C++ parser when we are actually working with C++ code.
        try:  # This try block attempts to run the parser on the current C++ source text and build a dependency graph.
            parser = CppParser()
            project_map = parser.parse_string(state["code"])  # Parse once and reuse this semantic map across nodes.
            parsed_functions = project_map.get("functions", {})
            if isinstance(parsed_functions, dict):
                functions_info = list(parsed_functions.values())
            else:
                functions_info = []
            legacy_findings = detect_legacy_patterns(state["code"])

            dep_graph = DependencyGraph(functions_info)  # This line constructs a DependencyGraph object from the function metadata.
            dependency_map = dep_graph.dependency_map  # This line reads the caller -> callees adjacency list.

            graph_metrics = dep_graph.analyze()  # This line analyzes orphans and cycles in the call graph.
            orphans = list(graph_metrics.get("orphans", []))  # This line records the set of orphan functions.
            cycles = list(graph_metrics.get("cycles", []))  # This line records the list of cycles.

            analysis_report = build_analysis_report(functions_info, dependency_map, orphans, cycles)  # This line builds a concise human-readable report.
            call_graph_data = dep_graph.to_dict()  # This line exports the call graph as simple nodes/edges data for visualization and prompts.
        except Exception as exc:  # This except block catches any parsing- or graph-related errors so they do not crash the workflow.
            parser_error = f"Analyzer failed to parse or analyze C++ file: {exc!r}"  # This line records a human-readable message explaining why analysis failed.

    analysis: dict[str, Any] = {  # This dictionary will hold the structured analysis information that we will turn into JSON.
        "language": state["language"],  # This entry records the language label that was used for this analysis.
        "functions": functions_info,  # This entry stores detailed per-function metadata including names, calls, and byte spans.
        "function_summary": {  # This nested dictionary describes the functions we discovered (if any).
            "count": len(functions_info),  # This entry records how many functions the parser found in the C++ file.
            "names": [fn.get("name", "") for fn in functions_info],  # This list comprehension collects just the function names from the parser output.
        },  # This closing brace ends the function_summary dictionary.
        "dependency_map": dependency_map,  # This entry captures the call-graph style function dependency map for the current translation unit.
        "call_graph_data": call_graph_data,  # This entry stores a simple nodes/edges representation of the call graph.
        "orphans": orphans,  # This entry records functions that have no callers.
        "cycles": cycles,  # This entry records any circular recursion cycles in the call graph.
        "analysis_report": analysis_report,  # This entry stores a concise human-readable description of the code structure.
        "parser_error": parser_error,  # This entry stores any error message from the parser, or an empty string if everything went fine.
        "legacy_findings": legacy_findings,
    }  # This closing brace ends the analysis dictionary.

    state["analysis"] = json.dumps(analysis, indent=2, sort_keys=True)  # This line converts the analysis dictionary into a stable, human-readable JSON string and stores it in the state.
    state["dependency_map"] = dependency_map  # This line stores the raw dependency map separately so downstream nodes can consume it without re-parsing JSON.
    state["call_graph_data"] = call_graph_data  # This line stores the simple call graph data for downstream nodes and visualization.
    state["impact_map"] = dependency_map  # This line aliases the dependency_map as impact_map to emphasize its use in prompts.
    state["orphans"] = orphans  # This line stores the list of orphan functions so downstream nodes can choose to prune them.
    state["analysis_report"] = analysis_report  # This line stores the human-readable report so it can be included in prompts if desired.
    state["legacy_findings"] = legacy_findings
    state["functions_info"] = list(functions_info)
    state["project_map"] = dict(project_map)

    if dependency_map:
        state["modernization_order"] = get_modernization_order(dependency_map)
    else:
        state["modernization_order"] = []
    state["modernized_functions"] = {}
    state["current_function_index"] = 0
    state["current_function_name"] = ""
    state["current_function_span"] = (0, 0)
    state["partial_success"] = False
    print(f"Analysis (JSON):\n{state['analysis']}")  # This print displays the analysis JSON in the console so you can see exactly what the analyzer found.

    return state  # This return passes the updated state object along to the next node in the workflow.


def pruner_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Pruner - Removes orphan code (except main) before modernization.

    This node:
      - Reads the function metadata and orphan list from the deep analyzer.
      - Uses AST byte ranges to surgically remove orphan function definitions from the
        source text, except for the entry point 'main'.
      - Updates state["code"] with a pruned version to reduce the LLM's workload.
    """
    print("\n✂️  PRUNER NODE")  # This print marks the start of the pruning step in the console.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20", "c++23"}  # This line checks whether we are working with C++ code.
    if not is_cpp:  # This condition skips pruning for non-C++ languages.
        print("Pruner: skipping (non-C++ language).")
        return state  # This return leaves the state unchanged.

    if not state.get("analysis"):
        try:
            pre_functions = _parse_functions_from_source(state["code"])
            pre_graph = DependencyGraph(pre_functions)
            pre_orphans = list(pre_graph.analyze().get("orphans", []))
            pre_analysis = {
                "functions": pre_functions,
                "orphans": pre_orphans,
            }
            state["analysis"] = json.dumps(pre_analysis)
            state["orphans"] = pre_orphans
        except Exception:
            print("Pruner: no analysis found and pre-analysis failed, skipping pruning.")
            return state

    try:
        analysis_obj = json.loads(state["analysis"])  # This line parses the JSON analysis string into a Python dictionary.
    except json.JSONDecodeError:
        print("Pruner: failed to parse analysis JSON, skipping pruning.")  # This print warns that analysis could not be decoded.
        return state  # This return leaves the state unchanged.

    functions_info = state.get("functions_info") or analysis_obj.get("functions") or []  # This line reads cached function metadata first.
    orphans = state.get("orphans") or analysis_obj.get("orphans") or []  # This line prefers the orphans recorded on the state but falls back to the analysis JSON.
    if not functions_info or not orphans:  # This condition checks whether there is anything to prune.
        print("Pruner: no functions or no orphans detected, skipping pruning.")  # This print explains why no pruning will happen.
        return state  # This return leaves the state unchanged.

    # We never prune the program entry point 'main', even if it has no callers.
    orphans_to_prune = {str(name) for name in orphans if str(name) != "main"}  # This set records orphan function names that are eligible for pruning.
    if not orphans_to_prune:
        print("Pruner: orphan list only contains 'main' or is empty, nothing to prune.")  # This print explains that we intentionally keep main.
        return state  # This return leaves the state unchanged.

    original_code = state["code"]  # This line reads the original source code text from the state.
    original_bytes = original_code.encode("utf-8")  # This line encodes the code as UTF-8 bytes so we can use Tree-sitter byte offsets.

    # Collect byte-span ranges for every function definition that should be removed.
    spans_to_remove: list[tuple[int, int]] = []  # This list will hold (start_byte, end_byte) pairs for orphan functions.
    for fn in functions_info:  # This loop inspects each function metadata entry.
        name = str(fn.get("name") or "")  # This line normalizes the function name.
        if name not in orphans_to_prune:  # This condition ensures we only consider orphan functions for pruning.
            continue
        start_byte = fn.get("start_byte")
        end_byte = fn.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int) and 0 <= start_byte <= end_byte <= len(original_bytes):  # This condition validates the byte span.
            spans_to_remove.append((start_byte, end_byte))  # This line records the valid span.

    if not spans_to_remove:  # This condition checks whether we have any valid spans to delete.
        print("Pruner: no valid byte spans found for orphan functions, skipping pruning.")  # This print explains why pruning will not proceed.
        return state  # This return leaves the state unchanged.

    spans_to_remove.sort(key=lambda pair: pair[0])  # This line sorts the spans by starting offset so we can rebuild the code in order.

    # Rebuild the code bytes, skipping over the regions marked for removal.
    new_chunks: list[bytes] = []  # This list will hold the surviving segments of the original code.
    cursor = 0  # This integer tracks our current position in the original byte array.
    for start_byte, end_byte in spans_to_remove:  # This loop processes each span to remove.
        if start_byte < cursor:  # This condition skips spans that overlap with or are fully contained in a previously removed region.
            continue
        if start_byte > cursor:  # This condition preserves the code between the previous cursor and the start of this span.
            new_chunks.append(original_bytes[cursor:start_byte])
        cursor = end_byte  # This line moves the cursor past the removed span.

    if cursor < len(original_bytes):  # This condition ensures we keep any trailing code after the last removed span.
        new_chunks.append(original_bytes[cursor:])

    pruned_bytes = b"".join(new_chunks)  # This line concatenates all preserved segments into a single byte string.
    pruned_code = pruned_bytes.decode("utf-8", errors="replace")  # This line decodes the pruned bytes back into text.

    def _prune_dead_includes(code_text: str, removed_functions: list[str]) -> str:
        lines = code_text.splitlines()
        kept_lines: list[str] = []
        removed_lower = [name.lower() for name in removed_functions]
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#include"):
                lowered = stripped.lower()
                if any(name in lowered for name in removed_lower):
                    continue
            kept_lines.append(line)
        return "\n".join(kept_lines)

    pruned_code = _prune_dead_includes(pruned_code, sorted(orphans_to_prune))

    state["code"] = pruned_code  # This line updates the state with the pruned source code so the modernizer works on a smaller, focused input.
    state["last_working_code"] = pruned_code  # Keep subsequent passes anchored to the pruned baseline so removed orphans do not come back.
    state["functions_info"] = _parse_functions_from_source(pruned_code)

    # Remove pruned functions from the modernization order so the modernizer
    # does not attempt to re-add them.
    existing_order = state.get("modernization_order") or []
    state["modernization_order"] = [
        name for name in existing_order if name not in orphans_to_prune
    ]

    print(f"✂️  Pruning {len(spans_to_remove)} orphan function(s): {', '.join(sorted(orphans_to_prune))}")  # This print summarizes what was removed in a visually distinct way.

    return state  # This return passes the updated state object along to the next node in the workflow.


def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Modernizer - Rewrites code to modern standards
    Receives error_log if coming from verifier feedback
    """
    print("\n✏️  MODERNIZER NODE")  # This print marks the start of a modernization attempt in the console.
    print(f"Attempt: {state['attempt_count']} (per-current-function)")  # This print shows how many retries have been attempted for the current function.

    if state["error_log"]:  # This condition checks whether the verifier or tester reported any previous issues.
        print(f"Previous Feedback:\n{state['error_log']}")  # This print shows that feedback so you can see what the model is being asked to fix.

    # Prefer the last known compiling snapshot as our base, then fall back.
    source_to_improve = (
        state.get("last_working_code")
        or state["modernized_code"]
        or state["code"]
    )

    print(f"Source Snapshot (preview, first 200 chars only for logging):\n{source_to_improve[:200]}...")  # This print is only a console preview.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20", "c++23"}  # This line checks whether we are working with C++ so we know whether to ask for C++23 modernization.

    modernized = source_to_improve  # This line initializes the modernized code to the source as a safe default in case the model call fails.
    state["current_target_function"] = ""
    state["current_function_name"] = ""
    state["current_function_span"] = (0, 0)

    if is_cpp:
        functions_info = state.get("functions_info") or []
        if not functions_info:
            functions_info = _parse_functions_from_source(source_to_improve)
            state["functions_info"] = list(functions_info)

        modernization_order = state.get("modernization_order") or [
            str(fn.get("name") or "")
            for fn in functions_info
            if fn.get("name")
        ]
        current_index = int(state.get("current_function_index", 0))

        while current_index < len(modernization_order):
            candidate_name = str(modernization_order[current_index])
            has_candidate = any(str(fn.get("name") or "") == candidate_name for fn in functions_info)
            if has_candidate:
                break
            current_index += 1

        if current_index >= len(modernization_order):
            print("No remaining functions in modernization_order; leaving code unchanged.")
            state["modernized_code"] = source_to_improve
            state["attempt_count"] += 1
            return state

        state["current_function_index"] = current_index
        current_function_name = str(modernization_order[current_index]) if modernization_order else ""
        state["current_target_function"] = current_function_name
        state["current_function_name"] = current_function_name

        current_function_info: dict[str, Any] = {}
        for fn in functions_info:
            if str(fn.get("name") or "") == current_function_name:
                current_function_info = fn
                break

        start_byte = current_function_info.get("start_byte")
        end_byte = current_function_info.get("end_byte")
        source_bytes = source_to_improve.encode("utf-8")
        if not (
            isinstance(start_byte, int)
            and isinstance(end_byte, int)
            and 0 <= start_byte <= end_byte <= len(source_bytes)
        ):
            warning = f"Missing valid byte span for function '{current_function_name}', skipping."
            print(f"⚠️  {warning}")
            state["error_log"] = warning
            state["attempt_count"] += 1
            state["modernized_code"] = source_to_improve
            return state

        state["current_function_span"] = (start_byte, end_byte)
        function_source = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
        transformed_function_source, transform_notes = _apply_rule_based_function_transforms(function_source)

        # Always apply deterministic transformations before LLM so model effort
        # focuses on complex semantic cleanup.
        source_after_rules = _replace_function_by_span(
            source_to_improve,
            start_byte,
            end_byte,
            transformed_function_source,
        )

        if transform_notes:
            print("Rule-based transforms applied: " + ", ".join(transform_notes))
        else:
            print("Rule-based transforms applied: none")

        callers_map = _build_callers_map(state.get("dependency_map", {}))
        callers = sorted(callers_map.get(current_function_name, []))
        callers_display = ", ".join(callers) if callers else "none"
        prompt_parts: list[str] = []
        prompt_parts.append(
            "Rewrite ONLY this function to modern C++23 while preserving behavior. "
            "Return ONLY the updated function."
        )
        prompt_parts.append(f"Function name: {current_function_name}")
        prompt_parts.append(f"Direct callers: {callers_display}")
        prompt_parts.append("Target function:\n```cpp\n" + transformed_function_source + "\n```")

        if transform_notes:
            prompt_parts.append(
                "A deterministic rule-based pass already handled simple rewrites: "
                + ", ".join(transform_notes)
                + ". Only fix remaining complex logic and correctness issues."
            )

        if state["error_log"]:
            prompt_parts.append(
                "Compiler errors from the previous attempt for this same function:\n"
                f"{state['error_log']}"
            )

        prompt_parts.append(
            "Rules: do not output markdown fences, do not output the whole file, do not change function name/signature incompatibly."
        )

        full_prompt = "\n\n".join(prompt_parts)  # This line joins all prompt sections together with blank lines so the text is readable and well structured.

        try:
            raw_text = call_model(_MODEL_SYSTEM_PROMPT, full_prompt)
            print(f"DEBUG: Gemini response (first 200 chars): {raw_text[:200]}...")

            # The model is asked to return only one function, and we replace only that span.
            cleaned_candidate = _clean_model_code_block(raw_text).strip()
            if not cleaned_candidate:
                warning = "Model returned empty function output."
                print(f"⚠️  {warning}")
                state["error_log"] = warning
                modernized = source_after_rules
            else:
                modernized = _replace_function_by_span(
                    source_after_rules,
                    start_byte,
                    end_byte,
                    cleaned_candidate,
                )

                # Keep header updates in place, but only derive required includes from
                # the rewritten function content.
                modernized = IncludeManager().update_file_includes(modernized, cleaned_candidate)
        except Exception as exc:
            error_message = f"Gemini call failed in modernizer: {exc!r}"
            print(error_message)
            if state["error_log"]:
                state["error_log"] += f"\n{error_message}"
            else:
                state["error_log"] = error_message
    else:
        # For non-C++ languages, apply a minimal placeholder transformation.
        modernized = source_to_improve.replace("var ", "auto ")

    state["modernized_code"] = modernized  # This line stores the modernized (or minimally transformed) code back into the shared workflow state.
    state["attempt_count"] += 1  # This line increments the attempt counter so the router can track how many modernization passes we have performed.

    print(f"Modernized Code (first 200 chars):\n{modernized[:200]}...")  # This print shows the beginning of the modernized code so you can see what changed.

    return state  # This return passes the updated state on to the verifier node.


def verifier_node(state: ModernizationState) -> ModernizationState:
    """
    Node 3: Verifier - Validates modernized code via MCP compiler tool
    """
    print("\n✅ VERIFIER NODE")  # This print marks the start of the verification step in the console.
    print("Validating modernized code with real g++ compilation...")  # This print explains that we are about to run the actual compiler on the code.

    code_to_verify = state["modernized_code"]  # This line uses only the modernized_code field, because we want to compile exactly what the modernizer produced.

    if not code_to_verify.strip():  # This condition checks if the AI returned no code (empty or whitespace-only); if so, we stop the workflow entirely.
        message = "CRITICAL: Gemini returned no code. Check your model selection, API key, and max token settings."  # This line sets a precise failure message for empty AI output.
        print(message)  # This print makes the critical condition visible in the console so you know why the workflow is stopping.
        verification_result = {  # This dictionary records a failure result so the workflow can finish without calling the compiler.
            "success": False,  # This field marks the verification as a failure because there was nothing to compile.
            "errors": [message],  # This list stores the same message for the verification result.
            "warnings": [],  # This empty list indicates that there were no compiler warnings.
            "compilation_time_ms": 0,  # This field records zero time because we never invoked the compiler.
            "raw_stdout": "",  # This field is empty because the compiler never ran.
            "raw_stderr": "",  # This field is empty for the same reason.
        }  # This closing brace ends the verification_result dictionary for the empty-code case.
        state["verification_result"] = verification_result  # This line stores the failure result in the shared state.
        state["error_log"] = message  # This line copies the exact error message into the error log for visibility in later inspection.
        state["attempt_count"] = 3  # This line sets attempts to max retry budget so the router safely terminates.
        print(f"❌ Verification FAILED: {message}")  # This print reinforces the failure message in the console.
        return state  # This return exits early so we do not try to call g++ with empty input.

    verification_result = compile_cpp_source(code_to_verify)
    state["verification_result"] = verification_result

    if verification_result["success"]:  # This condition checks whether compilation finished successfully without errors.
        print("✅ Verification PASSED")  # This print confirms in the console that the modernized code compiled successfully.
        state["error_log"] = ""  # This line clears the error log because there are no compiler errors to feed back into the model.
        state["last_working_code"] = state["modernized_code"]
        state["compliance_report"] = score_cpp23_compliance(state["modernized_code"])
        compliance_raw = state["compliance_report"].get("percent", 0)
        try:
            compliance_percent = int(float(str(compliance_raw)))
        except (TypeError, ValueError):
            compliance_percent = 0
        print(f"C++23 Compliance Score: {compliance_percent}%")

        # Refresh parser/cache + dependency graph/order after each successful function replacement.
        parser = CppParser()
        refreshed_project_map = parser.parse_string(state["modernized_code"])
        parsed_functions = refreshed_project_map.get("functions", {})
        if isinstance(parsed_functions, dict):
            refreshed_functions = list(parsed_functions.values())
        else:
            refreshed_functions = []
        state["project_map"] = dict(refreshed_project_map)
        state["functions_info"] = refreshed_functions

        refreshed_dep_graph = DependencyGraph(refreshed_functions)
        state["dependency_map"] = refreshed_dep_graph.dependency_map
        state["impact_map"] = refreshed_dep_graph.dependency_map
        state["call_graph_data"] = refreshed_dep_graph.to_dict()
        if state["dependency_map"]:
            state["modernization_order"] = get_modernization_order(state["dependency_map"])

        target_function = state.get("current_target_function") or ""
        if target_function:
            updated_function_code = _extract_function_text_from_code(
                state["modernized_code"],
                target_function,
            )
            if updated_function_code:
                modernized_functions = state.get("modernized_functions") or {}
                modernized_functions[target_function] = updated_function_code
                state["modernized_functions"] = modernized_functions

        completed_functions = set((state.get("modernized_functions") or {}).keys())
        next_index = 0
        updated_order = state.get("modernization_order") or []
        while next_index < len(updated_order) and str(updated_order[next_index]) in completed_functions:
            next_index += 1
        state["current_function_index"] = next_index
    else:  # This branch handles the case where g++ returned at least one error.
        print("❌ Verification FAILED")  # This print reports in the console that compilation failed.
        state["compliance_report"] = {}
        raw_stderr = verification_result.get("raw_stderr", "")
        errors_list = verification_result.get("errors") or []
        combined_error = raw_stderr or "\n".join(errors_list)
        error_lines = _extract_error_line_numbers(combined_error)
        critical_fix_section = ""
        if error_lines:
            first_error_line = error_lines[0]
            focused_snippet = _get_code_snippet_by_line(code_to_verify, first_error_line)
            critical_fix_section = (
                "\n\nCRITICAL FIX REQUIRED: "
                f"The compiler reported an error at Line {first_error_line}. "
                "Look specifically at this logic:\n"
                f"{focused_snippet}"
            )

        state["error_log"] = combined_error + critical_fix_section
        state["modernized_code"] = state.get("last_working_code") or state["code"]
        print(f"Errors from g++:\n{state['error_log']}")  # This print shows the compiler errors so you can see exactly what went wrong.

    return state  # This return passes the updated state (including verification results) back into the LangGraph router.


def tester_node(state: ModernizationState) -> ModernizationState:
    """
    Node 4: Tester - Runs a differential test to check functional parity.

    This node:
      - Only runs when compilation succeeded.
      - Uses run_differential_test to compare outputs of the original and modernized code.
      - Sets is_parity_passed and, on failure, records a unified diff in error_log so
        the modernizer can perform a self-healing pass focused on logic parity.
    """
    print("\n🧪 TESTER NODE")  # This print marks the start of the tester step in the console.

    # Reset parity / equivalence flags and previous diff output before running a new test.
    state["is_parity_passed"] = True  # This line pessimistically assumes parity will pass; we flip it to False on failure.
    state["is_functionally_equivalent"] = True  # This line assumes functional equivalence until the tester proves otherwise.
    state["diff_output"] = ""  # This line clears any previous diff output.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20", "c++23"}  # This line checks whether we are working with C++ code.
    if not is_cpp:
        print("Tester: skipping (non-C++ language).")
        return state  # This return leaves the state unchanged for non-C++ code.

    if not state["verification_result"].get("success"):  # This condition ensures we only run the tester when compilation succeeded.
        print("Tester: skipping because compilation failed.")  # This print clarifies why we do not run the differential test.
        state["is_parity_passed"] = False  # This line records that parity has not been confirmed.
        state["is_functionally_equivalent"] = False
        return state  # This return leaves error_log with compiler errors.

    if not state["modernized_code"].strip():  # This condition checks that we have modernized code to test.
        print("Tester: skipping because modernized_code is empty.")  # This print explains why the tester is being skipped.
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state

    # Differential testing compares the current final candidate against the original
    # source file path stored in state.
    original_cpp_path = state.get("source_file") or ""

    if not os.path.isfile(original_cpp_path):
        print(f"Tester: original C++ file not found at {original_cpp_path}, skipping parity test.")  # This print warns that we cannot run the differential test.
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state

    print("🧪 Running differential test for functional parity...")  # This print signals that we are about to run the parity test.
    parity_result = run_differential_test(original_cpp_path, state["modernized_code"])
    parity_ok = bool(parity_result.get("parity_ok"))
    diff_text = parity_result.get("diff_text", "")

    state["is_parity_passed"] = parity_ok  # This line records whether the latest parity test passed.
    state["is_functionally_equivalent"] = parity_ok  # This line mirrors the parity result into the functionally-equivalent flag.

    if parity_ok:
        print("✅ Parity Test PASSED (outputs match).")  # This print confirms that the modernized code matches the original program output.
        return state

    # On parity failure, store the unified diff so the modernizer can fix logic.
    print("❌ Parity Test FAILED (outputs differ).")  # This print reports that the outputs do not match.
    state["diff_output"] = diff_text  # This line stores the unified diff for the modernizer prompt.
    state["error_log"] = diff_text  # This line mirrors the diff into error_log for backward-compatible prompts and logging.

    return state  # This return passes the updated state (including parity info) back into the LangGraph router.


def verify_node(state: ModernizationState) -> ModernizationState:
    """Combined verification stage: compile, then run behavioral differential test."""
    compiled_state = verifier_node(state)
    if not compiled_state.get("verification_result", {}).get("success"):
        return compiled_state

    # Run parity once after all functions are modernized.
    current_index = int(compiled_state.get("current_function_index", 0))
    modernization_order = compiled_state.get("modernization_order") or []
    if current_index < len(modernization_order):
        compiled_state["is_parity_passed"] = True
        compiled_state["is_functionally_equivalent"] = True
        compiled_state["diff_output"] = ""
        print("Tester: deferred until final function is modernized.")
        return compiled_state

    return tester_node(compiled_state)


def surgical_router(state: ModernizationState) -> str:
    """
    Router function: Decides whether to retry modernization or proceed to END.

    After a function is successfully modernized (compilation + parity), checks
    whether more functions remain in the modernization order.  If so, routes
    back to the modernizer node to process the next function; otherwise ends.
    """
    verification_success = bool(state["verification_result"].get("success"))
    parity_passed = bool(state.get("is_parity_passed", False))
    attempt_count = int(state.get("attempt_count", 0))

    if parity_passed and verification_success:
        # Check if more functions remain to be modernized.
        current_index = int(state.get("current_function_index", 0))
        modernization_order = state.get("modernization_order") or []
        if current_index < len(modernization_order):
            print(f"\n🔄 Routing back to MODERNIZER (next function: {modernization_order[current_index]})")
            # Reset attempt counter and error log for the next function.
            state["attempt_count"] = 0
            state["error_log"] = ""
            return "modernizer"
        print("\n🏁 Routing to END (SUCCESS: all functions modernized)")
        return "end"

    if attempt_count >= 3:
        # Current function exhausted retries — skip it and try the next one.
        current_index = int(state.get("current_function_index", 0))
        modernization_order = state.get("modernization_order") or []
        state["current_function_index"] = current_index + 1
        if current_index + 1 < len(modernization_order):
            state["partial_success"] = True
            state["attempt_count"] = 0
            state["error_log"] = ""
            print(f"\n🔄 Skipping failed function, routing to MODERNIZER (next: {modernization_order[current_index + 1]})")
            return "modernizer"
        state["partial_success"] = True
        print("\n🏁 Routing to END (PARTIAL_SUCCESS after max attempts)")
        return "end"

    if attempt_count < 3 and not verification_success:
        print("\n🔄 Routing back to MODERNIZER (compiler failure, surgical retry)")
        return "modernizer"

    if attempt_count < 3 and not parity_passed:
        print("\n🔄 Routing back to MODERNIZER (parity failure, surgical retry)")
        return "modernizer"

    print("\n🏁 Routing to END")
    return "end"


def build_workflow():
    """
    Builds and returns the LangGraph workflow
    """
    workflow = StateGraph(ModernizationState)
    
    # Add nodes
    workflow.add_node("prune", pruner_node)
    workflow.add_node("analyze", analyzer_node)
    workflow.add_node("transform", modernizer_node)
    workflow.add_node("verify", verify_node)
    
    # Define edges for the graph-first, self-healing pipeline
    workflow.add_edge("prune", "analyze")
    workflow.add_edge("analyze", "transform")
    workflow.add_edge("transform", "verify")
    
    # Conditional edge from tester based on compilation and parity results
    workflow.add_conditional_edges(
        "verify",
        surgical_router,
        {
            "modernizer": "transform",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("prune")
    
    return workflow.compile()


def run_modernization_workflow(code: str, language: str = "c++23", source_file: str = "", output_file_path: str = ""):
    """
    Executes the modernization workflow with provided code
    """
    print("=" * 60)
    print("🚀 STARTING CODE MODERNIZATION WORKFLOW")
    print("=" * 60)

    is_cpp = language.lower() in {"cpp", "c++", "c++20", "c++23"}

    # Primary path: function-level modernization pipeline.
    if is_cpp and source_file:
        source_abs = os.path.abspath(source_file)
        if output_file_path:
            target_cpp_file = os.path.abspath(output_file_path)
        else:
            base, _ext = os.path.splitext(source_abs)
            target_cpp_file = f"{base}_modernized.cpp"

        target_dir = os.path.dirname(target_cpp_file)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        with open(target_cpp_file, "w", encoding="utf-8") as out_file:
            out_file.write(code)

        _MODEL_BRIDGE.start_modernization_trace(
            input_payload={
                "operation": "function_level_workflow_run",
                "language": language,
                "source_file": source_abs,
                "target_file": target_cpp_file,
                "source_length": len(code),
            }
        )

        parser = CppParser()
        function_modernizer = FunctionModernizer(
            parser,
            _MODEL_BRIDGE,
            fallback_llm=_FALLBACK_MODEL_BRIDGE,
        )
        report_path = os.path.join(_PROJECT_ROOT, "modernization_report.txt")
        failed_functions: list[tuple[str, str]] = []

        project_map = parser.parse_file(target_cpp_file)
        parsed_functions = project_map.get("functions") or {}
        if isinstance(parsed_functions, dict):
            functions_info = list(parsed_functions.values())
        else:
            functions_info = []

        dependency_graph = DependencyGraph(
            functions_info=functions_info,
            types_info=project_map.get("types") or [],
        )
        levels = dependency_graph.get_dependency_levels()

        name_to_fqns: dict[str, list[str]] = {}
        if isinstance(parsed_functions, dict):
            for fqn, meta in parsed_functions.items():
                simple_name = str(meta.get("name") or "")
                if simple_name:
                    name_to_fqns.setdefault(simple_name, []).append(str(fqn))

        def _write_modernization_report(final_code: str) -> None:
            final_score = score_cpp23_compliance(final_code)
            percent = int(final_score.get("percent", 0) or 0)
            stats = function_modernizer.stats
            lines = [
                "## Modernization Report",
                "",
                f"Functions analyzed: {stats.get('functions_analyzed', 0)}",
                f"Functions modernized: {stats.get('functions_modernized', 0)}",
                f"Rule transformations applied: {stats.get('rule_transformations', 0)}",
                f"LLM transformations applied: {stats.get('llm_transformations', 0)}",
                f"Legacy constructs detected: {stats.get('legacy_constructs_detected', 0)}",
                f"Final C++23 compliance score: {percent}%",
            ]
            with open(report_path, "w", encoding="utf-8") as report_file:
                report_file.write("\n".join(lines) + "\n")

        try:
            for level_index, level in enumerate(levels):
                print(f"Modernizing level {level_index + 1}/{len(levels)}")
                print(f"Functions in level: {level}")

                level_fqns: list[str] = []
                for function_name in level:
                    for fqn in sorted(name_to_fqns.get(function_name, [])):
                        level_fqns.append(fqn)

                if not level_fqns:
                    continue

                def _worker(function_fqn: str) -> tuple[str, bool, str]:
                    try:
                        function_modernizer.modernize_function(target_cpp_file, function_fqn)
                        return function_fqn, True, ""
                    except Exception as exc:
                        return function_fqn, False, str(exc)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [
                        executor.submit(
                            _worker,
                            function_fqn,
                        )
                        for function_fqn in level_fqns
                    ]
                    for future in futures:
                        function_fqn, ok, error_message = future.result()
                        if not ok:
                            failed_functions.append((function_fqn, error_message))
                            print(f"Worker failed for {function_fqn}: {error_message}")

            if failed_functions:
                print("Modernization finished with worker failures:")
                for function_fqn, error_message in failed_functions:
                    print(f" - {function_fqn}: {error_message}")

            with open(target_cpp_file, "r", encoding="utf-8") as out_file:
                modernized_code = out_file.read()

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".cpp",
                delete=False,
                encoding="utf-8",
            ) as original_tmp:
                original_tmp.write(code)
                original_cpp_path = original_tmp.name

            try:
                parity_result = run_differential_test(
                    original_cpp_path=original_cpp_path,
                    modernized_code=modernized_code,
                )
            finally:
                if os.path.exists(original_cpp_path):
                    os.remove(original_cpp_path)

            if not bool(parity_result.get("parity_ok")):
                raise RuntimeError(
                    "Final differential test failed after parallel worker modernization.\n"
                    + str(parity_result.get("diff_text") or "")
                )

            similarity_ratio = similarity(code, modernized_code)
            if similarity_ratio > 0.95:
                raise RuntimeError(
                    "Modernization failed: output too similar to input. "
                    f"similarity={similarity_ratio:.2f}"
                )
            verification_result = compile_cpp_source(modernized_code)
            compliance_report = score_cpp23_compliance(modernized_code)
            parity_ok = bool(verification_result.get("success"))
            error_log = ""
            _write_modernization_report(modernized_code)
        except Exception as exc:
            with open(target_cpp_file, "r", encoding="utf-8") as out_file:
                modernized_code = out_file.read()
            compile_snapshot = compile_cpp_source(modernized_code)
            verification_result = {
                "success": False,
                "errors": [str(exc)],
                "warnings": compile_snapshot.get("warnings", []),
                "compilation_time_ms": compile_snapshot.get("compilation_time_ms", 0),
                "raw_stdout": compile_snapshot.get("raw_stdout", ""),
                "raw_stderr": compile_snapshot.get("raw_stderr", ""),
            }
            compliance_report = score_cpp23_compliance(modernized_code)
            parity_ok = False
            error_log = str(exc)
            _write_modernization_report(modernized_code)

        final_state = {
            "code": code,
            "language": language,
            "analysis": "",
            "dependency_map": {},
            "call_graph_data": {},
            "impact_map": {},
            "orphans": [],
            "analysis_report": "",
            "modernized_code": modernized_code,
            "verification_result": verification_result,
            "error_log": error_log,
            "attempt_count": 0,
            "is_parity_passed": parity_ok,
            "is_functionally_equivalent": parity_ok,
            "diff_output": error_log if not parity_ok else "",
            "feedback_loop_count": 0,
            "modernization_order": [],
            "modernized_functions": {},
            "current_function_index": 0,
            "partial_success": False,
            "last_working_code": modernized_code,
            "current_target_function": "",
            "functions_info": [],
            "current_function_name": "",
            "current_function_span": (0, 0),
            "project_map": {},
            "source_file": source_abs,
            "output_file_path": target_cpp_file,
            "legacy_findings": [],
            "compliance_report": compliance_report,
        }

        print("\n" + "=" * 60)
        print("📊 MODERNIZATION COMPLETE")
        print("=" * 60)
        print(f"Language: {final_state['language']}")
        print(f"Verification Success: {final_state['verification_result'].get('success')}")
        print(f"\nFinal Modernized Code:\n{final_state['modernized_code']}")
        print(f"\n💾 Modernized code saved to: {target_cpp_file}")
        print(f"🧾 Modernization report saved to: {report_path}")
        return final_state
    
    # Initialize state
    initial_state = ModernizationState(
        code=code,
        language=language,
        analysis="",
        dependency_map={},
        call_graph_data={},
        impact_map={},
        orphans=[],
        analysis_report="",
        modernized_code="",
        verification_result={},
        error_log="",
        attempt_count=0,
        is_parity_passed=False,
        is_functionally_equivalent=False,
        diff_output="",
        feedback_loop_count=0,
        modernization_order=[],
        modernized_functions={},
        current_function_index=0,
        partial_success=False,
        last_working_code=code,
        current_target_function="",
        functions_info=[],
        current_function_name="",
        current_function_span=(0, 0),
        project_map={},
        source_file=source_file,
        output_file_path=output_file_path,
        legacy_findings=[],
        compliance_report={},
    )
    
    # Build and run the workflow
    _MODEL_BRIDGE.start_modernization_trace(
        input_payload={
            "operation": "workflow_run",
            "language": language,
            "source_file": source_file,
            "source_length": len(code),
        }
    )

    graph = build_workflow()
    final_state = graph.invoke(initial_state)

    if _STRICT_CPP23_MODE:
        compliance_percent = int(final_state.get("compliance_report", {}).get("percent", 0))
        strict_ok = (
            bool(final_state.get("verification_result", {}).get("success"))
            and bool(final_state.get("is_parity_passed", False))
            and compliance_percent >= _STRICT_CPP23_TARGET_PERCENT
        )
        if not strict_ok:
            strict_message = (
                "STRICT MODE FAILURE: final output did not satisfy required C++23 compliance target "
                f"({_STRICT_CPP23_TARGET_PERCENT}%). Final score: {compliance_percent}%."
            )
            final_state["verification_result"] = {
                "success": False,
                "errors": [strict_message],
                "warnings": [],
                "compilation_time_ms": final_state.get("verification_result", {}).get("compilation_time_ms", 0),
                "raw_stdout": final_state.get("verification_result", {}).get("raw_stdout", ""),
                "raw_stderr": final_state.get("verification_result", {}).get("raw_stderr", ""),
            }
            final_state["error_log"] = strict_message
            print(f"❌ {strict_message}")
        else:
            print(
                "✅ STRICT MODE PASSED: "
                f"C++23 compliance {compliance_percent}% >= {_STRICT_CPP23_TARGET_PERCENT}%"
            )
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 MODERNIZATION COMPLETE")
    print("=" * 60)
    print(f"Language: {final_state['language']}")
    print(f"Total Attempts: {final_state['attempt_count']}")
    print(f"Verification Success: {final_state['verification_result'].get('success')}")
    print(f"\nFinal Modernized Code:\n{final_state['modernized_code']}")

    # ------------------------------------------------------------------
    # Save modernized code to a separate output file
    # ------------------------------------------------------------------
    output_path: Optional[str] = final_state.get("output_file_path")
    if not output_path:
        # Derive output path from source_file in state, or fall back to cwd.
        source_path: str = final_state.get("source_file", "") or ""
        if source_path:
            base, _ext = os.path.splitext(source_path)
            output_path = f"{base}_modernized.cpp"
        else:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "output_modernized.cpp",
            )

    try:
        with open(output_path, "w", encoding="utf-8") as _out:
            _out.write(final_state["modernized_code"])
        print(f"\n💾 Modernized code saved to: {output_path}")
    except OSError as _save_err:
        print(f"\n⚠️  Could not save output file: {_save_err}")

    return final_state


if __name__ == "__main__":  # This block runs only when this file is executed as a script (e.g. python -m agents.workflow), not when imported.
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This line gets the project root directory where test.cpp lives.
    _test_cpp_path = os.path.join(_base_dir, "test.cpp")  # This line builds the full path to test.cpp in the project root.

    if not check_model_health():
        exit(1)

    try:  # This try block attempts to read the contents of test.cpp so we can run the full modernization loop on it.
        with open(_test_cpp_path, "r", encoding="utf-8") as f:  # This line opens test.cpp for reading as UTF-8 text.
            cpp_code = f.read()  # This line reads the entire file contents into a string for the workflow.
        print(f"📄 Loaded test.cpp ({len(cpp_code)} characters)")  # This print confirms the file was loaded and shows its size.
    except FileNotFoundError:  # This except block runs if test.cpp does not exist at the expected path.
        print("❌ test.cpp not found at", _test_cpp_path)  # This print tells the user where the script looked for the file.
        exit(1)  
        
    result = run_modernization_workflow(cpp_code, language="c++23", source_file=_test_cpp_path)  # This line invokes the full workflow (prune -> analyze -> transform -> verify with retries) on the loaded code.
    if not bool(result.get("verification_result", {}).get("success")):
        exit(2)