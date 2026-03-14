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
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=os_module.path.join(_PROJECT_ROOT, ".env"),
    override=True,
)  # Load .env from project root explicitly and override inherited empty values.

from core.parser import extract_functions_from_cpp_file
from core.parser import detect_legacy_patterns
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
from core.inspect_parser import score_cpp23_compliance


_MODEL_SYSTEM_PROMPT = CPP_MODERNIZATION_SYSTEM_PROMPT
_MODEL_BRIDGE = GeminiBridge.from_env(log_fn=print)
_MIN_CPP23_COMPLIANCE_PERCENT = 40


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

    if is_cpp:  # This condition ensures we only run the C++ parser when we are actually working with C++ code.
        code_value = state["code"]  # This line saves the code field to a shorter variable name, which may be either text or a file path.
        temp_file_path: str | None = None  # This variable will keep track of a temporary file path if we need to create one.

        if os.path.isfile(code_value):  # This check handles the case where the caller passed in a real file path instead of raw code text.
            cpp_path = code_value  # This line simply reuses the existing file path when it already points to an on-disk C++ file.
        else:  # This branch handles the case where the caller passed in raw C++ source code as text.
            temp_file = tempfile.NamedTemporaryFile(  # This call creates a new temporary file that will hold the C++ source code long enough for parsing.
                delete=False,  # This argument keeps the file on disk after closing so the parser can open it.
                suffix=".cpp",  # This argument gives the temporary file a .cpp extension so tools can recognize it as C++.
                mode="w",  # This argument opens the file for writing text rather than reading.
                encoding="utf-8",  # This argument ensures the text we write is encoded in UTF-8, matching our parser expectations.
            )  # This closing parenthesis ends the NamedTemporaryFile call.
            temp_file_path = temp_file.name  # This line records the path of the temporary file so we can pass it into the parser and delete it later.
            temp_file.write(code_value)  # This line writes the raw C++ code text into the temporary file so the parser can read it from disk.
            temp_file.flush()  # This line forces the operating system to write any buffered data to disk immediately.
            temp_file.close()  # This line closes the temporary file handle so there are no open descriptors left when the parser runs.
            cpp_path = temp_file_path  # This line sets the parser input path to the newly created temporary C++ file.

        try:  # This try block attempts to run the parser on the chosen C++ file path and build a dependency graph.
            functions_info = extract_functions_from_cpp_file(cpp_path)  # type: ignore[arg-type]  # This call extracts function definitions, calls, and byte spans.
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
        finally:  # This block always runs, whether analysis succeeded or failed.
            if temp_file_path is not None and os.path.exists(temp_file_path):  # This check ensures we only delete the temporary file if we actually created one and it still exists.
                os.remove(temp_file_path)  # This line deletes the temporary file to avoid leaving unnecessary files on disk.

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

    if dependency_map:
        state["modernization_order"] = get_modernization_order(dependency_map)
    else:
        state["modernization_order"] = []
    state["modernized_functions"] = {}
    state["current_function_index"] = 0
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

    functions_info = analysis_obj.get("functions") or []  # This line reads the detailed function metadata list.
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
    print(f"Attempt: {state['attempt_count']}")  # This print shows how many modernization attempts have already been made.

    if state["error_log"]:  # This condition checks whether the verifier or tester reported any previous issues.
        print(f"Previous Feedback:\n{state['error_log']}")  # This print shows that feedback so you can see what the model is being asked to fix.

    # Prefer the last known compiling snapshot as our base, then fall back.
    source_to_improve = (
        state.get("last_working_code")
        or state["modernized_code"]
        or state["code"]
    )

    print(f"Source Code To Modernize (first 200 chars):\n{source_to_improve[:200]}...")  # This print shows the first part of the code that will be sent to the model.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20", "c++23"}  # This line checks whether we are working with C++ so we know whether to ask for C++23 modernization.

    modernized = source_to_improve  # This line initializes the modernized code to the source as a safe default in case the model call fails.
    state["current_target_function"] = ""

    if is_cpp:
        functions_info = _parse_functions_from_source(source_to_improve)
        functions_by_name: dict[str, dict[str, Any]] = {
            str(fn.get("name") or ""): fn for fn in functions_info if fn.get("name")
        }

        modernization_order = state.get("modernization_order") or list(functions_by_name.keys())
        current_index = int(state.get("current_function_index", 0))

        while current_index < len(modernization_order):
            candidate = modernization_order[current_index]
            if candidate in functions_by_name:
                break
            current_index += 1

        if current_index >= len(modernization_order):
            print("No remaining functions to modernize surgically; leaving code unchanged.")
            state["attempt_count"] += 1
            state["modernized_code"] = source_to_improve
            return state

        state["current_function_index"] = current_index
        current_function_name = modernization_order[current_index]
        state["current_target_function"] = current_function_name
        current_function_info = functions_by_name[current_function_name]
        current_function_source = _extract_function_text_from_code(source_to_improve, current_function_name)

        callers_map = _build_callers_map(state.get("dependency_map", {}))
        callers = sorted(callers_map.get(current_function_name, []))
        callers_display = ", ".join(callers) if callers else "none"

        # Build a detailed natural-language prompt that explains the task and enforces strict C++20 rules for the deepseek-coder model.
        prompt_parts: list[str] = []  # This list will collect different sections of the prompt before joining them into a single string.
        prompt_parts.append(
            f"Modernize only the function [{current_function_name}]. "
            f"Keep the signature compatible with its callers: [{callers_display}]."
        )

        signature_text = str(current_function_info.get("signature") or "")
        comments_text = str(current_function_info.get("comments") or "")
        if signature_text:
            prompt_parts.append(f"Current function signature:\n{signature_text}")
        if comments_text:
            prompt_parts.append(f"Current function comments:\n{comments_text}")

        already_modernized = state.get("modernized_functions") or {}
        if already_modernized:
            dependency_context_sections: list[str] = []
            for fn_name in modernization_order[:current_index]:
                updated_fn = already_modernized.get(fn_name)
                if not updated_fn:
                    continue
                dependency_context_sections.append(
                    f"Function: {fn_name}\n{updated_fn.strip()}"
                )
            if dependency_context_sections:
                prompt_parts.append(
                    "Already-modernized dependency context (use these updated APIs/signatures):\n"
                    + "\n\n".join(dependency_context_sections)
                )

        if state["error_log"]:  # This condition checks whether previous attempts produced any feedback that needs to be fixed.
            # Distinguish between compiler failures and parity (logic) failures so we can give targeted instructions.
            if state["verification_result"].get("success") and not state.get("is_functionally_equivalent", True):
                diff_text = state.get("diff_output") or state["error_log"]
                prompt_parts.append(
                    "The code compiled, but the output changed. Fix the logic to ensure functional parity with the original program.\n"
                    "Here is a unified diff between the original and modernized program outputs:\n"
                    f"{diff_text}"
                )
            else:
                prompt_parts.append(
                    "Here are compilation errors from the last attempt. Fix these while modernizing:\n"
                    f"{state['error_log']}"
                )

        if state.get("legacy_findings"):
            legacy_lines = []
            for finding in state.get("legacy_findings", []):
                line = finding.get("line", "?")
                message = finding.get("message", "")
                match = finding.get("match", "")
                legacy_lines.append(f"- line {line}: {match} -> {message}")
            if legacy_lines:
                prompt_parts.append(
                    "Legacy regions tagged for C++23 Overhaul:\n" + "\n".join(legacy_lines)
                )

        prompt_parts.append(
            "Current full C++ file to modernize:\n"
            "```cpp\n"
            f"{source_to_improve}\n"
            "```\n\n"
            f"Orphan functions that MUST NOT exist in final output: {', '.join([name for name in (state.get('orphans') or []) if name != 'main']) or 'none'}.\n"
            "C++23 target priorities (apply when behavior is preserved):\n"
            "1) Error handling: std::expected/std::optional\n"
            "2) I/O and formatting: std::print/std::format\n"
            "3) Data handling: std::span/std::mdspan\n"
            "4) Iteration: std::ranges/views\n"
            "5) Memory: std::unique_ptr and stack allocation\n"
            "You MUST return the ENTIRE updated C++ file content. "
            "Do not return a single function. Do not truncate. Do not use placeholders. "
            "Do not add explanation text outside code. Keep comments minimal and preserve program behavior exactly. "
            "If generation would exceed limits, end at a logical boundary and explicitly include '// CONTINUATION REQUIRED'."
        )

        full_prompt = "\n\n".join(prompt_parts)  # This line joins all prompt sections together with blank lines so the text is readable and well structured.

        try:
            raw_text = call_model(_MODEL_SYSTEM_PROMPT, full_prompt)
            print(f"DEBUG: Gemini response (first 200 chars): {raw_text[:200]}...")

            # The model is asked to return only code, but we still defensively strip out any markdown triple backticks so only raw code remains.
            cleaned_candidate = _clean_model_code_block(raw_text)

            # Enforce full-file responses: reject suspiciously short snippets.
            source_len = len(source_to_improve.strip())
            candidate_len = len(cleaned_candidate.strip())
            min_required = int(source_len * 0.6)
            looks_like_whole_file = (
                candidate_len >= max(200, min_required)
                and "int main(" in cleaned_candidate
            )

            if looks_like_whole_file:
                modernized = cleaned_candidate

                # Re-apply orphan pruning guard to prevent the model from re-introducing dead functions.
                orphan_names = {name for name in (state.get("orphans") or []) if name != "main"}
                if orphan_names:
                    modernized = _remove_functions_by_name(modernized, orphan_names)
            else:
                modernized = source_to_improve
                warning = (
                    "Model response looked truncated or function-only; "
                    "full-file response required."
                )
                print(f"⚠️  {warning}")
                if state["error_log"]:
                    state["error_log"] += f"\n{warning}"
                else:
                    state["error_log"] = warning
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

    # Inject any missing #include directives required by the newly written C++ code.
    if is_cpp:
        modernized = IncludeManager().update_file_includes(modernized, modernized)

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
        state["attempt_count"] = 5  # This line sets attempts to max retry budget so the router safely terminates.
        print(f"❌ Verification FAILED: {message}")  # This print reinforces the failure message in the console.
        return state  # This return exits early so we do not try to call g++ with empty input.

    verification_result = compile_cpp_source(code_to_verify)
    state["verification_result"] = verification_result

    if verification_result["success"]:  # This condition checks whether compilation finished successfully without errors.
        print("✅ Verification PASSED")  # This print confirms in the console that the modernized code compiled successfully.
        state["error_log"] = ""  # This line clears the error log because there are no compiler errors to feed back into the model.
        state["last_working_code"] = state["modernized_code"]
        state["compliance_report"] = score_cpp23_compliance(state["modernized_code"])
        compliance_percent = int(state["compliance_report"].get("percent", 0))
        print(f"C++23 Compliance Score: {compliance_percent}%")

        required_percent = (
            _STRICT_CPP23_TARGET_PERCENT
            if _STRICT_CPP23_MODE
            else _MIN_CPP23_COMPLIANCE_PERCENT
        )
        if compliance_percent < required_percent:
            weak_rules = [
                detail.get("id", "unknown")
                for detail in state["compliance_report"].get("details", [])
                if int(detail.get("score", 0)) < int(detail.get("max_score", 0))
            ]
            mode_label = "strict" if _STRICT_CPP23_MODE else "soft"
            guidance = (
                f"C++23 compliance score below {mode_label} threshold "
                f"({required_percent}%). Missing/weak areas: {', '.join(weak_rules) or 'unknown'}."
            )
            state["error_log"] = guidance
            state["verification_result"] = {
                "success": False,
                "errors": [guidance],
                "warnings": [],
                "compilation_time_ms": verification_result.get("compilation_time_ms", 0),
                "raw_stdout": verification_result.get("raw_stdout", ""),
                "raw_stderr": verification_result.get("raw_stderr", ""),
            }
            print(f"⚠️  Verification downgraded: {guidance}")
            return state

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

        state["current_function_index"] = int(state.get("current_function_index", 0)) + 1
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

    # The differential tester expects a path to the original C++ file. In this workflow,
    # we assume the original is test.cpp at the project root (same as the __main__ block).
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This line gets the project root directory where test.cpp lives.
    original_cpp_path = os.path.join(base_dir, "test.cpp")  # This line builds the full path to test.cpp in the project root.

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

    if attempt_count >= 5:
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

    if attempt_count < 5 and not verification_success:
        print("\n🔄 Routing back to MODERNIZER (compiler failure, surgical retry)")
        return "modernizer"

    if attempt_count < 5 and not parity_passed:
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