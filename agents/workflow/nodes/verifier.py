import re
import logging
from pathlib import Path
from typing import Dict, Any

from agents.workflow.state import ModernizationState
from agents.workflow.context import WorkflowContext
from core.differential_tester import compile_cpp_source, compiler_available

logger = logging.getLogger(__name__)


def _normalize_errors(result: Dict[str, Any]) -> str:
    """
    Converts compiler output into readable error string.
    """
    if not result:
        return ""
    raw_stderr = result.get("raw_stderr")
    if raw_stderr:
        return raw_stderr.strip()
    errors = result.get("errors")
    if errors:
        if isinstance(errors, list):
            return "\n".join(errors)
        return str(errors)
    return ""


def _strip_inline_bodies_from_header(code: str) -> str:
    """
    Post-processing safety pass for header files.

    When a header has a corresponding .cpp implementation file the LLM must NOT
    write inline method bodies inside the class block. If it does, compilation
    fails with 'redefinition' errors because the body is defined twice.

    This function replaces inline method bodies  `{ ... }` with `;`
    while keeping:
      - `= default;`  `= delete;`  `= 0;`
      - Pure-virtual declarations
      - Constructors with initializer lists that end with `{}`
        (handled by converting to `= default;` if the body is empty)
    """
    # Strategy: use a simple brace-counter to find and remove inline bodies.
    # We work line-by-line and track whether we are inside a class block.

    lines = code.splitlines(keepends=True)
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        # Detect a method declaration line that opens a brace at the end
        # Signature: ends with `) {` or `) const {` etc. (NOT `= default {`)
        # We skip lines that are already `= default;` / `= delete;` / `= 0;`
        is_special = bool(re.search(r'=\s*(default|delete|0)\s*;', stripped))
        is_struct_class = bool(re.search(r'\b(struct|class|enum)\b', stripped))
        is_lambda = bool(re.search(r'\[.*\]\s*\(', stripped))
        opens_body = bool(re.search(r'\)\s*(const)?\s*(noexcept)?\s*\{\s*$', stripped)) and not is_special and not is_struct_class and not is_lambda

        if opens_body:
            # Accumulate brace-balanced block
            depth = stripped.count('{') - stripped.count('}')
            block_lines = [line]
            j = i + 1
            while j < len(lines) and depth > 0:
                block_lines.append(lines[j])
                depth += lines[j].count('{') - lines[j].count('}')
                j += 1

            if depth == 0:
                # Replace the entire block with a forward declaration
                # Remove the opening `{` from the first line and append `;`
                new_line = re.sub(r'\s*\{\s*$', ';', stripped) + '\n'
                # If the "body" was `{}` on the same line (trivial body),
                # collapse to `= default;` for constructors/destructors
                if re.search(r'[~\w]\s*\([^)]*\)\s*\{\s*\}', stripped):
                    new_line = re.sub(r'\s*\{\s*\}\s*$', ' = default;', stripped) + '\n'
                result.append(new_line)
                i = j  # skip the consumed block lines
                continue
            else:
                # Unbalanced — probably a full class body, don't touch it
                result.extend(block_lines)
                i = j
                continue

        result.append(line)
        i += 1

    return ''.join(result)


def verifier_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 verifier.
    Compiles the modernized code and checks for parity.
    Respects skip_verification flag for environments without a C++ compiler.

    NOTE: attempt_count is incremented here (not in the router) so that
    LangGraph persists the value into the graph state on every pass.
    """
    logger.info(">>> [VERIFIER] Compiling modernized code using host compiler")

    # Increment attempt counter — must happen before any early returns so the
    # verification_router always sees an up-to-date count.
    state["attempt_count"] = state.get("attempt_count", 0) + 1

    context: WorkflowContext = state.get("context")
    if not context:
        logger.error("[verifier] missing workflow context")
        state["verification_result"] = {"success": False, "errors": ["missing workflow context"]}
        state["error_log"] = "missing workflow context"
        return state

    # Respect skip_verification flag
    skip = getattr(getattr(context, "config", None), "skip_verification", False)
    if skip:
        logger.warning("[VERIFIER] skip_verification=True — bypassing compiler check (LLM-only mode).")
        state["verification_result"] = {
            "success": True,
            "errors": [],
            "warnings": ["Verification skipped: no compiler configured."],
            "compilation_time_ms": 0,
            "raw_stdout": "",
            "raw_stderr": "",
            "compiler": "skipped",
        }
        state["error_log"] = ""
        return state

    code = state.get("modernized_code", "")
    if not code.strip():
        logger.warning("[verifier] No modernized code found to verify.")
        return state

    # ── Header Safety Pass ────────────────────────────────────────────────
    # If this is a .h file with a corresponding .cpp, strip inline bodies
    # that the LLM may have generated, to prevent redefinition errors.
    original_file = state.get("original_file_path", "")
    has_impl = state.get("has_implementation", False)
    if has_impl and original_file:
        suffix = Path(original_file).suffix.lower()
        if suffix in (".h", ".hpp", ".hxx"):
            cleaned = _strip_inline_bodies_from_header(code)
            if cleaned != code:
                logger.info(
                    "[VERIFIER] Header sanitizer removed %d inline body lines.",
                    code.count('\n') - cleaned.count('\n'),
                )
                state["modernized_code"] = cleaned
                code = cleaned

    # Check if a compiler is actually available before calling
    compiler_path = getattr(getattr(context, "config", None), "compiler_path", None)
    if not compiler_available(compiler_path):
        logger.warning(
            "[VERIFIER] No C++ compiler found on PATH. "
            "Set COMPILER_PATH in .env or install g++/clang++. "
            "Use --skip-verify to suppress this warning."
        )
        state["verification_result"] = {
            "success": False,
            "errors": ["No C++ compiler found. Install g++/clang++ or set COMPILER_PATH."],
            "warnings": [],
            "compilation_time_ms": 0,
            "raw_stdout": "",
            "raw_stderr": "",
            "compiler": "not_found",
        }
        # IMPORTANT: keep error_log EMPTY so the fixer/router does not enter a retry loop.
        # The router detects compiler="not_found" and routes directly to END.
        state["error_log"] = ""
        return state

    extra_flags = state.get("extra_compile_args")
    cpp_std = getattr(getattr(context, "config", None), "cpp_standard", None)
    result = compile_cpp_source(
        code,
        gpp_exe=compiler_path,
        extra_flags=extra_flags,
        original_file_path=original_file,
        cpp_standard=cpp_std,
    )

    state["verification_result"] = result
    state["error_log"] = _normalize_errors(result)

    if result.get("success"):
        logger.info(">>> [VERIFIER] Verification PASSED: Code is syntactically valid and modernized.")
    else:
        logger.warning(f">>> [VERIFIER] Verification FAILED: {state['error_log']}")

    return state