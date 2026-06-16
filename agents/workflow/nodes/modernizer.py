import logging
from typing import Dict, Any, List, Tuple

from agents.workflow.state import ModernizationState
from agents.workflow.context import WorkflowContext
from agents.workflow.infra.model_provider import ModelClient
from agents.workflow.infra.code_utils import extract_code
from core.rule_modernizer import RuleModernizer, complexity_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking threshold: files with more than this many lines are processed
# function-by-function to avoid LLM output truncation (max_tokens=4096).
# ---------------------------------------------------------------------------
_LINE_THRESHOLD = 1000

MANDATORY_REQUIREMENTS = """1. LOGIC PARITY: Preserve every single calculation, index, and mathematical operation.
2. RAII: Use smart pointers (unique_ptr/shared_ptr), vectors, and file streams. Replace malloc/free/new/delete. However, for circular associations (where two classes refer to each other), use non-owning raw pointers (e.g., `Class*`) to prevent ownership cycles and memory leaks.
3. CHRONO: Avoid using std::chrono::system_clock::now() stream output directly in C++17 as it is unsupported. Use standard ctime/strftime or custom formatting, or simply avoid logging time if not originally present.
4. MACROS: Convert functional #define macros into constexpr inline functions.
5. LOGICAL CONST: Use 'mutable' ONLY for synchronization (mutex) or logging resources (ofstream) that must be modified in a const method. NEVER use it for general data members.
6. CLEAN INITIALIZATION: Prefer direct string assignment (data_ = "text") over complex replace/strncpy logic for literals.
7. NO REDUNDANCY: Eliminate duplicated statements or redundant function calls that were not in the original code.
8. EFFICIENCY: Use 'std::string_view' for read-only parameters. In C++17, std::string_view does not implicitly convert to std::string. Explicitly construct std::string when passing to functions that expect std::string (e.g., `std::string(sv)`).
9. NO CLASS REDEFINITION: If this is an implementation (.cpp) file, do NOT declare or redefine the classes or structs again. Only define their methods using scope resolution (e.g., `void ClassName::methodName(...)`).
10. SIGNATURE PARITY: Method signatures in this file MUST match the modernized header declarations exactly.
11. CSTRING COMPATIBILITY: Include <cstring> if using standard C string functions like strcpy or strlen, and use std::strcpy/std::strlen.
12. VECTOR CONVERSION: When migrating from raw arrays and dynamic allocation to std::vector, do NOT keep separate size/count member variables or initializers (like count(0)) unless specifically declared in the modernized header. Instead, use vector size (e.g., `.size()`), `.push_back()`, etc. directly to manage the collection.
13. ALGORITHM TYPE COMPATIBILITY: When using standard library algorithms (like `std::remove_if` or `std::find_if`), ensure that the lambda or predicate parameter type matches the element type of the container exactly. For example, if a vector holds pointers (`std::vector<T*>`), the lambda parameter must be `T*` or `const T*`, NOT `const T&` or `T&`."""

MODERNIZER_SYSTEM_PROMPT = """You are a C++ modernization engine. Your ONLY job is to rewrite legacy C++ to idiomatic C++17.

STRICT OUTPUT RULES:
- Output ONLY valid C++ code — no markdown, no explanation, no backticks
- Preserve EVERY function, class, and method from the input
- NEVER truncate or abbreviate code with "// ..." or similar
- NEVER redeclare classes if modernizing a .cpp implementation file

MANDATORY TRANSFORMATIONS (apply ALL of these):
1. malloc/realloc/free -> std::vector, std::make_unique, std::make_shared
2. new T / delete -> std::make_unique<T> / (remove delete — RAII handles it)
3. char* -> std::string (for names/text); keep char* only in C-API boundaries
4. NULL -> nullptr
5. typedef -> using
6. #define CONSTANT -> inline constexpr
7. FILE* + fopen/fclose -> std::ofstream / std::ifstream (RAII)
8. printf/fprintf -> std::cout / std::ofstream operator<<
9. C-style callbacks (void*) -> std::function<>
10. Raw arrays with separate size -> std::vector (remove the size variable)
11. Linked lists (Node* next) -> std::vector (unless order or O(1) insert required)

HEADERS: Add exactly the headers needed. Do not add unused headers.
DESTRUCTOR: Remove manual delete/free — RAII handles cleanup automatically."""

def _should_chunk(source: str) -> bool:
    """Return True if *source* exceeds the line threshold for chunked processing."""
    return source.count("\n") + 1 > _LINE_THRESHOLD


# ---------------------------------------------------------------------------
# Chunked modernization helpers
# ---------------------------------------------------------------------------

def _build_function_prompt(function_body: str, plan_desc: str, global_context: str, developer_feedback: str = "", header_code: str = "", cpp_standard: str = "c++17") -> str:
    """Build a focused LLM prompt for modernizing a single function with global file context."""
    feedback_section = f"\nDEVELOPER CRITICAL FEEDBACK/INSTRUCTIONS:\n{developer_feedback}\n" if developer_feedback else ""
    header_section = f"\nMODERNIZED HEADER DECLARATIONS CONTEXT:\nUse these declarations to match method signatures and data structures. Do NOT redefine any class, struct, or main function declared here:\n```cpp\n{header_code}\n```\n" if header_code else ""
    std_upper = (cpp_standard or "c++17").upper()
    return (
        f"Modernize this SINGLE C++ function to PERFECT {std_upper} standards.\n"
        f"{feedback_section}"
        f"{header_section}"
        "Here is the global declaration context of the file (classes, structs, templates, headers) to guide type and member usage:\n"
        "```cpp\n"
        f"{global_context}\n"
        "```\n\n"
        "STRATEGIC PLAN:\n"
        f"{plan_desc}\n\n"
        "MANDATORY REQUIREMENTS:\n"
        f"{MANDATORY_REQUIREMENTS}\n"
    )


def _modernize_chunked(
    state: ModernizationState,
    client: ModelClient,
    rm: RuleModernizer,
    normalized_source: str,
    plan_desc: str,
) -> str:
    """Process a large file function-by-function and reassemble the result.

    Uses ``project_map`` (set by the analyzer node) to locate function
    boundaries.  Each function is sent to the LLM individually with global context
    so that the response fits within the token budget.

    Falls back to the full-file approach if ``project_map`` is unavailable or
    contains no functions.
    """
    project_map: Dict[str, Any] = state.get("project_map", {})
    functions: Dict[str, Any] = project_map.get("functions", {}) if project_map else {}

    if not functions:
        logger.warning(
            "[MODERNIZER] Chunked mode requested but project_map has no functions; "
            "falling back to full-file LLM call."
        )
        return ""  # empty signals caller to use full-file path

    # Build a sorted list of (start_byte, end_byte, function_id) so we can
    # splice replacements back into the source in order.
    source_bytes = normalized_source.encode("utf-8")

    boundaries: List[Tuple[int, int, str]] = []
    for f_id, f_meta in functions.items():
        start_byte = f_meta.get("start_byte")
        end_byte = f_meta.get("end_byte")
        if start_byte is not None and end_byte is not None:
            boundaries.append((int(start_byte), int(end_byte), f_id))

    if not boundaries:
        logger.warning(
            "[MODERNIZER] No function boundaries found in project_map; "
            "falling back to full-file LLM call."
        )
        return ""

    # Sort by start_byte so we can reconstruct the file sequentially.
    boundaries.sort(key=lambda t: t[0])

    logger.info(
        "[MODERNIZER] Chunked mode: processing %d functions concurrently in parallel.", len(boundaries)
    )

    # Extract all code outside the function scopes to form a global context block
    global_parts: List[str] = []
    cursor = 0
    for start_byte, end_byte, f_id in boundaries:
        if start_byte > cursor:
            global_parts.append(source_bytes[cursor:start_byte].decode("utf-8", errors="replace"))
        cursor = end_byte
    if cursor < len(source_bytes):
        global_parts.append(source_bytes[cursor:].decode("utf-8", errors="replace"))

    global_context = "\n".join(part.strip() for part in global_parts if part.strip())

    # Cap global context to avoid pushing total input over the model's context window.
    # At ~3.5 chars/token, 3000 chars ≈ 850 tokens — safe headroom for the function body.
    _MAX_GLOBAL_CONTEXT_CHARS = 3000
    if len(global_context) > _MAX_GLOBAL_CONTEXT_CHARS:
        global_context = (
            global_context[:_MAX_GLOBAL_CONTEXT_CHARS]
            + "\n// [global context truncated for token budget]"
        )
        logger.debug("[MODERNIZER] global_context truncated to %d chars.", _MAX_GLOBAL_CONTEXT_CHARS)

    developer_feedback = state.get("developer_feedback", "")
    header_code = state.get("header_code", "")

    # Submit tasks concurrently to ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    tasks = []
    with ThreadPoolExecutor(max_workers=min(4, len(boundaries))) as executor:
        for start_byte, end_byte, f_id in boundaries:
            original_function_text = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
            cpp_std = getattr(getattr(state.get("context"), "config", None), "cpp_standard", "c++17")
            prompt = _build_function_prompt(original_function_text, plan_desc, global_context, developer_feedback, header_code, cpp_std)

            future = executor.submit(
                client.call,
                MODERNIZER_SYSTEM_PROMPT,
                prompt,
                role="modernizer",
            )
            tasks.append((start_byte, end_byte, f_id, original_function_text, future))

    # Reconstruct the file sequentially using the completed results
    result_parts: List[str] = []
    cursor = 0  # current byte offset in source_bytes

    for start_byte, end_byte, f_id, original_function_text, future in tasks:
        # Append any text between the previous function and this one.
        if start_byte > cursor:
            result_parts.append(source_bytes[cursor:start_byte].decode("utf-8", errors="replace"))

        try:
            from concurrent.futures import TimeoutError as FutureTimeoutError
            timeout_val = int(os.environ.get("MODERNIZER_CHUNK_TIMEOUT", "1200"))
            raw_output = future.result(timeout=timeout_val)
            if raw_output:
                modernized_function = extract_code(raw_output)
                logger.info("[MODERNIZER] Chunk '%s' modernized successfully.", f_id)
            else:
                logger.warning(
                    "[MODERNIZER] LLM returned empty for chunk '%s'; keeping original.", f_id
                )
                modernized_function = original_function_text
        except FutureTimeoutError:
            logger.error("[MODERNIZER] LLM call timed out for chunk '%s' after %ss. Keeping original.", f_id, timeout_val)
            modernized_function = original_function_text
        except Exception as e:
            logger.error(
                "[MODERNIZER] LLM call failed for chunk '%s': %s. Keeping original.", f_id, e
            )
            modernized_function = original_function_text

        result_parts.append(modernized_function)
        cursor = end_byte

    # Append any trailing text after the last function.
    if cursor < len(source_bytes):
        result_parts.append(source_bytes[cursor:].decode("utf-8", errors="replace"))

    return "".join(result_parts)



# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 modernizer (Industrial Grade).
    Implements a 3-layer pipeline:
    1. Deterministic Normalization (Safe Rules)
    2. LLM Strategic Modernization (Complex Logic)
    3. Deterministic Enforcement (Mandatory Consistency)

    Large files (> _LINE_THRESHOLD lines) are processed function-by-function
    to prevent LLM output truncation.
    """
    logger.info(">>> [MODERNIZER] Executing Industrial Transformation Pipeline")
    context: WorkflowContext = state.get("context")
    if not context:
        logger.error("[modernizer] missing workflow context")
        return state

    # On retries (attempt_count > 0), use the best modernized code so far as the base
    # so the LLM only needs to fix specific issues flagged by semantic_guard/verifier.
    # On the first pass, always use original code.
    attempt = state.get("attempt_count", 0)
    prev_modernized = state.get("modernized_code", "").strip()
    original_code = state.get("code", "")
    if attempt > 0 and prev_modernized and prev_modernized != original_code:
        logger.info("[MODERNIZER] Retry pass %d — starting from previous modernized output.", attempt)
        source = prev_modernized
    else:
        source = original_code

    plan = state.get("modernization_plan", {})
    client = ModelClient(context)
    rm = RuleModernizer()

    # ======================================================
    # LAYER 1: DETERMINISTIC NORMALIZATION
    # ======================================================
    logger.info("[MODERNIZER] Layer 1: Normalizing source code with deterministic rules")
    cpp_std = getattr(getattr(context, "config", None), "cpp_standard", "c++17")
    normalized_source, applied_rules, _needs_llm = rm.modernize_with_report(
        source, file_path=state.get("original_file_path"), cpp_standard=cpp_std
    )
    complexity = complexity_score(normalized_source)
    logger.info("[MODERNIZER] Rule pass complete: %d rules applied, complexity=%d", len(applied_rules), complexity)

    # ======================================================
    # LAYER 2: LLM STRATEGIC MODERNIZATION (skipped when not needed)
    # ======================================================
    from agents.workflow.infra.model_provider import _get_role_config
    _modernizer_cfg = _get_role_config("modernizer")
    _llm_available = client._use_llm and bool(_modernizer_cfg.api_key)

    # ── LLM GATE: skip LLM entirely if rules handled everything ──────────
    if not _needs_llm:
        logger.info(
            "[MODERNIZER] ✓ Complexity score %d < threshold — LLM SKIPPED. "
            "Rules handled all transformations in this file.", complexity
        )
        # Still run consistency repair on rules-only output
        from core.consistency_repair import repair_semantic_consistency
        repaired_source, consistency_repairs = repair_semantic_consistency(normalized_source)
        state["modernized_code"] = repaired_source
        state["semantic_ok"] = True
        state["pipeline_metadata"] = {
            "normalization_applied": normalized_source != source,
            "enforcement_applied": False,
            "chunked_processing": False,
            "llm_skipped": True,
            "llm_skip_reason": "complexity_below_threshold",
            "complexity_score": complexity,
            "rules_applied": applied_rules,
            "consistency_repairs": consistency_repairs,
            "attribution": "deterministic_rules_only",
        }
        logger.info("[MODERNIZER] ✓ Rules-only modernization complete (instant).")
        return state

    logger.info("[MODERNIZER] Layer 2: Invoking LLM for semantic restructuring (complexity=%d)", complexity)

    # --- Guard: abort retry loop early if LLM is known-unavailable ---
    if not _llm_available and state.get("feedback_loop_count", 0) >= 1:
        logger.error(
            "[MODERNIZER] LLM unavailable (no valid API key) and this is a retry — "
            "aborting retry loop. Rule-based output is the best achievable without a real key."
        )
        state["modernized_code"] = normalized_source
        state["semantic_ok"] = True   # prevent infinite retry loop
        return state

    # Extract plan details for the prompt
    transformations = plan.get("transformations", [])
    plan_desc = "\n".join([f"- {t.get('type')}: {t.get('description')}" for t in transformations])
    chunked_used = False
    llm_success = False  # Initialize early to prevent UnboundLocalError
    llm_success_before = getattr(context, "llm_calls_succeeded", 0)

    if _should_chunk(normalized_source):
        logger.info(
            "[MODERNIZER] Source exceeds %d-line threshold; attempting chunked processing.",
            _LINE_THRESHOLD,
        )
        modernized_llm = _modernize_chunked(state, client, rm, normalized_source, plan_desc)
        if modernized_llm:
            chunked_used = True
            llm_success = True
        else:
            logger.info("[MODERNIZER] Chunked fallback: reverting to full-file LLM call.")

    if not chunked_used:
        # --- Original full-file path (unchanged) ---
        developer_feedback = state.get("developer_feedback", "")
        feedback_section = f"\nDEVELOPER CRITICAL FEEDBACK/INSTRUCTIONS:\n{developer_feedback}\n" if developer_feedback else ""
        header_code = state.get("header_code", "")
        header_section = f"\nMODERNIZED HEADER DECLARATIONS CONTEXT:\nUse these declarations to match method signatures and data structures. Do NOT redefine any class, struct, or main function declared here:\n```cpp\n{header_code}\n```\n" if header_code else ""
        has_impl = state.get("has_implementation", False)
        header_guard_instruction = ""
        if has_impl:
            header_guard_instruction = (
                "\n======================================================================\n"
                "CRITICAL WARNING: THIS FILE IS A HEADER (.h) THAT HAS A CORRESPONDING .cpp FILE!\n"
                "YOU MUST NOT DEFINE ANY CLASS METHOD OR CONSTRUCTOR BODIES IN THIS HEADER!\n"
                "ALL METHODS/CONSTRUCTORS MUST ONLY BE DECLARED (ending with a semicolon `;`, NOT braces `{}`).\n"
                "Do NOT include function bodies with braces `{}` inside the class block (except `= default`).\n"
                "======================================================================\n\n"
            )
        impl_section = "\n13. NO INLINE METHOD IMPLEMENTATIONS: Since this header file has a corresponding implementation (.cpp) file, you MUST NOT implement class methods or constructors inline inside this header (except for simple '= default'). Only declare them. Every function/method inside the class must end with a semicolon `;` and have no brace `{}` definition body.\n" if has_impl else ""
        cpp_std_upper = (cpp_std or "c++17").upper()
        prompt = (
            f"{header_guard_instruction}"
            f"Modernize this C++ file to PERFECT {cpp_std_upper} standards.\n"
            f"{feedback_section}"
            f"{header_section}"
            "STRATEGIC PLAN:\n"
            f"{plan_desc}\n\n"
            "MANDATORY REQUIREMENTS:\n"
            f"{MANDATORY_REQUIREMENTS}\n"
            f"{impl_section}"
            "\nReturn ONLY valid C++17 code, no markdown fences, no explanation.\n\n"
            f"SOURCE TO MODERNIZE:\n```cpp\n{normalized_source}\n```"
        )

        try:
            raw_output = client.call(
                MODERNIZER_SYSTEM_PROMPT,
                prompt,
                role="modernizer"
            )
            if raw_output:
                modernized_llm = extract_code(raw_output)
                logger.info("[MODERNIZER] LLM transformation successful.")
                llm_success = True
            else:
                logger.warning("[MODERNIZER] LLM returned empty; falling back to normalized source.")
                modernized_llm = normalized_source
        except Exception as e:
            logger.error(f"[MODERNIZER] LLM call failed: {e}. Using safety fallback.")
            modernized_llm = normalized_source
    llm_success_after = getattr(context, "llm_calls_succeeded", 0)
    # Use counter check, but preserve chunked_used flag as a fallback
    llm_success = (llm_success_after > llm_success_before) or chunked_used

    # ======================================================
    # LAYER 3: DETERMINISTIC ENFORCEMENT
    # ======================================================
    logger.info("[MODERNIZER] Layer 3: Enforcing mandatory rules on LLM output")
    # Overrides any LLM regressions (e.g. if LLM re-introduced NULL or #define)
    final_output, _, _ = rm.modernize_with_report(
        modernized_llm, file_path=state.get("original_file_path"), cpp_standard=cpp_std
    )

    # ======================================================
    # LAYER 3.5: SEMANTIC CONSISTENCY REPAIR
    # ======================================================
    # Fixes cross-cutting type mismatches that arise when independent
    # rules transform types without propagating to surrounding code.
    # e.g. char*→std::string but delete/free/new still reference old type.
    logger.info("[MODERNIZER] Layer 3.5: Repairing cross-cutting semantic inconsistencies")
    from core.consistency_repair import repair_semantic_consistency
    final_output, consistency_repairs = repair_semantic_consistency(final_output)
    if consistency_repairs:
        logger.info("[MODERNIZER] Applied %d consistency repairs", len(consistency_repairs))

    state["modernized_code"] = final_output
    state["pipeline_metadata"] = {
        "normalization_applied": normalized_source != source,
        "enforcement_applied": final_output != modernized_llm,
        "chunked_processing": chunked_used,
        "llm_skipped": not llm_success,
        "complexity_score": complexity,
        "rules_applied": applied_rules,
        "consistency_repairs": consistency_repairs,
        "attribution": f"llm:{_modernizer_cfg.model}" if llm_success else "deterministic_rules_only",
    }

    logger.info(">>> [MODERNIZER] 3-Layer Transformation Complete.")
    return state