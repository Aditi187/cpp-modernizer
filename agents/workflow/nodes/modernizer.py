import logging
import os
from typing import Dict, Any, List, Tuple

from agents.workflow.state import ModernizationState
from agents.workflow.context import WorkflowContext
from agents.workflow.infra.model_provider import ModelClient
from agents.workflow.infra.code_utils import extract_code
from core.rule_modernizer import RuleModernizer, complexity_score

logger = logging.getLogger(__name__)

_LINE_THRESHOLD = 1000

# ─────────────────────────────────────────────────────────────────────────────
# FIXED SYSTEM PROMPT
# The original was too vague for small 7B models. This explicit, rule-by-rule
# prompt significantly improves output quality without changing the model.
# ─────────────────────────────────────────────────────────────────────────────
_MODERNIZER_SYSTEM = (
    "You are a C++ modernization engine. Your ONLY job is to rewrite legacy C++ to idiomatic C++17.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Output ONLY valid C++ code — no markdown fences, no explanation, no backticks\n"
    "- Preserve EVERY function, class, and method from the input — never truncate\n"
    "- NEVER redeclare classes if modernizing a .cpp implementation file\n\n"
    "MANDATORY TRANSFORMATIONS (apply ALL of these):\n"
    "1. malloc/realloc/free -> std::vector, std::make_unique, std::make_shared\n"
    "2. new T / delete -> std::make_unique<T> (remove delete — RAII handles it)\n"
    "3. char* -> std::string for text/names; char* only at C-API boundaries\n"
    "4. NULL -> nullptr\n"
    "5. typedef X Y -> using Y = X\n"
    "6. #define CONSTANT value -> inline constexpr auto CONSTANT = value\n"
    "7. FILE* + fopen/fclose -> std::ofstream / std::ifstream\n"
    "8. printf/fprintf -> std::cout / stream operator<<\n"
    "9. C-style function pointer callbacks -> std::function<>\n"
    "10. Linked lists (Node* next) -> std::vector (unless O(1) insert required)\n"
    "11. Raw arrays with size variable -> std::vector (remove size variable)\n\n"
    "HEADERS: Add #include only for headers that are actually used. Remove C headers replaced by C++ equivalents.\n"
    "DESTRUCTOR: Remove manual delete/free — destructors of std::unique_ptr/vector handle cleanup automatically.\n"
)

MANDATORY_REQUIREMENTS = """1. LOGIC PARITY: Preserve every single calculation, index, and mathematical operation.
2. RAII: Use smart pointers (unique_ptr/shared_ptr), vectors, and file streams. Replace malloc/free/new/delete.
3. MACROS: Convert functional #define macros into constexpr inline functions.
4. LOGICAL CONST: Use 'mutable' ONLY for synchronization (mutex) or logging (ofstream) in const methods.
5. CLEAN INITIALIZATION: Prefer direct string assignment (data_ = "text") over strcpy/strncpy for literals.
6. NO REDUNDANCY: Eliminate duplicated statements.
7. EFFICIENCY: Use 'std::string_view' for read-only parameters when no conversion to std::string is needed.
8. NO CLASS REDEFINITION: In .cpp files, only define methods using scope resolution (ClassName::method).
9. SIGNATURE PARITY: Method signatures must match header declarations exactly.
10. ALGORITHM TYPE COMPATIBILITY: Lambda parameter types must match the container element type exactly."""


def _should_chunk(source: str) -> bool:
    return source.count("\n") + 1 > _LINE_THRESHOLD


def _build_function_prompt(
    function_body: str,
    plan_desc: str,
    global_context: str,
    developer_feedback: str = "",
    header_code: str = "",
    cpp_standard: str = "c++17",
) -> str:
    feedback_section = f"\nDEVELOPER CRITICAL FEEDBACK:\n{developer_feedback}\n" if developer_feedback else ""
    header_section = (
        f"\nMODERNIZED HEADER DECLARATIONS:\n```cpp\n{header_code}\n```\n"
        if header_code else ""
    )
    std_upper = (cpp_standard or "c++17").upper()
    return (
        f"Modernize this SINGLE C++ function to PERFECT {std_upper} standards.\n"
        f"{feedback_section}"
        f"{header_section}"
        "Global declaration context (classes, structs, headers):\n"
        "```cpp\n"
        f"{global_context}\n"
        "```\n\n"
        "STRATEGIC PLAN:\n"
        f"{plan_desc}\n\n"
        "MANDATORY REQUIREMENTS:\n"
        f"{MANDATORY_REQUIREMENTS}\n"
        "\nFunction to modernize:\n"
        "```cpp\n"
        f"{function_body}\n"
        "```"
    )


def _modernize_chunked(
    state: ModernizationState,
    client: ModelClient,
    rm: RuleModernizer,
    normalized_source: str,
    plan_desc: str,
) -> str:
    project_map: Dict[str, Any] = state.get("project_map", {})
    functions: Dict[str, Any] = project_map.get("functions", {}) if project_map else {}

    if not functions:
        logger.warning("[MODERNIZER] Chunked mode: no functions in project_map; falling back to full-file.")
        return ""

    source_bytes = normalized_source.encode("utf-8")

    boundaries: List[Tuple[int, int, str]] = []
    for f_id, f_meta in functions.items():
        start_byte = f_meta.get("start_byte")
        end_byte = f_meta.get("end_byte")
        if start_byte is not None and end_byte is not None:
            boundaries.append((int(start_byte), int(end_byte), f_id))

    if not boundaries:
        logger.warning("[MODERNIZER] Chunked mode: no boundaries; falling back to full-file.")
        return ""

    boundaries.sort(key=lambda t: t[0])
    logger.info("[MODERNIZER] Chunked mode: %d functions.", len(boundaries))

    global_parts: List[str] = []
    cursor = 0
    for start_byte, end_byte, _ in boundaries:
        if start_byte > cursor:
            global_parts.append(source_bytes[cursor:start_byte].decode("utf-8", errors="replace"))
        cursor = end_byte
    if cursor < len(source_bytes):
        global_parts.append(source_bytes[cursor:].decode("utf-8", errors="replace"))

    global_context = "\n".join(part.strip() for part in global_parts if part.strip())
    _MAX_GLOBAL_CONTEXT_CHARS = 3000
    if len(global_context) > _MAX_GLOBAL_CONTEXT_CHARS:
        global_context = global_context[:_MAX_GLOBAL_CONTEXT_CHARS] + "\n// [context truncated]"

    developer_feedback = state.get("developer_feedback", "")
    header_code = state.get("header_code", "")
    cpp_std = getattr(getattr(state.get("context"), "config", None), "cpp_standard", "c++17")

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    timeout_val = int(os.environ.get("MODERNIZER_CHUNK_TIMEOUT", "300"))

    tasks = []
    with ThreadPoolExecutor(max_workers=min(4, len(boundaries))) as executor:
        for start_byte, end_byte, f_id in boundaries:
            original_function_text = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
            prompt = _build_function_prompt(
                original_function_text, plan_desc, global_context,
                developer_feedback, header_code, cpp_std
            )
            future = executor.submit(client.call, _MODERNIZER_SYSTEM, prompt, role="modernizer")
            tasks.append((start_byte, end_byte, f_id, original_function_text, future))

    result_parts: List[str] = []
    cursor = 0

    for start_byte, end_byte, f_id, original_function_text, future in tasks:
        if start_byte > cursor:
            result_parts.append(source_bytes[cursor:start_byte].decode("utf-8", errors="replace"))
        try:
            raw_output = future.result(timeout=timeout_val)
            if raw_output:
                modernized_function = extract_code(raw_output)
                logger.info("[MODERNIZER] Chunk '%s' modernized.", f_id)
            else:
                logger.warning("[MODERNIZER] LLM empty for chunk '%s'; keeping original.", f_id)
                modernized_function = original_function_text
        except FutureTimeoutError:
            logger.error("[MODERNIZER] Chunk '%s' timed out after %ds; keeping original.", f_id, timeout_val)
            modernized_function = original_function_text
        except Exception as e:
            logger.error("[MODERNIZER] Chunk '%s' failed: %s; keeping original.", f_id, e)
            modernized_function = original_function_text

        result_parts.append(modernized_function)
        cursor = end_byte

    if cursor < len(source_bytes):
        result_parts.append(source_bytes[cursor:].decode("utf-8", errors="replace"))

    return "".join(result_parts)


def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 modernizer (Industrial Grade).
    3-layer pipeline:
      1. Deterministic Normalization (rule-based, safe)
      2. LLM Strategic Modernization (semantic restructuring)
      3. Deterministic Enforcement (mandatory consistency)
    """
    logger.info(">>> [MODERNIZER] Executing Industrial Transformation Pipeline")
    context: WorkflowContext = state.get("context")
    if not context:
        logger.error("[modernizer] missing workflow context")
        return state

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

    # ── LAYER 1: DETERMINISTIC NORMALIZATION ──────────────────────────────
    logger.info("[MODERNIZER] Layer 1: Normalizing with deterministic rules")
    cpp_std = getattr(getattr(context, "config", None), "cpp_standard", "c++17")
    normalized_source, applied_rules, _needs_llm = rm.modernize_with_report(
        source, file_path=state.get("original_file_path"), cpp_standard=cpp_std
    )
    complexity = complexity_score(normalized_source)
    logger.info("[MODERNIZER] Rule pass: %d rules applied, complexity=%d", len(applied_rules), complexity)

    # ── LAYER 2: LLM STRATEGIC MODERNIZATION ─────────────────────────────
    from agents.workflow.infra.model_provider import _get_role_config
    _modernizer_cfg = _get_role_config("modernizer")
    _llm_available = client._use_llm and bool(_modernizer_cfg.api_key)

    if not _needs_llm:
        logger.info("[MODERNIZER] Complexity %d < threshold — LLM skipped (rules handled all).", complexity)
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
        logger.info("[MODERNIZER] Rules-only modernization complete.")
        return state

    logger.info("[MODERNIZER] Layer 2: Invoking LLM (complexity=%d)", complexity)

    if not _llm_available and state.get("feedback_loop_count", 0) >= 1:
        logger.error("[MODERNIZER] LLM unavailable on retry — aborting retry loop.")
        state["modernized_code"] = normalized_source
        state["semantic_ok"] = True
        return state

    transformations = plan.get("transformations", [])
    plan_desc = "\n".join([f"- {t.get('type')}: {t.get('description')}" for t in transformations])
    chunked_used = False
    llm_success = False
    llm_success_before = getattr(context, "llm_calls_succeeded", 0)

    if _should_chunk(normalized_source):
        logger.info("[MODERNIZER] Source >%d lines; chunked processing.", _LINE_THRESHOLD)
        modernized_llm = _modernize_chunked(state, client, rm, normalized_source, plan_desc)
        if modernized_llm:
            chunked_used = True
            llm_success = True
        else:
            logger.info("[MODERNIZER] Chunked fallback: using full-file LLM call.")

    if not chunked_used:
        developer_feedback = state.get("developer_feedback", "")
        feedback_section = f"\nDEVELOPER CRITICAL FEEDBACK:\n{developer_feedback}\n" if developer_feedback else ""
        header_code = state.get("header_code", "")
        header_section = (
            f"\nMODERNIZED HEADER DECLARATIONS:\nUse these declarations to match method signatures. "
            f"Do NOT redefine any class declared here:\n```cpp\n{header_code}\n```\n"
            if header_code else ""
        )
        has_impl = state.get("has_implementation", False)
        header_guard_instruction = ""
        if has_impl:
            header_guard_instruction = (
                "\n======================================================================\n"
                "CRITICAL: THIS IS A HEADER (.h) WITH A CORRESPONDING .cpp FILE!\n"
                "DO NOT DEFINE METHOD BODIES HERE — only declarations (ending with ;).\n"
                "======================================================================\n\n"
            )
        impl_section = (
            "\n13. NO INLINE METHOD IMPLEMENTATIONS: This header has a .cpp file. "
            "Only declare methods here (end with ; not {}).\n"
            if has_impl else ""
        )
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
            "\nSOURCE TO MODERNIZE:\n```cpp\n"
            f"{normalized_source}\n```"
        )

        try:
            raw_output = client.call(_MODERNIZER_SYSTEM, prompt, role="modernizer")
            if raw_output:
                modernized_llm = extract_code(raw_output)
                logger.info("[MODERNIZER] LLM transformation successful.")
                llm_success = True
            else:
                logger.warning("[MODERNIZER] LLM returned empty; using normalized source.")
                modernized_llm = normalized_source
        except Exception as e:
            logger.error("[MODERNIZER] LLM call failed: %s. Using safety fallback.", e)
            modernized_llm = normalized_source

    llm_success_after = getattr(context, "llm_calls_succeeded", 0)
    llm_success = (llm_success_after > llm_success_before) or chunked_used

    # ── LAYER 3: DETERMINISTIC ENFORCEMENT ───────────────────────────────
    logger.info("[MODERNIZER] Layer 3: Enforcing mandatory rules on LLM output")
    final_output, _, _ = rm.modernize_with_report(
        modernized_llm, file_path=state.get("original_file_path"), cpp_standard=cpp_std
    )

    # ── LAYER 3.5: SEMANTIC CONSISTENCY REPAIR ────────────────────────────
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
