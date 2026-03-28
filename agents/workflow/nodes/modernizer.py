import re
import logging
from typing import Dict, Any, Optional

from agents.workflow.state import ModernizationState
from agents.workflow.context import WorkflowContext
from agents.workflow.infra.model_provider import ModelClient
from core.rule_modernizer import RuleModernizer

logger = logging.getLogger(__name__)

def extract_code(text: str) -> str:
    """
    Extracts code from markdown fences if present.
    """
    if not text:
        return ""
    # Look for ```cpp ... ``` or ```c++ ... ``` or just ``` ... ```
    match = re.search(r"```(?:cpp|c\+\+)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 modernizer (Industrial Grade).
    Implements a 3-layer pipeline:
    1. Deterministic Normalization (Safe Rules)
    2. LLM Strategic Modernization (Complex Logic)
    3. Deterministic Enforcement (Mandatory Consistency)
    """
    logger.info(">>> [MODERNIZER] Executing Industrial Transformation Pipeline")
    context: WorkflowContext = state.get("context")
    if not context:
        logger.error("[modernizer] missing workflow context")
        return state

    source = state.get("code", "")
    plan = state.get("modernization_plan", {})
    client = ModelClient(context)
    rm = RuleModernizer()

    # ======================================================
    # LAYER 1: DETERMINISTIC NORMALIZATION
    # ======================================================
    logger.info("[MODERNIZER] Layer 1: Normalizing source code with deterministic rules")
    normalized_source = rm.modernize_text(source)

    # ======================================================
    # LAYER 2: LLM STRATEGIC MODERNIZATION
    # ======================================================
    logger.info("[MODERNIZER] Layer 2: Invoking LLM for semantic restructuring")
    
    # Extract plan details for the prompt
    transformations = plan.get("transformations", [])
    plan_desc = "\n".join([f"- {t.get('type')}: {t.get('description')}" for t in transformations])

    prompt = (
        "Modernize this C++ file to PERFECT C++17 standards.\n"
        "STRATEGIC PLAN:\n"
        f"{plan_desc}\n\n"
        "MANDATORY REQUIREMENTS:\n"
        "1. LOGIC PARITY: Preserve every single calculation, index, and mathematical operation (e.g., if code says 'i * 10', the MODERN code must also use 'i * 10').\n"
        "2. RAII: Use smart pointers (unique_ptr/shared_ptr), vectors, and file streams. Replace malloc/free/new/delete.\n"
        "3. LOGICAL CONST: Use 'mutable' for logging members (e.g., mutable std::ofstream) to allow logging from const methods.\n"
        "4. EFFICIENCY: Use 'std::string_view' for read-only string variables.\n"
        "5. THREAD SAFETY: Use 'localtime_s' (Windows style) or 'localtime_r' (POSIX) for time conversion.\n"
        "6. SIGNATURES: Maintain original function names unless the plan specifically targets a rename.\n"
        "Return ONLY valid C++17 code, no markdown fences, no explanation.\n\n"
        f"SOURCE TO MODERNIZE:\n```cpp\n{normalized_source}\n```"
    )

    try:
        raw_output = client.call(
            "You are AGENT 2: MODERNIZER. Convert normalized C++ to idiomatic C++17. Output code only.",
            prompt,
            role="modernizer"
        )
        if raw_output:
            modernized_llm = extract_code(raw_output)
            logger.info("[MODERNIZER] LLM transformation successful.")
        else:
            logger.warning("[MODERNIZER] LLM returned empty; falling back to normalized source.")
            modernized_llm = normalized_source
    except Exception as e:
        logger.error(f"[MODERNIZER] LLM call failed: {e}. Using safety fallback.")
        modernized_llm = normalized_source

    # ======================================================
    # LAYER 3: DETERMINISTIC ENFORCEMENT
    # ======================================================
    logger.info("[MODERNIZER] Layer 3: Enforcing mandatory rules on LLM output")
    # This overrides any LLM regressions (e.g., if LLM re-introduced NULL or #define)
    final_output = rm.modernize_text(modernized_llm)
    
    state["modernized_code"] = final_output
    state["pipeline_metadata"] = {
        "normalization_applied": normalized_source != source,
        "enforcement_applied": final_output != modernized_llm
    }
    
    logger.info(">>> [MODERNIZER] 3-Layer Transformation Complete.")
    return state