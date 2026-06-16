import logging

from agents.workflow.state import ModernizationState

logger = logging.getLogger(__name__)

def planner_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 planner (Industrial Grade).
    Defines a structured modernization strategy based on AST analysis.
    """
    logger.info(">>> [PLANNER] Synthesizing AST-level modernization strategy")
    
    findings = state.get("legacy_findings", [])
    targets = state.get("modernization_targets", [])
    
    # 1. Enrich Strategy with specific flags
    strategy_flags = {
        "memory_modernization": any("malloc" in f or "free" in f or "pointer" in f for f in findings),
        "container_upgrade": any("vector" not in f and ("malloc" in f or "array" in f.lower()) for f in findings),
        "string_modernization": any("char*" in f or "strcpy" in f or "strcat" in f or "string" in f.lower() for f in findings),
        "chrono_upgrade": any("time" in f or "localtime" in f for f in findings),
        "macro_cleanup": any("macro" in f or "#define" in f for f in findings),
        "type_safety": any("typedef" in f or "cast" in f for f in findings)
    }
    
    # 2. Map findings to structured transformations
    transformations = []
    
    if strategy_flags["memory_modernization"]:
        transformations.append({
            "type": "raii_migration",
            "target": "manual_memory_allocation",
            "confidence": 0.95,
            "priority": "high",
            "description": "Switch malloc/free/new/delete to std::unique_ptr or std::vector."
        })
        
    if strategy_flags["chrono_upgrade"]:
        transformations.append({
            "type": "api_upgrade",
            "target": "c_style_time_api",
            "confidence": 0.90,
            "priority": "medium",
            "description": "Replace std::time_t and localtime with std::chrono::system_clock."
        })
        
    if strategy_flags["macro_cleanup"]:
        transformations.append({
            "type": "preprocessor_cleanup",
            "target": "functional_macros",
            "confidence": 0.85,
            "priority": "medium",
            "description": "Convert functional #define macros into constexpr inline functions."
        })

    if strategy_flags["type_safety"]:
        transformations.append({
            "type": "type_safety_upgrade",
            "target": "typedefs_and_casts",
            "confidence": 0.95,
            "priority": "low",
            "description": "Convert typedef to using-aliases and C-casts to static_cast."
        })

    # 3. Dynamic Risk Assessment
    # Higher variety of findings = higher risk.
    finding_types = len([v for v in strategy_flags.values() if v])
    risk_score = min(0.05 * len(findings) + 0.1 * finding_types, 1.0)
    
    # Complexity Gate: skip LLM for empty findings or only simple patterns
    complex_keywords = ["malloc", "free", "new", "delete", "char*", "pointer", "FILE", "sprintf", "realloc", "va_list"]
    needs_llm = any(any(kw in f for kw in complex_keywords) for f in findings)
    
    if needs_llm:
        from agents.workflow.infra.model_provider import ModelClient
        import json
        client = ModelClient(state.get("context"))

        # Structured, constrained prompt — prevents the LLM from injecting an essay
        # into the modernizer's STRATEGIC PLAN section.
        findings_summary = ", ".join(findings[:20]) if findings else "none detected"
        structured_prompt = (
            f"You are analyzing legacy C++ code. Findings: {findings_summary}\n\n"
            "Respond with ONLY a JSON array of up to 5 transformation actions. "
            "Each item must have exactly two keys: "
            "'action' (one of: replace_malloc, replace_printf, replace_typedef, "
            "replace_null, replace_c_cast, replace_char_ptr, add_raii, "
            "replace_c_header, replace_define, add_noexcept) "
            "and 'reason' (max 80 characters, one sentence). "
            "Example: [{\"action\": \"replace_malloc\", \"reason\": \"3 malloc calls found\"}]\n"
            "JSON array only — no markdown, no explanation, no other text."
        )
        llm_plan_raw = client.call(
            "You are AGENT 4: PLANNER. Output only a JSON array of transformation actions.",
            structured_prompt,
            role="planner",
        )
        if llm_plan_raw:
            try:
                # Strip markdown fences if the LLM wrapped JSON in them
                clean = llm_plan_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                actions = json.loads(clean)
                if isinstance(actions, list):
                    for item in actions[:5]:  # hard cap
                        action = str(item.get("action", ""))[:40]
                        reason = str(item.get("reason", ""))[:80]  # hard cap
                        if action:
                            transformations.append({
                                "type": f"llm_{action}",
                                "target": action,
                                "confidence": 0.90,
                                "priority": "high",
                                "description": reason,
                            })
                    logger.info("[PLANNER] LLM returned %d structured actions.", len(actions[:5]))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("[PLANNER] LLM response was not valid JSON (%s) — skipping LLM plan.", e)
    else:
        logger.info("[PLANNER] Complexity gate passed (only simple patterns). Skipping LLM strategy call.")

        
    # 4. Finalize Plan
    plan = {
        "strategy": "comprehensive_modernization",
        "flags": strategy_flags,
        "risk_score": risk_score,
        "transformations": transformations,
        "signature_preservation": True,
        "target_standard": "C++17"
    }
    
    state["modernization_plan"] = plan
    state["plan_summary"] = f"Industrial modernization for {len(transformations)} subsystems detected (Risk: {risk_score:.2f})."
    
    logger.info(f">>> [PLANNER] Plan finalized with {len(transformations)} transformations. Risk Score: {risk_score:.2f}")
    return state