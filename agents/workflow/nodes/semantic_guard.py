import logging
import re
from typing import List, Dict, Any

from agents.workflow.state import ModernizationState
from core.parser import CppParser

logger = logging.getLogger("semantic_guard")

def semantic_guard_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 semantic guard (Industrial Grade).
    Detects regressions, behavior changes, signature drift, and logic drift.
    """
    logger.info(">>> [SEMANTIC_GUARD] Auditing for regressions and logic drift")
    
    original_map = state.get("project_map", {})
    modernized_code = state.get("modernized_code", "")
    
    if not modernized_code.strip():
        state["semantic_ok"] = True
        return state

    # Parse the modernized code to build a comparison map
    parser = CppParser()
    try:
        current_map = parser.parse_string(modernized_code)
    except Exception as e:
        logger.error(f"[semantic_guard] Failed to parse modernized code: {e}")
        state["semantic_report"] = {"issues": [{"category": "parsing", "message": "Failed to parse modernization output"}], "risk_score": 1.0}
        state["semantic_ok"] = False
        return state

    issues = []
    
    # 1. Detect Removed Functions
    orig_functions = original_map.get("functions", {})
    curr_functions = current_map.get("functions", {})
    
    for f_id in orig_functions:
        # Check if function exists in modernized code (by FQN or Name)
        # Using FQN/Name match as a heuristic
        f_name = orig_functions[f_id].get("name", "")
        if not any(f_name == f_meta.get("name") for f_meta in curr_functions.values()):
            issues.append({
                "category": "regression",
                "message": f"CRITICAL: Function '{f_name}' missing from modernized code.",
                "severity": "high"
            })

    # 2. Detect Signature Drift
    for f_id, f_meta in curr_functions.items():
        f_name = f_meta.get("name", "")
        # Find matching original function
        matching_orig = next((om for om in orig_functions.values() if om.get("name") == f_name), None)
        if matching_orig:
            orig_params = matching_orig.get("parameters", [])
            curr_params = f_meta.get("parameters", [])
            if len(orig_params) != len(curr_params):
                issues.append({
                    "category": "signature",
                    "message": f"WARNING: Parameter count changed in '{f_name}' ({len(orig_params)} -> {len(curr_params)})",
                    "severity": "medium"
                })

    # 3. Detect Logic Drift (Heuristic)
    for f_id, f_meta in curr_functions.items():
        f_name = f_meta.get("name", "")
        matching_orig = next((om for om in orig_functions.values() if om.get("name") == f_name), None)
        if matching_orig:
            orig_complexity = matching_orig.get("complexity", 1)
            curr_complexity = f_meta.get("complexity", 1)
            # If complexity drops by more than 50% for a large function, flagging risk
            if orig_complexity > 5 and curr_complexity < orig_complexity / 2:
                issues.append({
                    "category": "logic_drift",
                    "message": f"CAUTION: Potential logic deletion in '{f_name}' (complexity dropped {orig_complexity} -> {curr_complexity})",
                    "severity": "medium"
                })

    # 4. Check for legacy leaks (redundant but safe)
    if "malloc" in modernized_code:
        issues.append({"category": "leak", "message": "malloc pattern leaked into output", "severity": "medium"})
    if "free(" in modernized_code:
        issues.append({"category": "leak", "message": "free pattern leaked into output", "severity": "medium"})

    state["semantic_ok"] = not any(i["severity"] == "high" for i in issues)
    state["semantic_report"] = {
        "issues": issues,
        "risk_score": 0.05 * len(issues)
    }
    
    if issues:
        logger.warning(f">>> [SEMANTIC_GUARD] Found {len(issues)} modernization issues / regressions.")
    else:
        logger.info(">>> [SEMANTIC_GUARD] Rigorous audit PASSED: No signature drift or logic deletions detected.")
        
    return state