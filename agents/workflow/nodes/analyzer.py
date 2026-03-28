import re
import logging
from typing import Dict, Any, List

from agents.workflow.state import ModernizationState
from core.parser import CppParser

logger = logging.getLogger(__name__)

def analyzer_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 analyzer (Industrial Grade).
    Uses CppParser for deep AST analysis.
    """
    logger.info(">>> [ANALYZER] Starting deep AST-based structural analysis")
    code = state.get("code", "")
    if not code:
        logger.warning("[analyzer] empty source")
        return state

    parser = CppParser()
    try:
        project_map = parser.parse_string(code)
    except Exception as e:
        logger.error(f"[analyzer] AST parsing failed: {e}. Falling back to shallow analysis.")
        project_map = {}

    findings: List[str] = []
    functions_info = project_map.get("functions", {})
    
    # 1. Aggregate patterns from all functions detected by CppParser
    if functions_info:
        for f_id, f_meta in functions_info.items():
            patterns = f_meta.get("legacy_patterns", {})
            if patterns.get("has_malloc"): findings.append(f"malloc in {f_id}")
            if patterns.get("has_free"): findings.append(f"free in {f_id}")
            if patterns.get("has_printf"): findings.append(f"printf in {f_id}")
            if patterns.get("has_raw_pointer"): findings.append(f"unprotected raw pointer in {f_id}")
            if patterns.get("has_null_macro"): findings.append(f"NULL macro in {f_id}")

    # 2. Extract structural findings from global context
    type_defs = project_map.get("type_definitions", {})
    if type_defs:
        for t_name in type_defs:
            findings.append(f"legacy type definition: {t_name}")

    if "FILE*" in code: findings.append("C-style FILE* API usage")
    if "#define" in code: findings.append("Preprocessor macro usage")
    if "typedef" in code: findings.append("typedef detected")
    if "time_t" in code or "localtime" in code: findings.append("legacy C-style time API usage")

    # 3. Target mapping (Strategic)
    targets = []
    if any("malloc" in f or "pointer" in f for f in findings): 
        targets.append("memory_management")
    if any("printf" in f or "FILE" in f for f in findings) or "printf" in code or "strcpy" in code: 
        targets.append("iostream_upgrade")
    if any("typedef" in f or "macro" in f or "type definition" in f for f in findings): 
        targets.append("structural_modernization")
    
    if not targets:
        targets.append("general_modernization")

    # 4. State Update
    state["legacy_findings"] = findings
    state["functions_info"] = list(functions_info.keys())
    state["modernization_targets"] = targets
    state["project_map"] = project_map # store for planner/modernizer

    # 5. Metrics integration
    metrics = state.get("metrics", {})
    metrics["legacy_pattern_count"] = len(findings)
    metrics["function_count"] = len(functions_info)
    metrics["ast_depth"] = 10 # heuristic
    state["metrics"] = metrics

    logger.info(f">>> [ANALYZER] Analysis complete: {len(findings)} legacy patterns found across {len(functions_info)} functions.")
    return state