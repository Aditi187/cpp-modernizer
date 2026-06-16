import logging
import re
import os

from agents.workflow.state import ModernizationState
from core.parser import CppParser

logger = logging.getLogger("semantic_guard")

# Check if semantic guard should be lenient (for small LLM models)
def _is_lenient() -> bool:
    return os.getenv("SEMANTIC_GUARD_LENIENT", "0").strip().lower() in {"1", "true", "yes"}

def semantic_guard_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 semantic guard (Industrial Grade).
    Detects regressions, behavior changes, signature drift, and logic drift.
    
    When SEMANTIC_GUARD_LENIENT=1, allows incomplete modernization (for small LLM models).
    """
    logger.info(">>> [SEMANTIC_GUARD] Auditing for regressions and logic drift")
    
    original_map = state.get("project_map", {})
    modernized_code = state.get("modernized_code", "")
    
    meta = state.get("pipeline_metadata", {}) or {}
    llm_skipped = meta.get("llm_skipped", False) or (meta.get("attribution") == "deterministic_rules_only")

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

    # 4. Check for legacy pattern leaks in the output.
    # When SEMANTIC_GUARD_LENIENT=1, only flag as critical (not medium)
    leak_hints: list = []

    from core.rule_modernizer import _mask_comments_and_strings
    masked_code = _mask_comments_and_strings(modernized_code)

    # Critical structural leaks: malloc/free mean RAII wasn't applied
    if re.search(r'\bmalloc\s*\(', masked_code):
        severity = "medium" if _is_lenient() else ("high" if not llm_skipped else "medium")
        issues.append({
            "category": "leak",
            "message": "CRITICAL: malloc() still present — must be replaced with std::make_unique / std::vector" if severity == "high" else "INFO: malloc() present (acceptable for small models)",
            "severity": severity,
        })
        if severity == "high":
            leak_hints.append("Replace malloc(...) with std::make_unique<T[]>(N) or std::vector<T>(N). Remove sizeof() from the allocation.")

    if re.search(r'\bfree\s*\(', masked_code):
        severity = "medium" if _is_lenient() else ("high" if not llm_skipped else "medium")
        issues.append({
            "category": "leak",
            "message": "CRITICAL: free() still present — must be removed (RAII smart-pointer handles deallocation)" if severity == "high" else "INFO: free() present (acceptable for small models)",
            "severity": severity,
        })
        if severity == "high":
            leak_hints.append("Remove free() calls entirely — std::unique_ptr / std::vector destructor handles memory automatically.")

    # New/delete indicate missed RAII conversion
    if re.search(r'\bnew\s+\w', masked_code) and "make_unique" not in masked_code and "make_shared" not in masked_code:
        severity = "low" if _is_lenient() else "medium"
        issues.append({
            "category": "leak",
            "message": "WARNING: raw 'new' still present without make_unique/make_shared" if not _is_lenient() else "INFO: raw 'new' present (acceptable for small models)",
            "severity": severity,
        })
        if severity == "medium":
            leak_hints.append("Replace 'new T(...)' with 'std::make_unique<T>(...)' and remove matching 'delete'.")

    if re.search(r'\bdelete\s+\w', masked_code):
        severity = "low" if _is_lenient() else "medium"
        issues.append({
            "category": "leak",
            "message": "WARNING: raw 'delete' still present — RAII should handle this" if not _is_lenient() else "INFO: raw 'delete' present (acceptable for small models)",
            "severity": severity,
        })
        if severity == "medium":
            leak_hints.append("Remove 'delete' calls — std::unique_ptr destructor handles deallocation automatically.")

    if re.search(r'\bprintf\s*\(', masked_code):
        severity = "low" if _is_lenient() else "medium"
        issues.append({
            "category": "leak",
            "message": "WARNING: printf() still present — consider std::cout or std::format" if not _is_lenient() else "INFO: printf() present (acceptable for small models)",
            "severity": severity,
        })
        if severity == "medium":
            leak_hints.append("Replace printf() with std::cout or std::format.")

    if re.search(r'\(\s*(int|char|float|double|void)\s*\*\s*\)', masked_code):
        issues.append({
            "category": "leak",
            "message": "WARNING: C-style cast still present",
            "severity": "medium",
        })
        leak_hints.append("Replace C-style casts with static_cast, reinterpret_cast, or dynamic_cast.")

    if re.search(r'\btypedef\b', masked_code):
        issues.append({
            "category": "leak",
            "message": "WARNING: typedef still present",
            "severity": "medium",
        })
        leak_hints.append("Replace typedef with the 'using' keyword.")

    # ── Cross-cutting type mismatch detectors ────────────────────────────
    # These catch bugs where one transformation changed a type but
    # surrounding code still references the old type.

    # Detect: x->member = new std::string(...)  — compile error if member is std::string
    if re.search(r'\w+(?:->|\.)\w+\s*=\s*new\s+std::string\s*\(', masked_code):
        issues.append({
            "category": "type_mismatch",
            "message": "CRITICAL: 'new std::string(...)' assigned to std::string member — type mismatch (std::string* vs std::string)",
            "severity": "high",
        })
        leak_hints.append("Remove 'new' — assign directly: member = value; (std::string is a value type)")

    # Detect: delete s->member where member is std::string — compile error
    # Find std::string members, then check for delete on them
    _string_members = set(m.group(1) for m in re.finditer(r'std::string\s+(\w+)\s*[;={]', masked_code))
    for sm in _string_members:
        if re.search(rf'\bdelete\s+\w+(?:->|\.){re.escape(sm)}\s*;', masked_code):
            issues.append({
                "category": "type_mismatch",
                "message": f"CRITICAL: 'delete ...{sm}' — '{sm}' is std::string (value type), delete is a compile error",
                "severity": "high",
            })
            leak_hints.append(f"Remove 'delete ...{sm}' — std::string auto-destructs, no manual delete needed.")

    # Detect: delete inside range-for over unique_ptr container — double-free
    if 'unique_ptr' in masked_code and re.search(r'\bdelete\s+(?!\[\s*\])(?!this\b)\w+\s*;', masked_code):
        issues.append({
            "category": "type_mismatch",
            "message": "CRITICAL: 'delete' used alongside unique_ptr — causes double-free. unique_ptr handles deallocation.",
            "severity": "high",
        })
        leak_hints.append("Remove all manual 'delete' statements — unique_ptr destructor handles memory automatically.")

    # Detect: sscanf/sprintf with std::string variable (no .c_str())
    for sv in _string_members:
        if re.search(rf'(?:sscanf|scanf)\s*\([^;]*\b{re.escape(sv)}\b(?!\.c_str\(\))[^;]*;', masked_code):
            issues.append({
                "category": "type_mismatch",
                "message": f"CRITICAL: sscanf/scanf called with std::string '{sv}' — needs char buffer or .c_str()",
                "severity": "high",
            })
            leak_hints.append(f"Replace std::string '{sv}' in sscanf with a char buffer, or use std::istringstream.")

    # If any HIGH severity issues found, inject targeted developer_feedback for retry
    has_critical = any(i["severity"] == "high" for i in issues)
    if has_critical and leak_hints:
        feedback = (
            "SECOND PASS — The previous modernization was INCOMPLETE. "
            "The following C legacy patterns MUST be eliminated:\n"
            + "\n".join(f"- {h}" for h in leak_hints)
            + "\n\nDo NOT leave any malloc, free, or unguarded raw pointers in the output."
        )
        existing_feedback = state.get("developer_feedback", "")
        state["developer_feedback"] = (existing_feedback + "\n" + feedback).strip()
        logger.warning("[SEMANTIC_GUARD] Injecting retry feedback for incomplete modernization.")

    state["semantic_ok"] = not has_critical
    state["semantic_report"] = {
        "issues": issues,
        "risk_score": 0.05 * len(issues)
    }

    if issues:
        logger.warning(f">>> [SEMANTIC_GUARD] Found {len(issues)} modernization issues / regressions.")
    else:
        logger.info(">>> [SEMANTIC_GUARD] Rigorous audit PASSED: No signature drift or logic deletions detected.")

    return state