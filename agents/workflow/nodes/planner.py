import logging
from typing import Dict, Any, List

from agents.workflow.state import ModernizationState
from agents.workflow.context import WorkflowContext

logger = logging.getLogger(__name__)

def planner_node(state: ModernizationState) -> ModernizationState:
    """
    Phase 4 planner (Industrial Grade).
    Defines a structured modernization strategy based on AST analysis.
    """
    logger.info(">>> [PLANNER] Synthesizing AST-level modernization strategy")
    
    findings = state.get("legacy_findings", [])
    targets = state.get("modernization_targets", [])
    project_map = state.get("project_map", {})
    
    # 1. Map findings to structured transformations
    transformations = []
    
    if "memory_modernization" in targets:
        transformations.append({
            "type": "raii_migration",
            "target": "manual_memory_allocation",
            "confidence": 0.92,
            "priority": "high",
            "description": "Switch malloc/free to std::unique_ptr or std::vector."
        })
        
    if "io_modernization" in targets:
        transformations.append({
            "type": "api_upgrade",
            "target": "c_style_file_io",
            "confidence": 0.88,
            "priority": "medium",
            "description": "Replace printf/fprintf with std::cout or std::print."
        })
        
    if "structural_modernization" in targets:
        transformations.append({
            "type": "type_safety_upgrade",
            "target": "typedefs_and_macros",
            "confidence": 0.95,
            "priority": "low",
            "description": "Convert typedef to using-aliases and #define to constexpr."
        })

    # 2. Risk Assessment
    risk_score = 0.0
    if findings:
        risk_score = min(0.1 * len(findings), 1.0)
        
    # 3. Finalize Plan
    plan = {
        "strategy": "comprehensive_modernization",
        "risk_score": risk_score,
        "transformations": transformations,
        "signature_preservation": True,
        "target_standard": "C++17"
    }
    
    state["modernization_plan"] = plan
    state["plan_summary"] = f"Industrial modernization for {len(transformations)} subsystems detected."
    
    logger.info(f">>> [PLANNER] Plan finalized: {len(transformations)} transformation steps planned. Risk Score: {risk_score:.2f}")
    return state