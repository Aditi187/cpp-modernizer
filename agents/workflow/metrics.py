import logging

from typing import Any, Dict, Optional

logger = logging.getLogger("metrics")

class MetricsCollector:
    def __init__(self, initial_metrics: Optional[Dict[str, Any]] = None):
        self.metrics = initial_metrics or {}

    def add(self, key: str, value: Any):
        self.metrics[key] = value

    def report(self) -> Dict[str, Any]:
        return self.metrics

def calculate_modernization_score(state: Dict[str, Any]) -> float:
    """
    Calculates a professional modernization quality score (0.0 to 1.0).
    Formula: (Patterns Found / max(1, Patterns Found)) * Efficiency Ratio
    Simplified for this industrial engine: based on verification success and pattern reduction.
    """
    metrics = state.get("metrics", {})
    initial_patterns = metrics.get("legacy_pattern_count", 0)
    
    if initial_patterns == 0:
        return 1.0
        
    # Heuristic: 80% for verification success, 20% for semantic guard status
    verification = state.get("verification_result", {})
    v_score = 0.8 if verification.get("success") else 0.2
    s_score = 0.2 if state.get("semantic_ok") else 0.05
    
    return min(0.95, v_score + s_score)

def get_safety_rating(score: float) -> str:
    if score >= 0.9: return "EXCEPTIONAL"
    if score >= 0.7: return "HIGH"
    if score >= 0.5: return "MEDIUM"
    return "LOW (MANUAL REVIEW REQUIRED)"

__all__ = ["MetricsCollector", "calculate_modernization_score", "get_safety_rating"]
