import re
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


# Patterns that count as "legacy" (lower count = better modernized)
_LEGACY_PATTERNS = [
    r'\bNULL\b',
    r'\btypedef\s+\S',
    r'\bchar\s*\*',
    r'#\s*include\s*<(stdio|stdlib|string|time|math|assert|errno|float|limits)\.h>',
    r'\bprintf\s*\(',
    r'\bsprintf\s*\(',
    r'\bmalloc\s*\(',
    r'\bfree\s*\(',
    r'\bnew\s+\w',
    r'\bdelete\s',
    r'\bstd::auto_ptr\b',
    r'\bthrow\s*\(\s*\)',
    r'^\s*#\s*define\s+[A-Z][A-Z0-9_]*\s+[0-9]',
    r'\bregister\s+',
    r'\batoi\s*\(',
    r'\batof\s*\(',
    r'\bgets\s*\(',
    r'\bstrcpy\s*\(',
    r'\bstrcat\s*\(',
    r'\bTRUE\b|\bFALSE\b',
    r'\bBOOL\b',
    r'\bwhile\s*\(\s*1\s*\)',
    r'\bexit\s*\(\s*[01]\s*\)',
]

# Patterns that indicate modern C++17 idioms (more = better)
_MODERN_PATTERNS = [
    r'\bnullptr\b',
    r'\busing\s+\w+\s*=',
    r'\bconstexpr\b',
    r'\bnoexcept\b',
    r'\bstd::unique_ptr\b',
    r'\bstd::shared_ptr\b',
    r'\bstd::make_unique\b',
    r'\bstd::make_shared\b',
    r'\bstd::vector\b',
    r'\bstd::string\b',
    r'\bstd::string_view\b',
    r'\bstd::optional\b',
    r'\bauto\s+\w',
    r'for\s*\(auto',
    r'\bstatic_cast\b',
    r'\bstd::stoi\b',
    r'\bstd::getline\b',
    r'\bstd::abs\b',
    r'\bstd::numeric_limits\b',
    r'\bwhile\s*\(\s*true\s*\)',
    r'#include\s*<(memory|vector|string|array|optional|variant|algorithm)>',
]


def _count_legacy_patterns(code: str) -> int:
    if not code:
        return 0
    total = 0
    for pat in _LEGACY_PATTERNS:
        total += len(re.findall(pat, code, re.MULTILINE))
    return total


def _count_modern_patterns(code: str) -> int:
    if not code:
        return 0
    total = 0
    for pat in _MODERN_PATTERNS:
        total += len(re.findall(pat, code, re.MULTILINE))
    return total


def calculate_modernization_score(state: Dict[str, Any]) -> float:
    """
    Calculates a modernization quality score (0.0 to 1.0).

    Weights:
      60% -- legacy pattern reduction ratio
      20% -- modern C++17 idiom adoption (positive signal)
      10% -- verification success
      10% -- semantic guard pass

    Rules-only runs get partial credit for semantic/verification when
    not run, so they are not unfairly penalised for the absent LLM guard.
    """
    original_code  = state.get("code", "")
    modernized_code = state.get("modernized_code", "")

    patterns_before = _count_legacy_patterns(original_code)
    patterns_after  = _count_legacy_patterns(modernized_code)
    modern_after    = _count_modern_patterns(modernized_code)
    modern_before   = _count_modern_patterns(original_code)

    # Legacy reduction score (60%)
    if patterns_before > 0:
        reduction_ratio = 1.0 - (patterns_after / patterns_before)
        legacy_score = max(0.0, reduction_ratio) * 0.60
    else:
        legacy_score = 0.60  # already clean

    # Modern adoption score (20%) - net gain, normalised over 10 idioms
    modern_gain = max(0, modern_after - modern_before)
    modern_score = min(1.0, modern_gain / 10.0) * 0.20

    # Verification (10%)
    verification = state.get("verification_result", {})
    v_status = verification.get("compiler", "skipped")
    if verification.get("success"):
        v_score = 0.10
    elif v_status in ("skipped", None, ""):
        # Skipped is not failure -- partial credit if rules ran well
        v_score = 0.07
    else:
        v_score = 0.0

    # Semantic guard (10%)
    sem_ok = state.get("semantic_ok", None)
    if sem_ok is True:
        s_score = 0.10
    elif sem_ok is None:
        # Not run (rules-only) -- grant partial credit when legacy score is good
        s_score = 0.07 if legacy_score >= 0.35 else 0.0
    else:
        s_score = 0.0

    raw = legacy_score + modern_score + v_score + s_score
    return round(min(1.0, raw), 4)


def get_safety_rating(score: float) -> str:
    """Human-readable safety rating."""
    if score >= 0.65:
        return "SAFE"
    if score >= 0.45:
        return "REVIEW"
    return "UNSAFE"


__all__ = ["MetricsCollector", "calculate_modernization_score", "get_safety_rating"]
