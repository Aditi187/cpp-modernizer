from __future__ import annotations

import re
from typing import List, Tuple


class ModernizationRule:
    def __init__(self, pattern: str, replacement: str, description: str):
        self.pattern = re.compile(pattern)
        self.replacement = replacement
        self.description = description


_RULES: List[ModernizationRule] = [
    ModernizationRule(
        pattern=r"\bNULL\b",
        replacement="nullptr",
        description="NULL replaced with nullptr",
    ),
    ModernizationRule(
        pattern=r"new\s+([A-Za-z_]\w*)",
        replacement=r"std::make_unique<\1>()",
        description="raw new replaced with std::make_unique",
    ),
    ModernizationRule(
        pattern=r"delete\s+[A-Za-z_]\w*\s*;",
        replacement="",
        description="delete expression removed",
    ),
    ModernizationRule(
        pattern=r"printf\s*\(",
        replacement="std::print(",
        description="printf replaced with std::print",
    ),
    ModernizationRule(
        pattern=r"\((int|float|double)\)\s*",
        replacement=r"static_cast<\1>(",
        description="C-style numeric cast replaced with static_cast",
    ),
]


def apply_modernization_rules(code: str) -> Tuple[str, List[str]]:
    """Apply regex modernization rules and return updated code and applied descriptions."""
    updated_code = code
    applied_descriptions: List[str] = []

    for rule in _RULES:
        updated_code, substitutions = rule.pattern.subn(rule.replacement, updated_code)
        if substitutions > 0:
            applied_descriptions.append(rule.description)

    return updated_code, applied_descriptions
