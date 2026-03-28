from __future__ import annotations

import re
from typing import Dict, List, Tuple


class ModernizationRule:
    def __init__(
        self,
        pattern: str,
        replacement: str,
        description: str,
        ast_triggers: tuple[str, ...] = (),
        hint_only: bool = False,
    ):
        self.pattern = re.compile(pattern, re.MULTILINE)
        self.replacement = replacement
        self.description = description
        self.ast_triggers = ast_triggers
        self.hint_only = hint_only


_RULES: List[ModernizationRule] = [
    # ------------------------------------------------------
    # LAYER 1: DETERMINISTIC REWRITES (Normalization/Enforcement)
    # ------------------------------------------------------
    ModernizationRule(
        pattern=r"\bNULL\b",
        replacement="nullptr",
        description="NULL -> nullptr",
        ast_triggers=("null_macro",),
    ),
    ModernizationRule(
        pattern=r"\btypedef\s+([^;{}]+?)\s+([A-Za-z_]\w*)\s*;",
        replacement=r"using \2 = \1;",
        description="typedef -> using",
    ),
    ModernizationRule(
        pattern=r"\bthrow\s*\(\s*\)",
        replacement="noexcept",
        description="throw() -> noexcept",
    ),
    # Macro constants -> constexpr
    ModernizationRule(
        pattern=r"^\s*#\s*define\s+([A-Z][A-Z0-9_]*)\s+([0-9]+(?:\.[0-9]+)?f?L?U?L?)\b.*$",
        replacement=r"constexpr auto \1 = \2;",
        description="#define constant -> constexpr",
    ),
    # C Headers -> C++ Headers
    ModernizationRule(
        pattern=r"#\s*include\s*<stdio\.h>",
        replacement="#include <cstdio>",
        description="<stdio.h> -> <cstdio>",
    ),
    ModernizationRule(
        pattern=r"#\s*include\s*<stdlib\.h>",
        replacement="#include <cstdlib>",
        description="<stdlib.h> -> <cstdlib>",
    ),
    ModernizationRule(
        pattern=r"#\s*include\s*<string\.h>",
        replacement="#include <cstring>",
        description="<string.h> -> <cstring>",
    ),
    ModernizationRule(
        pattern=r"#\s*include\s*<time\.h>",
        replacement="#include <ctime>",
        description="<time.h> -> <ctime>",
    ),

    # ------------------------------------------------------
    # LAYER 2: HINTS (Semantic guidance for LLM)
    # ------------------------------------------------------
    ModernizationRule(
        pattern=r"\(\s*([A-Za-z_][A-Za-z0-9_:<>]*)\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)",
        replacement=r"\g<0>",
        description="C-style cast - consider static_cast",
        ast_triggers=("c_style_cast",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bchar\s*\*\s*([A-Za-z_]\w*)\b",
        replacement=r"\g<0>",
        description="char* pointer - consider std::string or std::string_view",
        ast_triggers=("char_pointer",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bstd\s*::\s*auto_ptr\b",
        replacement="std::unique_ptr",
        description="std::auto_ptr -> std::unique_ptr (deprecated)",
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bprintf\s*\(",
        replacement=r"\g<0>",
        description="printf detected - consider std::cout or std::print (C++23)",
        ast_triggers=("printf_usage",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bmalloc\s*\(",
        replacement=r"\g<0>",
        description="malloc usage - migrate to make_unique / vector",
        ast_triggers=("malloc_usage",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bfree\s*\(",
        replacement=r"\g<0>",
        description="free usage - replace with smart pointer ownership",
        ast_triggers=("free_usage",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\btime_t\b|\blocaltime\b",
        replacement=r"\g<0>",
        description="legacy time API - migrate to std::chrono",
        ast_triggers=("time_usage",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"^\s*#\s*define\s+([a-zA-Z_]\w*)\s*\(.*?\).*$",
        replacement=r"\g<0>",
        description="functional macro - migrate to constexpr inline function",
        ast_triggers=("functional_macro",),
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"^\s*(.*?;)\s*\n\s*\1\s*$",
        replacement=r"\1",
        description="redundant duplicate statement - removed consecutive call",
        hint_only=False,
    ),
]


_COMMENTS_AND_STRINGS_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _mask_comments_and_strings(code: str) -> str:
    """Return code with comments/string literals replaced by spaces.

    Keeps length and newlines stable so regex match spans still align with the
    original source for safe substitutions.
    """

    def _blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return _COMMENTS_AND_STRINGS_RE.sub(_blank, code)


def _apply_rule_outside_comments_and_strings(code: str, rule: ModernizationRule) -> Tuple[str, int]:
    """Apply a regex replacement only on code tokens, skipping comments/strings."""
    masked = _mask_comments_and_strings(code)
    matches = list(rule.pattern.finditer(masked))
    if not matches:
        return code, 0

    chunks: List[str] = []
    cursor = 0
    substitutions = 0

    for match in matches:
        start, end = match.span()
        chunks.append(code[cursor:start])
        chunks.append(match.expand(rule.replacement))
        cursor = end
        substitutions += 1

    chunks.append(code[cursor:])
    return "".join(chunks), substitutions


def apply_modernization_rules(
    code: str,
    detected_patterns: Dict[str, int] | None = None,
) -> Tuple[str, List[str]]:
    """Apply modernization rules and return updated code with applied descriptions.

    Rules are evaluated sequentially and each rule sees modifications produced by
    earlier rules. This ordering is intentional so deterministic rewrites can be
    composed in a predictable way.
    """
    updated_code = code
    applied_descriptions: List[str] = []
    active_patterns = detected_patterns or {}

    for rule in _RULES:
        if rule.ast_triggers:
            trigger_active = any(int(active_patterns.get(name, 0) or 0) > 0 for name in rule.ast_triggers)
            if not trigger_active and active_patterns:
                continue

        if rule.hint_only:
            masked = _mask_comments_and_strings(updated_code)
            substitutions = sum(1 for _ in rule.pattern.finditer(masked))
        else:
            updated_code, substitutions = _apply_rule_outside_comments_and_strings(updated_code, rule)

        if substitutions > 0:
            applied_descriptions.append(f"{rule.description} ({substitutions} times)")
            print(f"[WORKING] RuleModernizer: Applied '{rule.description}' ({substitutions} times)")

    return updated_code, applied_descriptions


class RuleModernizer:
    """Wrapper class for modernization rules to maintain compatibility with ModelClient."""

    def __init__(self):
        pass

    def modernize_text(self, text: str) -> str:
        """Apply modernization rules to the provided text."""
        updated, _ = apply_modernization_rules(text)
        return updated
