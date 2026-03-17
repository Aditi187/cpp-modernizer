"""Built-in rule: replace NULL macro with nullptr."""

from __future__ import annotations

import re

from plugins.base import RulePlugin


class NullToNullptrRule(RulePlugin):
    id          = "null_to_nullptr"
    description = "Replace NULL macro with nullptr (C++11+)"
    severity    = "major"
    tags        = ("safety", "cpp11")

    _PATTERN = re.compile(r"\bNULL\b")

    def matches(self, source_text: str) -> bool:
        return bool(self._PATTERN.search(source_text))

    def apply(self, source_text: str) -> tuple[str, int]:
        result, count = self._PATTERN.subn("nullptr", source_text)
        return result, count
