"""Built-in rule: replace `typedef` with `using` (C++11+)."""

from __future__ import annotations

import re

from plugins.base import RulePlugin


class TypedefToUsingRule(RulePlugin):
    id          = "typedef_to_using"
    description = "Replace `typedef T Name;` with `using Name = T;` (C++11+)"
    severity    = "minor"
    tags        = ("style", "cpp11")

    _PATTERN = re.compile(
        r"\btypedef\s+([^;{}]+?)\s+([A-Za-z_]\w*)\s*;",
        re.MULTILINE,
    )

    def matches(self, source_text: str) -> bool:
        return bool(self._PATTERN.search(source_text))

    def apply(self, source_text: str) -> tuple[str, int]:
        count = 0

        def _repl(m: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return f"using {m.group(2)} = {m.group(1).strip()};"

        result = self._PATTERN.sub(_repl, source_text)
        return result, count
