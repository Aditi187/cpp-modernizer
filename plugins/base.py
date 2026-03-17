"""Plugin architecture for C++ modernization rules.

Usage
-----
Drop a Python module into ``plugins/rules/`` that defines a subclass of
:class:`RulePlugin`.  The engine discovers and loads all plugins automatically
on first use via :func:`load_all_rules`.

Example plugin (``plugins/rules/custom_span.py``)::

    from plugins.base import RulePlugin

    class UseSpanRule(RulePlugin):
        id          = "use_span"
        description = "Replace raw pointer + size pairs with std::span"
        severity    = "major"
        tags        = ("safety", "cpp20")

        def matches(self, source_text: str) -> bool:
            return "char*" in source_text and "size_t" in source_text

        def apply(self, source_text: str) -> tuple[str, int]:
            # Return (modified_source, replacement_count).
            return source_text, 0  # hint-only
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class RulePlugin(ABC):
    """Base class for all pluggable modernization rules.

    Subclasses **must** populate the class attributes below and implement
    :meth:`matches` and :meth:`apply`.
    """

    #: Unique identifier (snake_case).  Used as the key in reports.
    id: str = ""
    #: Human-readable description shown in the modernization report.
    description: str = ""
    #: One of ``"critical"``, ``"major"``, ``"minor"``.
    severity: str = "minor"
    #: Arbitrary tags for grouping / filtering.
    tags: tuple[str, ...] = ()

    @abstractmethod
    def matches(self, source_text: str) -> bool:
        """Return ``True`` if this rule has something to do with *source_text*."""

    @abstractmethod
    def apply(self, source_text: str) -> Tuple[str, int]:
        """Apply the rule and return ``(modified_source, replacement_count)``.

        Hint-only rules should return ``(source_text, 0)`` unmodified.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r} severity={self.severity!r}>"
