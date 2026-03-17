"""Plugin loader — discovers and registers all :class:`RulePlugin` subclasses.

Plugins are loaded from:
1. The built-in ``plugins/rules/`` directory (ships with the project).
2. Any additional directories listed in the ``PLUGIN_DIRS`` environment
   variable (colon-separated on Linux/Mac, semicolon-separated on Windows).

Usage::

    from plugins.loader import load_all_rules
    rules = load_all_rules()
    for rule in rules:
        if rule.matches(my_code):
            new_code, n = rule.apply(my_code)
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import List, Type

from plugins.base import RulePlugin

_BUILTIN_RULES_DIR = Path(__file__).parent / "rules"
_loaded: bool = False
_registry: List[RulePlugin] = []


def _discover_plugins_in_dir(directory: Path) -> None:
    """Import all Python modules in *directory* so subclasses register themselves."""
    if not directory.is_dir():
        return
    package_path = str(directory.parent)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    package_name = directory.name
    for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(directory)]):
        full_name = f"{package_name}.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:  # pragma: no cover
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load plugin module %r: %s", full_name, exc
            )


def _collect_subclasses(cls: Type[RulePlugin]) -> List[Type[RulePlugin]]:
    """Recursively collect all non-abstract subclasses of *cls*."""
    result: List[Type[RulePlugin]] = []
    for sub in cls.__subclasses__():
        if sub.id:  # skip abstract intermediaries that don't set an id
            result.append(sub)
        result.extend(_collect_subclasses(sub))
    return result


def load_all_rules(force_reload: bool = False) -> List[RulePlugin]:
    """Return instantiated :class:`RulePlugin` objects from all discovered modules.

    Results are cached after the first call.  Pass ``force_reload=True`` to
    re-scan the plugin directories (useful in tests).
    """
    global _loaded, _registry
    if _loaded and not force_reload:
        return list(_registry)

    # Built-in rules directory.
    _discover_plugins_in_dir(_BUILTIN_RULES_DIR)

    # Extra directories from environment.
    sep = ";" if sys.platform == "win32" else ":"
    extra = os.environ.get("PLUGIN_DIRS", "").strip()
    for extra_dir in extra.split(sep):
        stripped = extra_dir.strip()
        if stripped:
            _discover_plugins_in_dir(Path(stripped))

    _registry = [cls() for cls in _collect_subclasses(RulePlugin)]
    _loaded = True
    return list(_registry)


def get_rules_by_severity(severity: str) -> List[RulePlugin]:
    """Return all loaded rules matching *severity* (``critical``, ``major``, ``minor``)."""
    return [r for r in load_all_rules() if r.severity == severity]
