"""Tests for the plugin loader and built-in rules."""

from __future__ import annotations

import pytest


class TestPluginLoader:
    def test_load_all_rules_returns_list(self):
        from plugins.loader import load_all_rules
        rules = load_all_rules(force_reload=True)
        assert isinstance(rules, list)

    def test_builtin_rules_discovered(self):
        from plugins.loader import load_all_rules
        rules = load_all_rules(force_reload=True)
        ids = [r.id for r in rules]
        assert "null_to_nullptr" in ids
        assert "typedef_to_using" in ids

    def test_all_rules_have_required_attrs(self):
        from plugins.loader import load_all_rules
        for rule in load_all_rules(force_reload=True):
            assert rule.id, f"Rule {type(rule).__name__} has no id"
            assert rule.description, f"Rule {rule.id} has no description"
            assert rule.severity in {"critical", "major", "minor"}, (
                f"Rule {rule.id} has invalid severity: {rule.severity!r}"
            )


class TestNullToNullptrRule:
    def _rule(self):
        from plugins.rules.null_to_nullptr import NullToNullptrRule
        return NullToNullptrRule()

    def test_matches_null(self):
        rule = self._rule()
        assert rule.matches("int* p = NULL;") is True

    def test_no_match_without_null(self):
        rule = self._rule()
        assert rule.matches("int* p = nullptr;") is False

    def test_apply_replaces_null(self):
        rule = self._rule()
        result, count = rule.apply("if (p == NULL) return NULL;")
        assert "NULL" not in result
        assert count == 2
        assert result == "if (p == nullptr) return nullptr;"

    def test_apply_no_change_when_no_match(self):
        rule = self._rule()
        code = "auto p = std::make_unique<int>(5);"
        result, count = rule.apply(code)
        assert result == code
        assert count == 0


class TestTypedefToUsingRule:
    def _rule(self):
        from plugins.rules.typedef_to_using import TypedefToUsingRule
        return TypedefToUsingRule()

    def test_matches_typedef(self):
        rule = self._rule()
        assert rule.matches("typedef int MyInt;") is True

    def test_no_match_without_typedef(self):
        rule = self._rule()
        assert rule.matches("using MyInt = int;") is False

    def test_apply_converts_typedef(self):
        rule = self._rule()
        result, count = rule.apply("typedef unsigned long size_type;")
        assert "using size_type = unsigned long;" in result
        assert count == 1

    def test_apply_multiple_typedefs(self):
        rule = self._rule()
        code = "typedef int A;\ntypedef float B;\n"
        result, count = rule.apply(code)
        assert count == 2
        assert "typedef" not in result
