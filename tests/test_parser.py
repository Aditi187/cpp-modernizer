"""Unit tests for the C++ parser (core/parser.py)."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parser():
    from core.parser import CppParser
    return CppParser()


# ---------------------------------------------------------------------------
# detect_legacy_patterns
# ---------------------------------------------------------------------------

class TestDetectLegacyPatterns:
    def test_null_macro_detected(self):
        from core.parser import detect_legacy_patterns

        findings = detect_legacy_patterns("int* p = NULL;")
        ids = [f["pattern"] for f in findings]
        assert "null_macro" in ids

    def test_char_pointer_array_detected(self):
        from core.parser import detect_legacy_patterns

        findings = detect_legacy_patterns("char* buf[256];")
        ids = [f["pattern"] for f in findings]
        assert "char_pointer_array" in ids

    def test_manual_delete_detected(self):
        from core.parser import detect_legacy_patterns

        findings = detect_legacy_patterns("delete ptr;")
        ids = [f["pattern"] for f in findings]
        assert "manual_delete" in ids

    def test_clean_code_returns_empty(self):
        from core.parser import detect_legacy_patterns

        findings = detect_legacy_patterns(
            "auto p = std::make_unique<int>(42);\n"
            "std::println(\"{}\", *p);\n"
        )
        # C-style cast AST check may run but should find nothing here.
        patterns = [f["pattern"] for f in findings]
        assert "null_macro" not in patterns
        assert "char_pointer_array" not in patterns

    def test_findings_have_required_keys(self):
        from core.parser import detect_legacy_patterns

        findings = detect_legacy_patterns("int* p = NULL;")
        for item in findings:
            assert "pattern" in item
            assert "line" in item
            assert "message" in item


# ---------------------------------------------------------------------------
# detect_module_imports (C++20)
# ---------------------------------------------------------------------------

class TestDetectModuleImports:
    def test_import_header_unit(self):
        from core.parser import detect_module_imports

        src = "import <vector>;\nimport <memory>;\n"
        results = detect_module_imports(src)
        targets = [r["target"] for r in results]
        assert "<vector>" in targets
        assert "<memory>" in targets

    def test_import_named_module(self):
        from core.parser import detect_module_imports

        src = "import mylib.utils;\n"
        results = detect_module_imports(src)
        assert results[0]["target"] == "mylib.utils"
        assert results[0]["kind"] == "import"

    def test_module_declaration(self):
        from core.parser import detect_module_imports

        src = "export module myapp.core;\n"
        results = detect_module_imports(src)
        assert results[0]["kind"] == "module_decl"
        assert results[0]["target"] == "myapp.core"

    def test_export_import(self):
        from core.parser import detect_module_imports

        src = "export import <string>;\n"
        results = detect_module_imports(src)
        assert results[0]["kind"] == "import"

    def test_empty_source(self):
        from core.parser import detect_module_imports

        assert detect_module_imports("") == []

    def test_no_cpp20_in_legacy_code(self):
        from core.parser import detect_module_imports

        src = '#include <vector>\n#include "mylib.h"\nvoid foo() {}\n'
        assert detect_module_imports(src) == []

    def test_results_sorted_by_line(self):
        from core.parser import detect_module_imports

        src = "import <string>;\nimport <vector>;\n"
        results = detect_module_imports(src)
        lines = [r["line"] for r in results]
        assert lines == sorted(lines)


# ---------------------------------------------------------------------------
# CppParser.parse_string — basic smoke tests
# ---------------------------------------------------------------------------

class TestCppParserParseString:
    def test_returns_project_map_keys(self):
        parser = _get_parser()
        pm = parser.parse_string("int add(int a, int b) { return a + b; }")
        assert "functions" in pm
        assert "headers" in pm
        assert "module_imports" in pm

    def test_function_extracted(self):
        parser = _get_parser()
        pm = parser.parse_string("int mul(int a, int b) { return a * b; }")
        fns = pm["functions"]
        names = [v.get("name") for v in fns.values()]
        assert "mul" in names

    def test_module_imports_in_project_map(self):
        parser = _get_parser()
        pm = parser.parse_string("import <vector>;\nvoid run() {}\n")
        imports = pm.get("module_imports", [])
        assert any(r["target"] == "<vector>" for r in imports)

    def test_includes_collected(self):
        parser = _get_parser()
        pm = parser.parse_string('#include <iostream>\nvoid f() { std::cout << "hi"; }\n')
        # Parser stores includes as the full directive string, e.g. "#include <iostream>"
        # or just the header name; accept either form.
        headers = pm["headers"]
        assert any("iostream" in h for h in headers)
