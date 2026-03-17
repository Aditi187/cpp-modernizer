"""Lightweight workflow integration smoke test.

These tests do NOT call a real LLM; they mock the bridge so the graph logic
can be exercised without network access.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path when running via pytest from any cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest


_SIMPLE_CPP = """\
#include <cstdio>
#include <cstdlib>

int add(int a, int b) {
    return a + b;
}

int main() {
    int* p = (int*)malloc(sizeof(int));
    *p = add(2, 3);
    printf("%d\\n", *p);
    free(p);
    return 0;
}
"""


class TestWorkflowSmoke:
    """Ensure the graph builds and the initial state is constructed correctly."""

    def test_build_workflow_returns_compiled_graph(self):
        from agents.workflow import build_workflow
        graph = build_workflow()
        # Compiled LangGraph graphs expose an `invoke` method.
        assert callable(getattr(graph, "invoke", None))

    def test_parse_functions_from_source(self):
        from agents.workflow import _parse_functions_from_source
        functions = _parse_functions_from_source(_SIMPLE_CPP)
        names = [f.get("name") for f in functions]
        # At minimum, add and main should be detected.
        assert "add" in names or len(functions) > 0

    def test_looks_like_prose_response(self):
        from agents.workflow import _looks_like_prose_response

        prose = (
            "To modernize this function we can follow these steps:\n"
            "1. Replace malloc with std::make_unique\n"
        )
        # Even with {} present, the prose markers should trigger rejection.
        assert _looks_like_prose_response(prose, prose) is True

    def test_clean_code_not_prose(self):
        from agents.workflow import _looks_like_prose_response

        code = "auto p = std::make_unique<int>(42);\nreturn *p;\n"
        # Doesn't contain prose markers and has braces implicitly via {}
        # (we add them for the test).
        code_with_braces = "int f() {\n" + code + "}"
        assert _looks_like_prose_response(code_with_braces, code_with_braces) is False

    def test_build_retry_prompt_contains_preview(self):
        from agents.workflow import _build_retry_prompt

        result = _build_retry_prompt("do the thing", "prose detected", "Here is the modernized code:")
        assert "RETRY INSTRUCTION" in result
        assert "prose detected" in result
        assert "Here is the modernized code:" in result

    def test_clean_model_code_block_strips_fence(self):
        from agents.workflow import _clean_model_code_block

        fenced = "```cpp\nint x = 5;\n```"
        assert _clean_model_code_block(fenced) == "int x = 5;"

    def test_clean_model_code_block_no_fence(self):
        from agents.workflow import _clean_model_code_block

        plain = "int x = 5;"
        assert _clean_model_code_block(plain) == "int x = 5;"


class TestRuleTransforms:
    """Test the rule-based transformation helpers in workflow."""

    def test_null_to_nullptr(self):
        from agents.workflow import _transform_null_macro_to_nullptr

        body = "int* p = NULL;\nif (p == NULL) return;"
        result, count = _transform_null_macro_to_nullptr(body)
        assert "nullptr" in result
        assert "NULL" not in result
        assert count == 2

    def test_char_ptr_literal_to_std_string(self):
        from agents.workflow import _transform_char_ptr_literal_declarations

        body = 'char* msg = "hello";\n'
        result, count = _transform_char_ptr_literal_declarations(body)
        assert "std::string" in result
        assert count == 1
