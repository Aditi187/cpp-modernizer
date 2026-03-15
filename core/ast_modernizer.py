from __future__ import annotations

from typing import Any, Dict

from core.parser import CppParser


class ASTModernizationDetector:
    def __init__(self, parser: CppParser | None = None) -> None:
        self.parser = parser or CppParser()

    def get_function_ast_node(self, function_source: str) -> Any:
        source_bytes = function_source.encode("utf-8")
        tree = self.parser._parser.parse(source_bytes)

        # Prefer a concrete function_definition node if present.
        for node in self.parser._iter_nodes(tree.root_node):
            if node.type == "function_definition":
                return node

        # Fallback to root node when function_definition is not found.
        return tree.root_node

    def detect_legacy_patterns(self, function_node: Any) -> Dict[str, int]:
        patterns: Dict[str, int] = {
            "raw_new": 0,
            "raw_delete": 0,
            "printf_usage": 0,
            "raw_pointer": 0,
        }

        if function_node is None:
            return patterns

        source_bytes = self._resolve_source_bytes(function_node)
        for node in self.parser._iter_nodes(function_node):
            node_type = getattr(node, "type", "")

            if node_type == "new_expression":
                patterns["raw_new"] += 1
            elif node_type == "delete_expression":
                patterns["raw_delete"] += 1
            elif node_type == "pointer_declarator":
                patterns["raw_pointer"] += 1
            elif node_type == "call_expression":
                node_text = self._node_text(node, source_bytes)
                if "printf" in node_text:
                    patterns["printf_usage"] += 1

        return patterns

    def _resolve_source_bytes(self, function_node: Any) -> bytes:
        root = getattr(function_node, "tree", None)
        if root is not None:
            # tree_sitter node objects in python do not expose source bytes directly.
            # We rely on text access as available, else fallback to encoded node text.
            pass
        node_text = self._node_text(function_node, b"")
        return node_text.encode("utf-8", errors="replace")

    def _node_text(self, node: Any, source_bytes: bytes) -> str:
        # If the parser helper can read from provided bytes, use that first.
        if source_bytes:
            try:
                return self.parser._node_text(node, source_bytes)
            except Exception:
                pass

        # Newer tree-sitter Python bindings often expose node.text bytes.
        node_text_value = getattr(node, "text", None)
        if isinstance(node_text_value, (bytes, bytearray)):
            return bytes(node_text_value).decode("utf-8", errors="replace")

        # Last-resort fallback.
        return str(node)
