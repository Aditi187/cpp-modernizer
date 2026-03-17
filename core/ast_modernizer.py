from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from core.parser import CppParser


class ASTModernizationDetector:
    def __init__(self, parser: CppParser | None = None) -> None:
        self.parser = parser or CppParser()

    def get_function_ast_node(self, function_source: str) -> Any:
        source_bytes = function_source.encode("utf-8")
        tree = self.parser.parse_bytes(source_bytes)

        # Prefer a concrete function_definition node if present.
        for node in self._iter_ast(tree.root_node):
            if node.type == "function_definition":
                return node

        # Fallback to root node when function_definition is not found.
        return tree.root_node

    def detect_legacy_patterns(self, function_node: Any, source_bytes: bytes) -> Dict[str, Any]:
        patterns: Dict[str, int] = {
            "raw_new": 0,
            "raw_delete": 0,
            "printf_usage": 0,
            "raw_pointer": 0,
            "malloc_usage": 0,
            "free_usage": 0,
            "null_macro": 0,
            "c_style_cast": 0,
            "index_loop": 0,
            "char_pointer": 0,
            "memcpy_usage": 0,
            "auto_ptr_usage": 0,
            "throw_spec": 0,
        }
        locations: Dict[str, List[Dict[str, int]]] = defaultdict(list)

        if function_node is None:
            return {"counts": patterns, "detected": [], "locations": {}}

        if not source_bytes:
            raise ValueError("source_bytes is required for AST text extraction")

        for node in self._iter_ast(function_node):
            node_type = getattr(node, "type", "")

            if node_type == "new_expression":
                patterns["raw_new"] += 1
                self._record_location(locations, "raw_new", node)
            elif node_type == "delete_expression":
                patterns["raw_delete"] += 1
                self._record_location(locations, "raw_delete", node)
            elif node_type == "pointer_declarator":
                decl_text = self._pointer_declaration_text(node, source_bytes)
                lowered = decl_text.lower()
                is_parameter = self._ancestor_of_type(node, "parameter_declaration") is not None
                is_local_decl = self._ancestor_of_type(node, "declaration") is not None
                is_const_qualified = bool(re.search(r"\bconst\b", lowered))

                # Count only mutable local pointers as raw pointers; parameters and
                # const-qualified declarations are intentionally excluded.
                if is_local_decl and not is_parameter and not is_const_qualified:
                    patterns["raw_pointer"] += 1
                    self._record_location(locations, "raw_pointer", node)

                if is_local_decl and not is_parameter and "char" in lowered and not is_const_qualified:
                    patterns["char_pointer"] += 1
                    self._record_location(locations, "char_pointer", node)
            elif node_type in {"cast_expression", "c_style_cast_expression"}:
                patterns["c_style_cast"] += 1
                self._record_location(locations, "c_style_cast", node)
            elif node_type == "for_statement":
                if self._is_index_based_for_loop(node, source_bytes):
                    patterns["index_loop"] += 1
                    self._record_location(locations, "index_loop", node)
            elif node_type == "call_expression":
                node_text = self._node_text(node, source_bytes)
                if "printf" in node_text:
                    patterns["printf_usage"] += 1
                    self._record_location(locations, "printf_usage", node)
                if "malloc" in node_text:
                    patterns["malloc_usage"] += 1
                    self._record_location(locations, "malloc_usage", node)
                if "free" in node_text:
                    patterns["free_usage"] += 1
                    self._record_location(locations, "free_usage", node)
                if "memcpy" in node_text:
                    patterns["memcpy_usage"] += 1
                    self._record_location(locations, "memcpy_usage", node)
            elif node_type == "identifier":
                node_text = self._node_text(node, source_bytes)
                if node_text == "NULL":
                    patterns["null_macro"] += 1
                    self._record_location(locations, "null_macro", node)
            elif node_type in {"type_identifier", "qualified_identifier", "scoped_identifier"}:
                node_text = self._node_text(node, source_bytes)
                if "auto_ptr" in node_text:
                    patterns["auto_ptr_usage"] += 1
                    self._record_location(locations, "auto_ptr_usage", node)
            elif node_type in {"noexcept", "throw_specifier", "dynamic_exception_specification"}:
                # Detect legacy dynamic exception specifications (throw()).
                node_text = self._node_text(node, source_bytes)
                if "throw" in node_text:
                    patterns["throw_spec"] += 1
                    self._record_location(locations, "throw_spec", node)

        detected = [name for name, count in patterns.items() if count > 0]
        return {
            "counts": patterns,
            "detected": detected,
            "locations": {name: rows for name, rows in locations.items() if rows},
        }

    def _iter_ast(self, root: Any) -> Iterable[Any]:
        stack: List[Any] = [root]
        while stack:
            current = stack.pop()
            yield current
            for child in reversed(getattr(current, "children", ())):
                stack.append(child)

    def _node_text(self, node: Any, source_bytes: bytes) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _record_location(self, bucket: Dict[str, List[Dict[str, int]]], pattern: str, node: Any) -> None:
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        bucket[pattern].append(
            {
                "start_line": int(start_row) + 1,
                "start_col": int(start_col) + 1,
                "end_line": int(end_row) + 1,
                "end_col": int(end_col) + 1,
            }
        )

    def _ancestor_of_type(self, node: Any, target_type: str) -> Any | None:
        current = getattr(node, "parent", None)
        while current is not None:
            if getattr(current, "type", "") == target_type:
                return current
            current = getattr(current, "parent", None)
        return None

    def _pointer_declaration_text(self, pointer_node: Any, source_bytes: bytes) -> str:
        param_decl = self._ancestor_of_type(pointer_node, "parameter_declaration")
        if param_decl is not None:
            return self._node_text(param_decl, source_bytes)

        decl = self._ancestor_of_type(pointer_node, "declaration")
        if decl is not None:
            return self._node_text(decl, source_bytes)

        return self._node_text(pointer_node, source_bytes)

    def _is_index_based_for_loop(self, for_node: Any, source_bytes: bytes) -> bool:
        init = for_node.child_by_field_name("initializer")
        condition = for_node.child_by_field_name("condition")
        update = for_node.child_by_field_name("update")

        if init is None or condition is None or update is None:
            return False

        init_text = self._node_text(init, source_bytes)
        cond_text = self._node_text(condition, source_bytes)
        update_text = self._node_text(update, source_bytes)

        init_match = re.search(
            r"\b(?:int|long|short|size_t|std::size_t|unsigned|std::ptrdiff_t)\b[^;=]*\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\b",
            init_text,
        )
        if not init_match:
            return False
        index_var = init_match.group(1)

        cond_ok = re.search(rf"\b{re.escape(index_var)}\b\s*(?:<|<=)\s*", cond_text) is not None
        update_ok = re.search(
            rf"(?:\+\+\s*{re.escape(index_var)}|{re.escape(index_var)}\s*\+\+|{re.escape(index_var)}\s*\+=\s*1)",
            update_text,
        ) is not None
        return cond_ok and update_ok
