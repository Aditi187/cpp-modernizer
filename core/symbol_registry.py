from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from core.parser import CppParser

logger = logging.getLogger(__name__)


class HeaderSymbolExtractor:
    """
    AST-based Symbol Extractor for C++ headers.
    Uses CppParser to obtain the tree-sitter AST and walks it to extract
    public/declarative API symbols (namespaces, classes, methods, functions, typedefs, aliases, enums).
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_bytes = source_code.encode("utf-8")
        self.parser = CppParser()
        try:
            self.tree = self.parser._parser.parse(self.source_bytes)
        except Exception as e:
            logger.error(f"Failed to parse source in HeaderSymbolExtractor: {e}")
            self.tree = None
        self.symbols: List[Dict[str, Any]] = []

    def _node_text(self, node: Any) -> str:
        if node is None:
            return ""
        return self.source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _find_function_declarator(self, node: Any) -> Optional[Any]:
        if node.type == "function_declarator":
            return node
        if node.type == "function_definition":
            dec = node.child_by_field_name("declarator")
            if dec:
                return self._find_function_declarator(dec)
        for child in node.children:
            if child.type in {"function_definition", "declaration", "field_declaration"}:
                continue
            res = self._find_function_declarator(child)
            if res:
                return res
        return None

    def _extract_declarator_name(self, declarator_node: Any) -> str:
        for child in declarator_node.children:
            if child.type == "parameter_list":
                continue
            if child.type in {
                "identifier",
                "field_identifier",
                "operator_name",
                "qualified_identifier",
                "scoped_identifier",
            }:
                return self._node_text(child).strip()
            res = self._extract_declarator_name(child)
            if res:
                return res
        return ""

    def walk(self, node: Any, scope_stack: List[str], current_namespace: List[str]) -> None:
        if node is None:
            return

        node_type = node.type
        pushed_scope = None
        pushed_namespace = None

        if node_type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            ns_name = ""
            if name_node:
                ns_name = self._node_text(name_node).strip()
            if ns_name:
                current_namespace.append(ns_name)
                pushed_namespace = ns_name

        elif node_type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            class_name = ""
            if name_node:
                class_name = self._node_text(name_node).strip()
            else:
                # Fallback for anonymous or other structures
                for child in node.children:
                    if child.type in {"type_identifier", "identifier"}:
                        class_name = self._node_text(child).strip()
                        break

            if class_name:
                scope_stack.append(class_name)
                pushed_scope = class_name

                # Extract class/struct declaration signature
                body_node = node.child_by_field_name("body")
                end_byte = body_node.start_byte if body_node else node.end_byte
                sig = self.source_bytes[node.start_byte:end_byte].decode("utf-8", errors="replace").strip()
                if sig.endswith("{"):
                    sig = sig[:-1].strip()

                fqn = "::".join(scope_stack)
                ns_str = "::".join(current_namespace)
                self.symbols.append({
                    "name": fqn,
                    "kind": "class" if node_type == "class_specifier" else "struct",
                    "signature": sig,
                    "namespace": ns_str
                })

        elif node_type == "type_definition":
            type_name = self.parser._extract_type_name(node, self.source_bytes)
            if type_name:
                fqn = "::".join(scope_stack + [type_name])
                sig = self._node_text(node).strip()
                self.symbols.append({
                    "name": fqn,
                    "kind": "typedef",
                    "signature": sig,
                    "namespace": "::".join(current_namespace)
                })

        elif node_type in {"alias_declaration", "using_declaration"}:
            type_name = self.parser._extract_type_name(node, self.source_bytes)
            if type_name:
                fqn = "::".join(scope_stack + [type_name])
                sig = self._node_text(node).strip()
                self.symbols.append({
                    "name": fqn,
                    "kind": "type_alias",
                    "signature": sig,
                    "namespace": "::".join(current_namespace)
                })

        elif node_type == "enum_specifier":
            name_node = node.child_by_field_name("name")
            if name_node:
                enum_name = self._node_text(name_node).strip()
                if enum_name:
                    fqn = "::".join(scope_stack + [enum_name])
                    body_node = node.child_by_field_name("body")
                    end_byte = body_node.start_byte if body_node else node.end_byte
                    sig = self.source_bytes[node.start_byte:end_byte].decode("utf-8", errors="replace").strip()
                    if sig.endswith("{"):
                        sig = sig[:-1].strip()
                    self.symbols.append({
                        "name": fqn,
                        "kind": "enum",
                        "signature": sig,
                        "namespace": "::".join(current_namespace)
                    })

        elif node_type in {"function_definition", "declaration", "field_declaration"}:
            func_dec = self._find_function_declarator(node)
            if func_dec:
                func_name = self._extract_declarator_name(func_dec)
                if func_name:
                    # Deduplicate: out-of-line method definitions like ClassName::method
                    # already contain the class scope in func_name.
                    if "::" in func_name:
                        fqn = func_name
                    else:
                        fqn = "::".join(scope_stack + [func_name])

                    body_node = node.child_by_field_name("body")
                    end_byte = body_node.start_byte if body_node else node.end_byte
                    sig = self.source_bytes[node.start_byte:end_byte].decode("utf-8", errors="replace").strip()
                    if sig.endswith("{"):
                        sig = sig[:-1].strip()
                    if sig.endswith(";"):
                        sig = sig[:-1].strip()

                    kind = "method" if scope_stack else "function"
                    self.symbols.append({
                        "name": fqn,
                        "kind": kind,
                        "signature": sig,
                        "namespace": "::".join(current_namespace)
                    })

        # Traverse children
        for child in node.children:
            self.walk(child, scope_stack, current_namespace)

        # Pop scope/namespace after finishing node children traversal
        if pushed_scope:
            scope_stack.pop()
        if pushed_namespace:
            current_namespace.pop()

    def get_extracted_symbols(self) -> List[Dict[str, Any]]:
        if self.tree is None:
            return []
        self.symbols = []
        self.walk(self.tree.root_node, [], [])
        return self.symbols


def extract_symbols_from_header(file_path: str, source_code: str) -> List[Dict[str, Any]]:
    """
    Extract public API declaration symbols from a C++ header code.
    Returns list of dict with name, kind, signature, namespace.
    """
    extractor = HeaderSymbolExtractor(source_code)
    return extractor.get_extracted_symbols()


def format_symbols_as_headers(symbols: List[Dict[str, Any]]) -> str:
    """
    Reconstruct a clean C++ header file declaring public API structures
    and signatures from a list of symbol records.
    """
    if not symbols:
        return ""

    ns_map: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        ns = sym.get("namespace") or ""
        name = sym.get("name") or ""
        kind = sym.get("kind") or ""
        sig = sym.get("signature") or ""

        if ns not in ns_map:
            ns_map[ns] = {"classes": {}, "globals": []}

        # Check if the symbol is a member of a class (e.g. name contains "::")
        if "::" in name:
            parts = name.split("::")
            class_name = parts[0]
            # Ensure class entry exists
            if class_name not in ns_map[ns]["classes"]:
                ns_map[ns]["classes"][class_name] = {
                    "signature": f"class {class_name}",
                    "methods": [],
                    "kind": "class"
                }
            if kind in ("method", "function"):
                ns_map[ns]["classes"][class_name]["methods"].append(sig)
        else:
            if kind in ("class", "struct", "enum"):
                if name not in ns_map[ns]["classes"]:
                    ns_map[ns]["classes"][name] = {
                        "signature": sig,
                        "methods": [],
                        "kind": kind
                    }
                else:
                    ns_map[ns]["classes"][name]["signature"] = sig
                    ns_map[ns]["classes"][name]["kind"] = kind
            else:
                ns_map[ns]["globals"].append({"kind": kind, "signature": sig})

    lines: List[str] = []
    # Sort namespaces so global/empty is processed last/consistently
    for ns in sorted(ns_map.keys()):
        ns_indent = ""
        if ns:
            lines.append(f"namespace {ns} {{")
            ns_indent = "    "

        # Format classes/structs/enums
        for class_name, class_info in ns_map[ns]["classes"].items():
            kind = class_info["kind"]
            sig = class_info["signature"]
            methods = class_info["methods"]

            lines.append(f"{ns_indent}{sig} {{")
            if kind in ("class", "struct"):
                lines.append(f"{ns_indent}public:")
                for m in sorted(methods):
                    lines.append(f"{ns_indent}    {m};")
            lines.append(f"{ns_indent}}};")
            lines.append("")

        # Format global functions and typedefs/aliases
        for glob in ns_map[ns]["globals"]:
            sig = glob["signature"]
            if not sig.endswith(";"):
                sig += ";"
            lines.append(f"{ns_indent}{sig}")

        if ns:
            if lines and lines[-1] == "":
                lines.pop()
            lines.append("}")
            lines.append("")

    return "\n".join(lines).strip()

