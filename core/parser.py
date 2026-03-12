from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tree_sitter import Language, Parser


_CPP_KEYWORDS_TO_IGNORE = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "alignof",
    "decltype",
    "new",
    "delete",
    "throw",
    "catch",
    "static_cast",
    "dynamic_cast",
    "reinterpret_cast",
    "const_cast",
}

_MODIFIER_TOKENS = ("virtual", "static", "inline", "constexpr", "consteval")


ProjectMap = Dict[str, Any]

_STD_HEADER_SYMBOLS: Dict[str, Set[str]] = {
    "<vector>": {"std::vector", "vector"},
    "<string>": {"std::string", "string"},
    "<map>": {"std::map", "map"},
    "<unordered_map>": {"std::unordered_map", "unordered_map"},
    "<set>": {"std::set", "set"},
    "<unordered_set>": {"std::unordered_set", "unordered_set"},
    "<memory>": {"std::unique_ptr", "std::shared_ptr", "unique_ptr", "shared_ptr"},
    "<optional>": {"std::optional", "optional"},
    "<variant>": {"std::variant", "variant"},
    "<tuple>": {"std::tuple", "tuple"},
    "<utility>": {"std::move", "std::forward", "move", "forward"},
    "<algorithm>": {"std::sort", "std::find", "sort", "find"},
    "<iostream>": {"std::cout", "std::cin", "std::cerr", "cout", "cin", "cerr"},
    "<sstream>": {"std::stringstream", "std::ostringstream", "std::istringstream"},
    "<thread>": {"std::thread", "thread"},
    "<mutex>": {"std::mutex", "std::lock_guard", "mutex", "lock_guard"},
    "<chrono>": {"std::chrono", "chrono"},
}


class CppParser:
    """High-fidelity C++ semantic extraction engine for modernization workflows."""

    def __init__(self) -> None:
        self._parser = self._create_cpp_parser()
        self._last_project_map: Optional[ProjectMap] = None

    @staticmethod
    def _create_cpp_parser() -> Parser:
        """
        Create a parser configured for C++ with tree-sitter v0.21+ compatibility.
        """

        try:
            import tree_sitter_cpp
        except ImportError as exc:
            raise RuntimeError(
                "C++ grammar package 'tree-sitter-cpp' is not installed. "
                "Install it with 'pip install tree-sitter-cpp'."
            ) from exc

        cpp_language = Language(tree_sitter_cpp.language())
        parser = Parser()

        # Compatible with latest API while preserving support for older parser bindings.
        if hasattr(parser, "language"):
            parser.language = cpp_language
        elif hasattr(parser, "set_language"):
            parser.set_language(cpp_language)
        else:
            raise RuntimeError("Unsupported tree-sitter Parser API for setting language")

        return parser

    def parse_file(self, file_path: str | Path) -> ProjectMap:
        """Parse a C++ source file into a Janus-style ProjectMap."""

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"C++ file not found: {path}")
        return self.parse_string(path.read_text(encoding="utf-8"))

    def parse_string(self, source_text: str) -> ProjectMap:
        """Parse C++ source text into a Janus-style ProjectMap."""

        source_bytes = source_text.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        project_map = self._collect_semantic_map_single_pass(
            tree.root_node,
            source_text,
            source_bytes,
        )
        self._last_project_map = project_map
        return project_map

    def get_context_for_function(self, fqn: str) -> Dict[str, Any]:
        """
        Return context bundle for a function to support safe modernization.

        Includes:
        - function body
        - signatures of called functions
        - referenced custom struct/class definitions
        """

        if self._last_project_map is None:
            raise ValueError("No ProjectMap available. Run parse_file or parse_string first.")

        functions = self._last_project_map.get("functions", {})
        if not isinstance(functions, dict) or fqn not in functions:
            raise KeyError(f"Function not found in ProjectMap: {fqn}")

        function_meta = functions[fqn]
        body = str(function_meta.get("body") or "")
        internal_calls = function_meta.get("internal_calls", [])

        called_signatures: Dict[str, str] = {}
        for called_fqn in internal_calls:
            called_meta = functions.get(called_fqn)
            if isinstance(called_meta, dict):
                called_signatures[called_fqn] = str(called_meta.get("signature") or "")

        type_definitions = self._last_project_map.get("type_definitions", {})
        referenced_type_definitions: Dict[str, str] = {}
        if isinstance(type_definitions, dict):
            for type_name, type_source in type_definitions.items():
                if self._symbol_in_text(type_name, body):
                    referenced_type_definitions[str(type_name)] = str(type_source)

        return {
            "fqn": fqn,
            "body": body,
            "called_function_signatures": called_signatures,
            "referenced_type_definitions": referenced_type_definitions,
        }

    def _collect_semantic_map_single_pass(
        self,
        root_node: Any,
        source_text: str,
        source_bytes: bytes,
    ) -> ProjectMap:
        """
        Traverse AST once with TreeCursor while maintaining scope for FQNs.
        """

        scope_stack: List[str] = []
        functions: List[Dict[str, Any]] = []
        types: List[Dict[str, Any]] = []
        headers: List[str] = []
        global_context: Dict[str, List[Dict[str, Any]]] = {
            "struct": [],
            "class": [],
            "enum": [],
            "typedef": [],
            "type_alias": [],
        }

        line_starts = self._compute_line_start_bytes(source_text)

        stack: List[tuple[Any, int, bool]] = [(root_node, 0, False)]

        while stack:
            node, phase, pushed_scope = stack.pop()

            if phase == 1:
                if pushed_scope and scope_stack:
                    scope_stack.pop()
                continue

            if node.type == "function_definition":
                functions.append(
                    self._build_function_record(
                        node=node,
                        scope_stack=scope_stack,
                        source_text=source_text,
                        source_bytes=source_bytes,
                        line_starts=line_starts,
                    )
                )

            if node.type == "preproc_include":
                header = self._extract_include_directive(node, source_bytes)
                if header and header not in headers:
                    headers.append(header)

            type_record = self._build_type_record(node, scope_stack, source_bytes, line_starts)
            if type_record is not None:
                types.append(type_record)
                context_bucket = type_record.get("type")
                if isinstance(context_bucket, str) and context_bucket in global_context:
                    global_context[context_bucket].append(type_record)

            scope_name = self._scope_name(node, source_bytes)
            pushed_now = False
            if scope_name:
                scope_stack.append(scope_name)
                pushed_now = True

            stack.append((node, 1, pushed_now))
            for child in reversed(node.children):
                stack.append((child, 0, False))

        return self._build_project_map(functions, types, headers, global_context)

    def _build_project_map(
        self,
        functions: List[Dict[str, Any]],
        types: List[Dict[str, Any]],
        headers: List[str],
        global_context: Dict[str, List[Dict[str, Any]]],
    ) -> ProjectMap:
        function_map: Dict[str, Dict[str, Any]] = {}
        type_definitions: Dict[str, str] = {}

        for t in types:
            type_name = str(t.get("name") or "")
            if type_name and type_name not in type_definitions:
                type_definitions[type_name] = str(t.get("source_code") or "")

        all_fqns = {str(f.get("fqn") or "") for f in functions if f.get("fqn")}
        all_names = {str(f.get("name") or "") for f in functions if f.get("name")}
        name_to_fqn: Dict[str, str] = {}
        for f in functions:
            name = str(f.get("name") or "")
            fqn = str(f.get("fqn") or "")
            if name and fqn and name not in name_to_fqn:
                name_to_fqn[name] = fqn

        inbound: Dict[str, Set[str]] = {fqn: set() for fqn in all_fqns}

        include_requirements: Dict[str, List[str]] = {}

        for f in functions:
            fqn = str(f.get("fqn") or "")
            if not fqn:
                continue

            call_details = f.get("call_details", [])
            internal_calls: List[str] = []
            external_calls: List[str] = []

            if isinstance(call_details, list):
                for entry in call_details:
                    if not isinstance(entry, dict):
                        continue
                    call_name = str(entry.get("name") or "")
                    call_display = str(entry.get("display") or call_name)
                    normalized = self._normalize_call_target(call_name, call_display)

                    target_fqn: Optional[str] = None
                    if normalized in all_fqns:
                        target_fqn = normalized
                    elif normalized in name_to_fqn:
                        target_fqn = name_to_fqn[normalized]

                    if target_fqn and target_fqn != fqn:
                        internal_calls.append(target_fqn)
                        inbound[target_fqn].add(fqn)
                    elif call_display:
                        external_calls.append(call_display)

            internal_calls = sorted(set(internal_calls))
            external_calls = sorted(set(external_calls))

            includes_for_function = self._compute_include_requirements_for_function(f, headers)
            include_requirements[fqn] = includes_for_function

            out_degree = len(internal_calls)
            in_degree = len(inbound.get(fqn, set()))
            is_leaf = out_degree == 0
            is_root = in_degree >= 2 and out_degree <= 1

            merged = dict(f)
            merged["internal_calls"] = internal_calls
            merged["external_calls"] = external_calls
            merged["incoming_calls_count"] = in_degree
            merged["outgoing_calls_count"] = out_degree
            merged["is_leaf"] = is_leaf
            merged["is_root"] = is_root
            function_map[fqn] = merged

        dependency_order = self._compute_modernization_priority(function_map)

        return {
            "functions": function_map,
            "type_definitions": type_definitions,
            "dependency_order": dependency_order,
            "include_requirements": include_requirements,
            "headers": headers,
            "types": types,
            "global_context": global_context,
        }

    def _normalize_call_target(self, call_name: str, call_display: str) -> str:
        if "::" in call_display:
            return call_display
        return call_name or call_display

    def _compute_modernization_priority(self, function_map: Dict[str, Dict[str, Any]]) -> List[str]:
        leaves: List[str] = []
        roots: List[str] = []
        middles: List[str] = []

        for fqn, meta in function_map.items():
            in_degree = int(meta.get("incoming_calls_count", 0))
            out_degree = int(meta.get("outgoing_calls_count", 0))

            if out_degree == 0:
                leaves.append(fqn)
            elif in_degree >= 2 and out_degree <= 1:
                roots.append(fqn)
            else:
                middles.append(fqn)

        leaves.sort(key=lambda f: (function_map[f].get("incoming_calls_count", 0), f))
        middles.sort(key=lambda f: (function_map[f].get("outgoing_calls_count", 0), -int(function_map[f].get("incoming_calls_count", 0)), f))
        roots.sort(key=lambda f: (-int(function_map[f].get("incoming_calls_count", 0)), int(function_map[f].get("outgoing_calls_count", 0)), f))

        return [*leaves, *middles, *roots]

    def _compute_include_requirements_for_function(
        self,
        function_meta: Dict[str, Any],
        headers: List[str],
    ) -> List[str]:
        body = str(function_meta.get("body") or "")
        signature = str(function_meta.get("signature") or "")
        calls = function_meta.get("calls", [])
        joined_calls = "\n".join([str(c) for c in calls]) if isinstance(calls, list) else ""
        text = "\n".join([signature, body, joined_calls])

        required: List[str] = []
        for header_line in headers:
            header_name = self._extract_header_name(header_line)
            if not header_name:
                continue

            known_symbols = _STD_HEADER_SYMBOLS.get(header_name)
            if known_symbols:
                if any(self._symbol_in_text(symbol, text) for symbol in known_symbols):
                    required.append(header_name)
                    continue

            # Fallback: if include is project header and its basename symbol appears.
            if header_name.startswith('"') and header_name.endswith('"'):
                base = Path(header_name.strip('"')).stem
                if base and self._symbol_in_text(base, text):
                    required.append(header_name)

        return sorted(set(required))

    @staticmethod
    def _extract_header_name(include_line: str) -> str:
        match = re.search(r"#\s*include\s*([<\"].*[>\"])", include_line)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _symbol_in_text(symbol: str, text: str) -> bool:
        if not symbol:
            return False
        pattern = r"(?<!\w)" + re.escape(symbol) + r"(?!\w)"
        return re.search(pattern, text) is not None

    @staticmethod
    def _extract_include_directive(node: Any, source_bytes: bytes) -> str:
        return CppParser._node_text(node, source_bytes).strip()

    @staticmethod
    def _scope_name(node: Any, source_bytes: bytes) -> Optional[str]:
        if node.type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return None
            return CppParser._node_text(name_node, source_bytes).strip() or None

        if node.type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                for child in node.children:
                    if child.type in {"type_identifier", "identifier"}:
                        name_node = child
                        break
            if name_node is None:
                return None
            return CppParser._node_text(name_node, source_bytes).strip() or None

        return None

    def _build_function_record(
        self,
        node: Any,
        scope_stack: List[str],
        source_text: str,
        source_bytes: bytes,
        line_starts: List[int],
    ) -> Dict[str, Any]:
        owner_node = self._ownership_node(node)

        # Resolve qualified name to handle out-of-line definitions (e.g. MyClass::render).
        qual_parts = self._extract_function_qualified_parts(node, source_bytes)
        function_name = qual_parts[-1]

        # Merge scope_stack with qual_parts, deduplicating any overlapping prefix so that
        # e.g. `namespace App { void App::Foo::f() {} }` yields FQN `App::Foo::f`.
        if len(qual_parts) > 1:
            best_overlap = 0
            for overlap_len in range(min(len(scope_stack), len(qual_parts)), 0, -1):
                if list(scope_stack[-overlap_len:]) == qual_parts[:overlap_len]:
                    best_overlap = overlap_len
                    break
            fqn = "::".join([*scope_stack, *qual_parts[best_overlap:]])
        else:
            fqn = "::".join([*scope_stack, function_name]) if scope_stack else function_name

        body_node = node.child_by_field_name("body")
        body_text = self._node_text(body_node, source_bytes).strip() if body_node else ""

        signature_end_byte = body_node.start_byte if body_node is not None else node.end_byte
        ownership_start = self._ownership_start_byte(owner_node, source_text, source_bytes, line_starts)
        signature_text = source_bytes[ownership_start:signature_end_byte].decode(
            "utf-8", errors="replace"
        ).strip()

        start_line = self._byte_to_line_number(ownership_start, line_starts)
        end_line = node.end_point[0] + 1

        calls = self._collect_function_calls(node, source_bytes)
        modifiers = self._extract_modifiers(signature_text)
        parameters = self._extract_structured_parameters(node, source_bytes)

        return {
            "fqn": fqn,
            "name": function_name,
            "signature": signature_text,
            "body": body_text,
            "parameters": parameters,
            "calls": [call["display"] for call in calls],
            "call_details": calls,
            "start_byte": ownership_start,
            "end_byte": node.end_byte,
            "line_numbers": {"start": start_line, "end": end_line},
            "is_template": owner_node.type == "template_declaration",
            "modifiers": modifiers,
        }

    def _build_type_record(
        self,
        node: Any,
        scope_stack: List[str],
        source_bytes: bytes,
        line_starts: List[int],
    ) -> Optional[Dict[str, Any]]:
        node_type_map = {
            "class_specifier": "class",
            "struct_specifier": "struct",
            "enum_specifier": "enum",
            "type_definition": "typedef",
            "alias_declaration": "type_alias",
            "using_declaration": "type_alias",
        }

        semantic_type = node_type_map.get(node.type)
        if semantic_type is None:
            return None

        type_name = self._extract_type_name(node, source_bytes)
        if not type_name:
            return None

        fqn = "::".join([*scope_stack, type_name]) if scope_stack else type_name
        source_code = self._node_text(node, source_bytes)
        start_line = self._byte_to_line_number(node.start_byte, line_starts)
        end_line = node.end_point[0] + 1

        # Extract base classes for class/struct specifiers.
        bases: List[str] = []
        if node.type in {"class_specifier", "struct_specifier"}:
            for child in node.children:
                if child.type == "base_class_clause":
                    for base_child in child.children:
                        if base_child.type in {
                            "type_identifier",
                            "qualified_identifier",
                            "scoped_type_identifier",
                            "scoped_identifier",
                        }:
                            base_name = self._node_text(base_child, source_bytes).strip()
                            if base_name:
                                bases.append(base_name)

        return {
            "fqn": fqn,
            "name": type_name,
            "type": semantic_type,
            "source_code": source_code,
            "line_numbers": {"start": start_line, "end": end_line},
            "bases": bases,
        }

    @staticmethod
    def _ownership_node(function_node: Any) -> Any:
        parent = function_node.parent
        if parent is not None and parent.type == "template_declaration":
            return parent
        return function_node

    @staticmethod
    def _compute_line_start_bytes(source_text: str) -> List[int]:
        starts: List[int] = [0]
        offset = 0
        for line in source_text.splitlines(keepends=True):
            offset += len(line.encode("utf-8"))
            starts.append(offset)
        return starts

    @staticmethod
    def _byte_to_line_number(byte_offset: int, line_starts: List[int]) -> int:
        lo = 0
        hi = len(line_starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= byte_offset:
                lo = mid + 1
            else:
                hi = mid - 1
        return max(1, hi + 1)

    def _ownership_start_byte(
        self,
        owner_node: Any,
        source_text: str,
        source_bytes: bytes,
        line_starts: List[int],
    ) -> int:
        """
        Include leading comments + template declaration (if any) + function definition.
        """

        owner_start_row = owner_node.start_point[0]
        lines = source_text.splitlines()

        row = owner_start_row - 1
        first_included_row = owner_start_row
        seen_comment = False

        while row >= 0:
            line = lines[row].strip() if row < len(lines) else ""
            if not line:
                if seen_comment:
                    first_included_row = row
                    row -= 1
                    continue
                break

            if self._is_comment_line(line):
                seen_comment = True
                first_included_row = row
                row -= 1
                continue

            break

        if seen_comment and 0 <= first_included_row < len(line_starts):
            return line_starts[first_included_row]
        return owner_node.start_byte

    @staticmethod
    def _is_comment_line(stripped_line: str) -> bool:
        return (
            stripped_line.startswith("//")
            or stripped_line.startswith("/*")
            or stripped_line.startswith("*")
            or stripped_line.endswith("*/")
        )

    def _extract_function_qualified_parts(self, function_node: Any, source_bytes: bytes) -> List[str]:
        """Return the qualifier parts of a function name as a list.

        Out-of-line definitions like ``void MyClass::render()`` return
        ``['MyClass', 'render']``.  Inline definitions return ``['render']``.
        """
        declarator = function_node.child_by_field_name("declarator")
        search_root = declarator if declarator is not None else function_node

        for found in self._iter_nodes(search_root):
            if found.type in {"qualified_identifier", "scoped_identifier"}:
                text = self._node_text(found, source_bytes).strip()
                if text:
                    parts = [p.strip() for p in text.split("::") if p.strip()]
                    if parts:
                        return parts

        skipped_subtrees = {
            "parameter_list",
            "template_parameter_list",
            "argument_list",
        }
        stack: List[Any] = [search_root]
        while stack:
            current = stack.pop()
            if current.type in {"identifier", "field_identifier", "operator_name"}:
                text = self._node_text(current, source_bytes).strip()
                if text:
                    return [text]
            for child in reversed(current.children):
                if child.type in skipped_subtrees:
                    continue
                stack.append(child)

        return ["<anonymous_function>"]

    def _extract_function_name(self, function_node: Any, source_bytes: bytes) -> str:
        declarator = function_node.child_by_field_name("declarator")
        search_root = declarator if declarator is not None else function_node

        for found in self._iter_nodes(search_root):
            if found.type in {"qualified_identifier", "scoped_identifier"}:
                text = self._node_text(found, source_bytes).strip()
                if text:
                    return text.split("::")[-1].strip()

        skipped_subtrees = {
            "parameter_list",
            "template_parameter_list",
            "argument_list",
        }

        stack: List[Any] = [search_root]
        while stack:
            current = stack.pop()
            if current.type in {"identifier", "field_identifier", "operator_name"}:
                text = self._node_text(current, source_bytes).strip()
                if text:
                    return text

            for child in reversed(current.children):
                if child.type in skipped_subtrees:
                    continue
                stack.append(child)

        return "<anonymous_function>"

    def _extract_structured_parameters(
        self, function_node: Any, source_bytes: bytes
    ) -> List[Dict[str, Any]]:
        """Parse the parameter list into structured objects.

        Each object contains:
        - ``name``: parameter name (empty string for unnamed parameters)
        - ``type``: type string (e.g. ``'const std::string'``)
        - ``is_pointer``: True if the parameter is a pointer
        - ``is_reference``: True if the parameter is an lvalue or rvalue reference
        - ``is_const``: True if ``const`` appears in the parameter declaration
        """
        declarator = function_node.child_by_field_name("declarator")
        if declarator is None:
            return []

        param_list_node: Optional[Any] = None
        for sub in self._iter_nodes(declarator):
            if sub.type == "parameter_list":
                param_list_node = sub
                break

        if param_list_node is None:
            return []

        params: List[Dict[str, Any]] = []
        for child in param_list_node.children:
            if child.type == "parameter_declaration":
                params.append(self._parse_parameter_node(child, source_bytes))
        return params

    def _parse_parameter_node(
        self, param_node: Any, source_bytes: bytes
    ) -> Dict[str, Any]:
        """Build a structured dict from a single parameter_declaration AST node."""
        full_text = self._node_text(param_node, source_bytes).strip()
        is_pointer = False
        is_reference = False
        is_const = False
        name = ""
        type_parts: List[str] = []

        for child in param_node.children:
            ctype = child.type
            ctext = self._node_text(child, source_bytes).strip()

            if ctype == "type_qualifier":
                if ctext == "const":
                    is_const = True
                type_parts.append(ctext)

            elif ctype in {
                "primitive_type",
                "type_identifier",
                "sized_type_specifier",
                "qualified_identifier",
                "scoped_type_identifier",
                "template_type",
                "placeholder_type_specifier",
                "auto",
            }:
                type_parts.append(ctext)

            elif ctype == "pointer_declarator":
                is_pointer = True
                name = self._extract_declarator_name(child, source_bytes)

            elif ctype in {"reference_declarator", "rvalue_reference_declarator"}:
                is_reference = True
                name = self._extract_declarator_name(child, source_bytes)

            elif ctype == "identifier":
                # Plain name: type_parts should already be populated at this point.
                if type_parts:
                    name = ctext
                else:
                    type_parts.append(ctext)

            elif ctype == "abstract_pointer_declarator":
                is_pointer = True

            elif ctype == "abstract_reference_declarator":
                is_reference = True

        # Fallback: derive is_const from raw text if the AST walk missed it.
        if not is_const and re.search(r"\bconst\b", full_text):
            is_const = True

        return {
            "name": name,
            "type": " ".join(type_parts),
            "is_pointer": is_pointer,
            "is_reference": is_reference,
            "is_const": is_const,
        }

    def _extract_declarator_name(
        self, declarator_node: Any, source_bytes: bytes
    ) -> str:
        """Recursively extract the parameter name from a pointer/reference declarator."""
        for child in declarator_node.children:
            if child.type == "identifier":
                return self._node_text(child, source_bytes).strip()
            if child.type in {
                "pointer_declarator",
                "reference_declarator",
                "rvalue_reference_declarator",
            }:
                result = self._extract_declarator_name(child, source_bytes)
                if result:
                    return result
        return ""

    def _collect_function_calls(self, function_node: Any, source_bytes: bytes) -> List[Dict[str, str]]:
        body_node = function_node.child_by_field_name("body")
        if body_node is None:
            return []

        calls: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for node in self._iter_nodes(body_node):
            if node.type != "call_expression":
                continue

            callee_node = node.child_by_field_name("function")
            if callee_node is None:
                continue

            call_info = self._extract_callee_info(callee_node, source_bytes)
            if call_info is None:
                continue

            lowered = call_info["name"].lower()
            if lowered in _CPP_KEYWORDS_TO_IGNORE:
                continue

            dedupe_key = (call_info["kind"], call_info["display"])
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                calls.append(call_info)

        return calls

    def _extract_callee_info(self, callee_node: Any, source_bytes: bytes) -> Optional[Dict[str, str]]:
        if callee_node.type == "field_expression":
            obj_node = callee_node.child_by_field_name("argument")
            field_node = callee_node.child_by_field_name("field")
            if field_node is not None:
                name = self._node_text(field_node, source_bytes).strip()
                owner = self._node_text(obj_node, source_bytes).strip() if obj_node is not None else ""
                display = f"{owner}.{name}" if owner else name
                return {"name": name, "display": display, "kind": "method"}

        if callee_node.type in {"identifier", "field_identifier", "operator_name"}:
            name = self._node_text(callee_node, source_bytes).strip()
            return {"name": name, "display": name, "kind": "local"}

        if callee_node.type in {"qualified_identifier", "scoped_identifier"}:
            scoped_text = self._node_text(callee_node, source_bytes).strip()
            simple_name = scoped_text.split("::")[-1].strip() if "::" in scoped_text else scoped_text
            return {"name": simple_name, "display": scoped_text, "kind": "scoped"}

        # Fallback: find the first meaningful identifier under the function field subtree.
        for subnode in self._iter_nodes(callee_node):
            if subnode.type in {"field_identifier", "identifier", "operator_name"}:
                text = self._node_text(subnode, source_bytes).strip()
                if text:
                    return {"name": text, "display": text, "kind": "local"}
        return None

    def _extract_type_name(self, node: Any, source_bytes: bytes) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return self._node_text(name_node, source_bytes).strip()

        if node.type == "type_definition":
            candidates = [
                child
                for child in node.children
                if child.type in {"type_identifier", "identifier"}
            ]
            if candidates:
                return self._node_text(candidates[-1], source_bytes).strip()

        if node.type in {"alias_declaration", "using_declaration"}:
            for child in node.children:
                if child.type in {"type_identifier", "identifier"}:
                    return self._node_text(child, source_bytes).strip()

        if node.type in {"class_specifier", "struct_specifier", "enum_specifier"}:
            for child in node.children:
                if child.type in {"type_identifier", "identifier"}:
                    return self._node_text(child, source_bytes).strip()

        return ""

    @staticmethod
    def _extract_modifiers(signature: str) -> List[str]:
        tokens = set(re.findall(r"\b\w+\b", signature))
        return [modifier for modifier in _MODIFIER_TOKENS if modifier in tokens]

    @staticmethod
    def _node_text(node: Any, source_bytes: bytes) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _iter_nodes(self, root: Any) -> Iterable[Any]:
        stack: List[Any] = [root]
        while stack:
            current = stack.pop()
            yield current
            for child in reversed(current.children):
                stack.append(child)


def extract_functions_from_cpp_file(file_path: str) -> List[Dict[str, Any]]:
    """Compatibility helper used by existing callers."""
    project_map = CppParser().parse_file(file_path)
    functions = project_map.get("functions", {})
    if isinstance(functions, dict):
        return list(functions.values())
    if not isinstance(functions, list):
        return []
    return functions