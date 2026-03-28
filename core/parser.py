from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)


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

_MODIFIER_TOKENS = ("virtual", "static", "inline", "constexpr", "consteval", "noexcept", "override", "final")


ProjectMap = Dict[str, Any]
# Configurable via CPP_PARSER_MAX_AST_NODES env var; default 200 000 which handles
# typical large files without Python recursion concerns.
MAX_AST_NODES: int = max(1_000, int(os.environ.get("CPP_PARSER_MAX_AST_NODES", "200000")))

# Thread-local storage for parser instances — tree-sitter parsers are NOT thread-safe.
_thread_local: threading.local = threading.local()

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
    "<span>": {"std::span", "span"},
    "<expected>": {"std::expected", "expected"},
    "<format>": {"std::format", "format"},
    "<print>": {"std::print", "print"},
    "<ranges>": {"std::ranges", "ranges", "views"},
}


_TEMPLATE_SYMBOL_BASES: Dict[str, Set[str]] = {
    "<vector>": {"vector"},
    "<string>": {"string", "string_view"},
    "<map>": {"map", "multimap"},
    "<unordered_map>": {"unordered_map", "unordered_multimap"},
    "<set>": {"set", "multiset"},
    "<unordered_set>": {"unordered_set", "unordered_multiset"},
    "<optional>": {"optional"},
    "<variant>": {"variant"},
    "<tuple>": {"tuple"},
    "<memory>": {"unique_ptr", "shared_ptr", "weak_ptr"},
    "<span>": {"span"},
    "<expected>": {"expected", "unexpected"},
    "<format>": {"format", "vformat"},
    "<print>": {"print", "println"},
    "<ranges>": {"ranges", "views"},
}


class CppParser:
    """High-fidelity C++ semantic extraction engine for modernization workflows."""

    def __init__(self) -> None:
        self._parser = self._create_cpp_parser()
        self._last_project_map: Optional[ProjectMap] = None
        self._workspace_root: Optional[Path] = None
        self._current_file_path: str = ""

    @staticmethod
    def _create_cpp_parser() -> Parser:
        """
        Create a parser configured for C++ with tree-sitter v0.21+ compatibility.

        A per-thread instance is cached in thread-local storage because tree-sitter
        Parser objects are NOT thread-safe; sharing one instance across threads can
        corrupt internal parser state.
        """
        existing: Parser | None = getattr(_thread_local, "cpp_parser", None)
        if existing is not None:
            return existing

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
            parser.set_language(cpp_language)  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Unsupported tree-sitter Parser API for setting language")

        _thread_local.cpp_parser = parser
        return parser

    def parse_file(
        self,
        file_path: str | Path,
        workspace_root: str | Path | None = None,
    ) -> ProjectMap:
        """Parse a C++ source file into a Janus-style ProjectMap.

        If *workspace_root* is given it is stored on the instance so that
        downstream helpers can use it to resolve ``#include`` paths to
        project-local headers.
        """

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"C++ file not found: {path}")
        if workspace_root is not None:
            self._workspace_root = Path(workspace_root)
        self._current_file_path = str(path)
        return self.parse_string(path.read_text(encoding="utf-8"), source_file=str(path))

    def parse_string(self, source_text: str, source_file: str = "") -> ProjectMap:
        """Parse C++ source text into a Janus-style ProjectMap."""

        source_bytes = source_text.encode("utf-8")
        try:
            tree = self._parser.parse(source_bytes)
        except Exception:
            project_map = {
                "functions": {},
                "type_definitions": {},
                "dependency_order": [],
                "include_requirements": {},
                "headers": [],
                "types": [],
                "global_context": {},
                "global_variables": [],
            }
            self._last_project_map = project_map
            return project_map

        project_map = self._collect_semantic_map_single_pass(
            tree.root_node,
            source_text,
            source_bytes,
            source_file=source_file or self._current_file_path,
        )
        project_map["module_imports"] = detect_module_imports(source_text)
        self._last_project_map = project_map
        return project_map

    def parse_bytes(self, source_bytes: bytes) -> Any:
        """Parse C++ bytes and return the raw tree-sitter tree object."""
        return self._parser.parse(source_bytes)

    def iter_nodes(self, root: Any) -> Iterable[Any]:
        """Public node traversal helper for consumers that need AST walking."""
        return self._iter_nodes(root)

    def node_text(self, node: Any, source_bytes: bytes) -> str:
        """Public node text extraction helper."""
        return self._node_text(node, source_bytes)

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
        if not isinstance(functions, dict):
            raise KeyError(f"Function not found in ProjectMap: {fqn}")

        if fqn not in functions:
            # Backward-compatible fallback: allow lookup by legacy non-unique FQN.
            legacy_matches = [
                key
                for key, meta in functions.items()
                if str(meta.get("fqn") or "") == fqn
            ]
            if len(legacy_matches) == 1:
                fqn = legacy_matches[0]
            elif len(legacy_matches) > 1:
                raise KeyError(
                    f"Function FQN '{fqn}' is ambiguous across overloads; use unique_fqn."
                )
            else:
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

    @staticmethod
    def _compute_signature_hash(parameters: List[Dict[str, Any]]) -> str:
        """Return a short deterministic hash of function parameter types."""
        type_str = ",".join(str(p.get("type") or "") for p in parameters)
        return hashlib.md5(type_str.encode("utf-8")).hexdigest()[:8]

    def _process_ast_node(
        self,
        node: Any,
        scope_stack: List[str],
        source_text: str,
        source_bytes: bytes,
        line_starts: List[int],
        source_file: str,
        functions: List[Dict[str, Any]],
        headers: List[str],
        types: List[Dict[str, Any]],
        global_context: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[str]:
        """Process a single AST node during the single-pass traversal.

        Mutates *functions*, *headers*, *types*, and *global_context* in place.
        Returns the scope name that should be pushed onto *scope_stack* (or None).
        """
        if node.type == "function_definition":
            functions.append(
                self._build_function_record(
                    node=node,
                    scope_stack=scope_stack,
                    source_text=source_text,
                    source_bytes=source_bytes,
                    line_starts=line_starts,
                    source_file=source_file,
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

        return self._scope_name(node, source_bytes)

    def _collect_semantic_map_single_pass(
        self,
        root_node: Any,
        source_text: str,
        source_bytes: bytes,
        source_file: str = "",
    ) -> ProjectMap:
        """Traverse the AST once while maintaining a scope stack for FQN construction."""

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

            scope_name = self._process_ast_node(
                node, scope_stack, source_text, source_bytes,
                line_starts, source_file, functions, headers, types, global_context,
            )
            pushed_now = scope_name is not None
            if pushed_now:
                scope_stack.append(scope_name)

            stack.append((node, 1, pushed_now))
            for child in reversed(node.children):
                stack.append((child, 0, False))

        global_variables = self._collect_global_variables(root_node, source_bytes, line_starts)
        return self._build_project_map(functions, types, headers, global_context, global_variables)

    def _build_project_map(
        self,
        functions: List[Dict[str, Any]],
        types: List[Dict[str, Any]],
        headers: List[str],
        global_context: Dict[str, List[Dict[str, Any]]],
        global_variables: List[Dict[str, Any]],
    ) -> ProjectMap:
        function_map: Dict[str, Dict[str, Any]] = {}
        type_definitions: Dict[str, str] = {}

        for t in types:
            type_name = str(t.get("name") or "")
            if type_name and type_name not in type_definitions:
                type_definitions[type_name] = str(t.get("source_code") or "")

        all_function_ids = {
            str(f.get("unique_fqn") or f.get("fqn") or "")
            for f in functions
            if f.get("unique_fqn") or f.get("fqn")
        }
        name_to_function_ids: Dict[str, List[str]] = {}
        legacy_fqn_to_ids: Dict[str, List[str]] = {}
        for f in functions:
            name = str(f.get("name") or "")
            function_id = str(f.get("unique_fqn") or f.get("fqn") or "")
            legacy_fqn = str(f.get("fqn") or "")
            if name and function_id:
                name_to_function_ids.setdefault(name, []).append(function_id)
            if legacy_fqn and function_id:
                legacy_fqn_to_ids.setdefault(legacy_fqn, []).append(function_id)

        inbound: Dict[str, Set[str]] = {function_id: set() for function_id in all_function_ids}

        include_requirements: Dict[str, List[str]] = {}

        for f in functions:
            function_id = str(f.get("unique_fqn") or f.get("fqn") or "")
            if not function_id:
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

                    if normalized in all_function_ids:
                        if normalized != function_id:
                            internal_calls.append(normalized)
                            inbound[normalized].add(function_id)
                        continue

                    if normalized in legacy_fqn_to_ids:
                        for candidate_id in legacy_fqn_to_ids[normalized]:
                            if candidate_id != function_id:
                                internal_calls.append(candidate_id)
                                inbound[candidate_id].add(function_id)
                        continue

                    if normalized in name_to_function_ids:
                        overload_candidates = name_to_function_ids[normalized]
                        for candidate_id in overload_candidates:
                            if candidate_id != function_id:
                                internal_calls.append(candidate_id)
                                inbound[candidate_id].add(function_id)
                        continue
                    if call_display:
                        external_calls.append(call_display)

            internal_calls = sorted(set(internal_calls))
            external_calls = sorted(set(external_calls))

            includes_for_function = self._compute_include_requirements_for_function(f, headers)
            include_requirements[function_id] = includes_for_function

            out_degree = len(internal_calls)
            in_degree = len(inbound.get(function_id, set()))
            is_leaf = out_degree == 0
            is_root = in_degree >= 2 and out_degree <= 1

            merged = dict(f)
            merged["internal_calls"] = internal_calls
            merged["external_calls"] = external_calls
            merged["incoming_calls_count"] = in_degree
            merged["outgoing_calls_count"] = out_degree
            merged["is_leaf"] = is_leaf
            merged["is_root"] = is_root
            function_map[function_id] = merged

        dependency_order = self._compute_modernization_priority(function_map)

        return {
            "functions": function_map,
            "type_definitions": type_definitions,
            "dependency_order": dependency_order,
            "include_requirements": include_requirements,
            "headers": headers,
            "types": types,
            "global_context": global_context,
            "global_variables": global_variables,
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
        call_details = function_meta.get("call_details") or []
        joined_calls = "\n".join(
            str(entry.get("display") or entry.get("name") or "")
            for entry in call_details
            if isinstance(entry, dict)
        )
        text = "\n".join([signature, body, joined_calls])

        required: List[str] = []
        candidate_headers: Set[str] = set()
        for header_line in headers:
            header_name = self._extract_header_name(header_line)
            if header_name:
                candidate_headers.add(header_name)
        candidate_headers.update(_STD_HEADER_SYMBOLS.keys())
        candidate_headers.update(_TEMPLATE_SYMBOL_BASES.keys())

        for header_name in sorted(candidate_headers):
            if not header_name:
                continue

            known_symbols = _STD_HEADER_SYMBOLS.get(header_name)
            if known_symbols:
                if any(self._symbol_in_text(symbol, text) for symbol in known_symbols):
                    required.append(header_name)
                    continue

            template_bases = _TEMPLATE_SYMBOL_BASES.get(header_name)
            if template_bases:
                if any(self._symbol_or_template_use(base_symbol, text) for base_symbol in template_bases):
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
    def _symbol_or_template_use(base_symbol: str, text: str) -> bool:
        """Detect `std::symbol`, `symbol`, and templated uses such as `symbol<T>`.

        The `std::` prefix must have no whitespace between `::` and the symbol
        because `std:: vector` (with a space) is invalid C++.
        """
        if not base_symbol:
            return False
        # `(?:std::)?` — no \s* after :: to reject the invalid `std:: symbol` form.
        plain_pattern = r"(?<!\w)(?:std::)?" + re.escape(base_symbol) + r"(?!\w)"
        template_pattern = r"(?<!\w)(?:std::)?" + re.escape(base_symbol) + r"\s*<"
        return re.search(plain_pattern, text) is not None or re.search(template_pattern, text) is not None

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
        source_file: str = "",
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
        loc = max(1, end_line - start_line + 1)

        calls = self._collect_function_calls(node, source_bytes)
        modifiers = self._extract_modifiers(signature_text)
        parameters = self._extract_structured_parameters(node, source_bytes)
        signature_hash = self._compute_signature_hash(parameters)
        unique_fqn = f"{fqn}#{signature_hash}" if fqn else f"{function_name}#{signature_hash}"
        lower_signature = signature_text.lower()
        lower_body = body_text.lower()
        loops = sum(
            1 for sub in self._iter_nodes(node)
            if sub.type in {"for_statement", "while_statement", "do_statement", "range_based_for_statement"}
        )
        branches = sum(
            1 for sub in self._iter_nodes(node)
            if sub.type in {"if_statement", "switch_statement", "conditional_expression"}
        )
        call_count = len(calls)
        complexity = loops + branches + call_count
        function_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        legacy_patterns = {
            "has_raw_pointer": "*" in signature_text and "const" not in lower_signature,
            "has_printf": "printf" in lower_body,
            "has_malloc": "malloc" in lower_body,
            "has_free": "free" in lower_body,
            "has_null_macro": bool(re.search(r"\bNULL\b", body_text)),
        }

        return {
            "fqn": fqn,
            "unique_fqn": unique_fqn,
            "name": function_name,
            "signature": signature_text,
            "signature_hash": signature_hash,
            "body": body_text,
            "parameters": parameters,
            "call_details": calls,
            "start_byte": ownership_start,
            "end_byte": node.end_byte,
            "line_numbers": {"start": start_line, "end": end_line},
            "loc": loc,
            "is_template": owner_node.type == "template_declaration" or "template<" in lower_signature,
            "modifiers": modifiers,
            "legacy_patterns": legacy_patterns,
            "complexity": complexity,
            "function_hash": function_hash,
            "file_path": source_file,
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
        """Return the unqualified function name (last component of the FQN)."""
        return self._extract_function_qualified_parts(function_node, source_bytes)[-1]

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

            if callee_node.type == "lambda_expression":
                call_info = {"name": "<lambda>", "display": "<lambda>", "kind": "lambda"}
                dedupe_key = (call_info["kind"], call_info["display"])
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    calls.append(call_info)
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
        if callee_node.type == "pointer_expression":
            pointer_text = self._node_text(callee_node, source_bytes).strip()
            # Function pointer calls often appear as (*fp)(...), normalize to fp.
            pointer_name = pointer_text
            pointer_name = re.sub(r"^\(\*", "", pointer_name)
            pointer_name = re.sub(r"\)$", "", pointer_name)
            pointer_name = pointer_name.strip("*() ")
            if pointer_name:
                return {"name": pointer_name, "display": pointer_name, "kind": "function_pointer"}

        if callee_node.type == "lambda_expression":
            return {"name": "<lambda>", "display": "<lambda>", "kind": "lambda"}

        if callee_node.type in {"parenthesized_expression", "subscript_expression"}:
            expr_text = self._node_text(callee_node, source_bytes).strip()
            if expr_text:
                name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", expr_text)
                if name_match:
                    inferred_name = name_match.group(1)
                    kind = "functor" if "[" in expr_text else "function_pointer"
                    return {"name": inferred_name, "display": expr_text, "kind": kind}

        if callee_node.type == "field_expression":
            obj_node = callee_node.child_by_field_name("argument")
            field_node = callee_node.child_by_field_name("field")
            if field_node is not None:
                name = self._node_text(field_node, source_bytes).strip()
                owner = self._node_text(obj_node, source_bytes).strip() if obj_node is not None else ""
                display = f"{owner}.{name}" if owner else name
                if name == "operator()":
                    return {"name": owner or "operator()", "display": display, "kind": "functor"}
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
                    if text == "operator()":
                        parent_text = self._node_text(callee_node, source_bytes).strip()
                        owner_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*operator\s*\(\)", parent_text)
                        owner_name = owner_match.group(1) if owner_match else "operator()"
                        return {"name": owner_name, "display": parent_text or owner_name, "kind": "functor"}
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
        count = 0
        while stack:
            current = stack.pop()
            yield current
            count += 1
            if count > MAX_AST_NODES:
                break
            for child in reversed(current.children):
                stack.append(child)

    def _collect_global_variables(
        self,
        root_node: Any,
        source_bytes: bytes,
        line_starts: List[int],
    ) -> List[Dict[str, Any]]:
        variables: List[Dict[str, Any]] = []
        for child in root_node.children:
            if child.type != "declaration":
                continue
            decl_text = self._node_text(child, source_bytes).strip()
            if not decl_text or "(" in decl_text or ")" in decl_text:
                continue
            names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?=\s*(?:=|;|,))", decl_text)
            if not names:
                continue
            line = self._byte_to_line_number(child.start_byte, line_starts)
            for name in names:
                variables.append(
                    {
                        "name": name,
                        "line": line,
                        "declaration": decl_text,
                    }
                )
        return variables

    def export_ast_graph(self, source_text: str, output_path: str, max_nodes: int = 1000) -> str:
        """Export a simplified AST graph in DOT format for debugging/demo use."""
        try:
            import importlib
            graphviz_module = importlib.import_module("graphviz")
            Digraph = getattr(graphviz_module, "Digraph")
        except ImportError as exc:
            raise RuntimeError("graphviz is required for export_ast_graph") from exc

        source_bytes = source_text.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        dot = Digraph("cpp_ast")

        queue: List[Any] = [tree.root_node]
        seen = 0
        while queue and seen < max_nodes:
            node = queue.pop(0)
            node_id = f"n{node.start_byte}_{node.end_byte}_{seen}"
            dot.node(node_id, label=node.type)
            for child in node.children:
                child_id = f"n{child.start_byte}_{child.end_byte}_{seen}_{child.type}"
                dot.node(child_id, label=child.type)
                dot.edge(node_id, child_id)
                queue.append(child)
            seen += 1

        dot.save(output_path)
        return output_path


def extract_functions_from_cpp_file(file_path: str) -> List[Dict[str, Any]]:
    """Compatibility helper used by existing callers."""
    project_map = CppParser().parse_file(file_path)
    functions = project_map.get("functions", {})
    if isinstance(functions, dict):
        return list(functions.values())
    if not isinstance(functions, list):
        return []
    return functions


# ---------------------------------------------------------------------------
# C++20/23 module-import patterns
# ---------------------------------------------------------------------------
# Matches: `import <header>;`, `import "header";`, `import module.name;`
_CPP20_IMPORT_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:export\s+)?import\s+"
    r"(?:"
    r"(<[^>]+>)"           # import <header>;
    r"|"
    r'("(?:[^"\\]|\\.)*")' # import "header";
    r"|"
    r"([\w.]+)"            # import module_name;
    r")\s*;",
    re.MULTILINE,
)
# Matches: `export module module_name;` and `module module_name;`
_CPP20_MODULE_DECL_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:export\s+)?module\s+([\w.]+)\s*;",
    re.MULTILINE,
)


def detect_module_imports(source_text: str) -> List[Dict[str, Any]]:
    """Return C++20 module-import and module-declaration records found in *source_text*.

    Each entry has the keys:
    - ``kind``: ``"import"`` or ``"module_decl"``
    - ``target``: the imported name / header / module identifier
    - ``line``: 1-based line number
    - ``raw``: the matched source text
    """
    results: List[Dict[str, Any]] = []
    for match in _CPP20_IMPORT_RE.finditer(source_text):
        angle, quoted, named = match.group(1), match.group(2), match.group(3)
        target = (angle or quoted or named or "").strip()
        line = source_text.count("\n", 0, match.start()) + 1
        results.append({"kind": "import", "target": target, "line": line, "raw": match.group(0).strip()})
    for match in _CPP20_MODULE_DECL_RE.finditer(source_text):
        line = source_text.count("\n", 0, match.start()) + 1
        results.append({"kind": "module_decl", "target": match.group(1).strip(), "line": line, "raw": match.group(0).strip()})
    results.sort(key=lambda r: r["line"])
    return results


_LEGACY_PATTERN_SPECS: list[tuple[str, str, re.Pattern[str], str]] = [
    (
        "char_pointer_array",
        "critical",
        re.compile(r"\bchar\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*(\[[^\]]*\])"),
        "char* array usage detected; prefer std::string/std::array/std::span.",
    ),
    (
        "null_macro",
        "major",
        re.compile(r"\bNULL\b"),
        "NULL macro detected; replace with nullptr.",
    ),
    (
        "manual_delete",
        "critical",
        re.compile(r"\bdelete\s*(\[\])?\s*[A-Za-z_][A-Za-z0-9_]*\s*;"),
        "Manual delete detected; prefer std::unique_ptr or stack allocation.",
    ),
]


def detect_legacy_patterns(source_text: str) -> List[Dict[str, Any]]:
    """Detect legacy C/C++ patterns and mark regions for C++23 overhaul."""
    findings: List[Dict[str, Any]] = []
    for pattern_id, severity, pattern_re, message in _LEGACY_PATTERN_SPECS:
        for match in pattern_re.finditer(source_text):
            start = match.start()
            line = source_text.count("\n", 0, start) + 1
            findings.append(
                {
                    "pattern": pattern_id,
                    "severity": severity,
                    "line": line,
                    "match": match.group(0),
                    "message": message,
                    "tag": "C++23 Overhaul",
                }
            )

    # AST-based C-style cast detection to avoid regex false positives such as
    # parenthesized conditions in if/while expressions.
    try:
        parser = CppParser()
        source_bytes = source_text.encode("utf-8")
        tree = parser._parser.parse(source_bytes)
        line_starts = parser._compute_line_start_bytes(source_text)
        cast_node_types = {"cast_expression", "c_style_cast_expression"}

        for node in parser._iter_nodes(tree.root_node):
            if node.type not in cast_node_types:
                continue
            snippet = parser._node_text(node, source_bytes).strip()
            if not snippet:
                continue
            findings.append(
                {
                    "pattern": "c_style_cast",
                    "severity": "major",
                    "line": parser._byte_to_line_number(node.start_byte, line_starts),
                    "match": snippet,
                    "message": "Potential C-style cast detected; prefer static_cast/reinterpret_cast.",
                    "tag": "C++23 Overhaul",
                }
            )
    except Exception as exc:
        # Keep regex-based findings; log so the failure is visible in debug mode.
        logger.debug("AST-based C-style cast detection failed: %s", exc, exc_info=True)

    findings.sort(key=lambda item: (int(item.get("line", 0)), str(item.get("pattern", ""))))
    return findings


def detect_legacy_patterns_from_cpp_file(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"C++ file not found: {path}")
    return detect_legacy_patterns(path.read_text(encoding="utf-8"))