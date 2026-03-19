"""
AST-based detection of legacy C++ patterns requiring modernization.

This module provides abstracted AST analysis for identifying patterns such as raw
pointers, manual memory management, printf usage, and outdated language constructs.
The architecture decouples pattern detection from the underlying parser implementation
via the ASTNode wrapper abstraction.

Pattern Registry: Patterns are defined in the PATTERN_REGISTRY and can be
selectively enabled/disabled via configuration.

Thread Safety: Instances are NOT thread-safe. Do not share a detector across threads.
For concurrent usage, create separate instances per thread or use external locking.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

from core.parser import CppParser

_log = logging.getLogger(__name__)


class PatternName(Enum):
    """Enumeration of all detectable legacy C++ patterns."""

    RAW_NEW = "raw_new"
    RAW_DELETE = "raw_delete"
    PRINTF_USAGE = "printf_usage"
    RAW_POINTER = "raw_pointer"
    MALLOC_USAGE = "malloc_usage"
    FREE_USAGE = "free_usage"
    NULL_MACRO = "null_macro"
    C_STYLE_CAST = "c_style_cast"
    INDEX_LOOP = "index_loop"
    CHAR_POINTER = "char_pointer"
    MEMCPY_USAGE = "memcpy_usage"
    AUTO_PTR_USAGE = "auto_ptr_usage"
    THROW_SPEC = "throw_spec"

    def __str__(self) -> str:
        return self.value


class ASTNodeType(Enum):
    """Hard-coded tree-sitter node types for C++."""

    FUNCTION_DEFINITION = "function_definition"
    NEW_EXPRESSION = "new_expression"
    DELETE_EXPRESSION = "delete_expression"
    POINTER_DECLARATOR = "pointer_declarator"
    PARAMETER_DECLARATION = "parameter_declaration"
    DECLARATION = "declaration"
    CAST_EXPRESSION = "cast_expression"
    C_STYLE_CAST_EXPRESSION = "c_style_cast_expression"
    FOR_STATEMENT = "for_statement"
    CALL_EXPRESSION = "call_expression"
    IDENTIFIER = "identifier"
    TYPE_IDENTIFIER = "type_identifier"
    QUALIFIED_IDENTIFIER = "qualified_identifier"
    SCOPED_IDENTIFIER = "scoped_identifier"
    NOEXCEPT = "noexcept"
    THROW_SPECIFIER = "throw_specifier"
    DYNAMIC_EXCEPTION_SPECIFICATION = "dynamic_exception_specification"

    def __str__(self) -> str:
        return self.value


class ASTNode:
    """
    Abstraction layer over tree-sitter nodes.

    Decouples pattern detection from the underlying parser implementation.
    If the parser changes (e.g., from tree-sitter to Clang AST), only this
    wrapper and the parser integration need to be updated.
    """

    def __init__(self, node: Any) -> None:
        """
        Initialize the wrapper.

        Args:
            node: A tree-sitter node object.
        """
        self._node = node

    @property
    def node_type(self) -> str:
        """Return the node type (e.g., 'function_definition')."""
        return getattr(self._node, "type", "")

    @property
    def start_byte(self) -> int:
        """Return the starting byte offset in the source."""
        return getattr(self._node, "start_byte", 0)

    @property
    def end_byte(self) -> int:
        """Return the ending byte offset in the source."""
        return getattr(self._node, "end_byte", 0)

    @property
    def start_point(self) -> tuple[int, int]:
        """Return (row, col) of the node's starting position."""
        return getattr(self._node, "start_point", (0, 0))

    @property
    def end_point(self) -> tuple[int, int]:
        """Return (row, col) of the node's ending position."""
        return getattr(self._node, "end_point", (0, 0))

    @property
    def parent(self) -> Optional[ASTNode]:
        """Return the parent node, or None if this is the root."""
        parent_node = getattr(self._node, "parent", None)
        return ASTNode(parent_node) if parent_node is not None else None

    @property
    def children(self) -> List[ASTNode]:
        """Return all child nodes."""
        raw_children = getattr(self._node, "children", ())
        return [ASTNode(child) for child in raw_children]

    def child_by_field_name(self, field_name: str) -> Optional[ASTNode]:
        """
        Retrieve a child node by field name (tree-sitter parser API).

        Args:
            field_name: The field name to query (e.g., 'initializer', 'condition').

        Returns:
            The child node if found, else None.
        """
        method = getattr(self._node, "child_by_field_name", None)
        if method is None:
            return None
        child = method(field_name)
        return ASTNode(child) if child is not None else None

    def get_text(self, source_bytes: bytes) -> str:
        """
        Extract text corresponding to this node from the source.

        Args:
            source_bytes: The full source code as bytes (UTF-8).

        Returns:
            The text of this node, or "" if indices are invalid.
        """
        start = self.start_byte
        end = self.end_byte
        if start < 0 or end < 0 or start > end or end > len(source_bytes):
            return ""
        try:
            return source_bytes[start:end].decode("utf-8", errors="replace")
        except (UnicodeDecodeError, Exception):
            _log.warning(f"Failed to decode node text at bytes {start}:{end}")
            return ""

    def ancestor_of_type(self, target_type: str) -> Optional[ASTNode]:
        """
        Find the nearest ancestor of a given type.

        Args:
            target_type: The node type to search for (e.g., 'declaration').

        Returns:
            The ancestor node if found, else None.
        """
        current = self.parent
        while current is not None:
            if current.node_type == target_type:
                return current
            current = current.parent
        return None


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for pattern detection behavior."""

    # Patterns to detector; if empty, all patterns are enabled.
    enabled_patterns: Set[PatternName] = field(default_factory=lambda: set(PatternName))

    # Whether to cache results by function signature hash.
    enable_cache: bool = True

    # Enable debug logging of detected patterns.
    debug: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.enabled_patterns, set):
            object.__setattr__(self, "enabled_patterns", set(self.enabled_patterns))

    def is_pattern_enabled(self, pattern: PatternName) -> bool:
        """Check if a pattern is enabled."""
        if not self.enabled_patterns:
            return True
        return pattern in self.enabled_patterns


@dataclass
class DetectionResult:
    """Result of pattern detection for a function."""

    counts: Dict[str, int]  # Pattern name -> count
    detected: List[str]  # List of pattern names with count > 0
    locations: Dict[str, List[Dict[str, int]]]  # Pattern name -> list of (line, col) locations
    should_modernize: bool  # Whether any non-zero pattern was found


class ASTModernizationDetector:
    """
    Detects legacy C++ patterns that should be modernized.

    Thread Safety: Instances are NOT thread-safe due to shared parser state.
    Create separate instances for concurrent analysis or use external locking.
    """

    def __init__(
        self, parser: Optional[CppParser] = None, config: Optional[DetectionConfig] = None
    ) -> None:
        """
        Initialize the detector.

        Args:
            parser: Optional CppParser instance; defaults to new CppParser().
            config: Optional DetectionConfig; defaults to detect all patterns.
        """
        self.parser = parser or CppParser()
        self.config = config or DetectionConfig()
        self._cache: Dict[str, DetectionResult] = {}

    def get_function_ast_node(self, function_source: str) -> Optional[ASTNode]:
        """
        Parse a function source and return its AST root node.

        Attempts to locate a function_definition node; if not found, returns the root.

        Args:
            function_source: C++ source code of a function.

        Returns:
            An ASTNode wrapping the function definition, or root if not found.
        """
        source_bytes = function_source.encode("utf-8")
        tree = self.parser.parse_bytes(source_bytes)

        # Prefer a concrete function_definition node if present.
        for node in self._iter_ast(ASTNode(tree.root_node)):
            if node.node_type == ASTNodeType.FUNCTION_DEFINITION.value:
                return node

        # Fallback to root node when function_definition is not found.
        return ASTNode(tree.root_node)

    def detect_legacy_patterns(
        self, function_node: Optional[ASTNode], source_bytes: bytes
    ) -> DetectionResult:
        """
        Detect legacy C++ patterns in a function's AST.

        If caching is enabled and the function was previously analyzed, returns cached result.
        Gracefully handles None input and empty source by returning empty detection.

        Args:
            function_node: The AST node of the function to analyze, or None.
            source_bytes: The raw source code as UTF-8 bytes.

        Returns:
            DetectionResult with pattern counts, detected patterns, locations, and should_modernize flag.
        """
        # Graceful handling of None input
        if function_node is None or not source_bytes:
            empty_result = self._empty_result()
            if self.config.debug:
                _log.debug("Returning empty detection: no function node or source")
            return empty_result

        # Check cache (if enabled)
        cache_key = self._compute_function_hash(source_bytes)
        if self.config.enable_cache and cache_key in self._cache:
            if self.config.debug:
                _log.debug(f"Cache hit for function {cache_key[:8]}...")
            return self._cache[cache_key]

        # Initialize pattern buckets
        patterns: Dict[str, int] = {str(p): 0 for p in PatternName}
        locations: Dict[str, List[Dict[str, int]]] = defaultdict(list)

        # Walk AST and detect patterns
        for node in self._iter_ast(function_node):
            node_type = node.node_type

            if not self.config.is_pattern_enabled(PatternName.RAW_NEW):
                pass
            elif node_type == ASTNodeType.NEW_EXPRESSION.value:
                patterns[str(PatternName.RAW_NEW)] += 1
                self._record_location(locations, PatternName.RAW_NEW, node)

            if not self.config.is_pattern_enabled(PatternName.RAW_DELETE):
                pass
            elif node_type == ASTNodeType.DELETE_EXPRESSION.value:
                patterns[str(PatternName.RAW_DELETE)] += 1
                self._record_location(locations, PatternName.RAW_DELETE, node)

            if self.config.is_pattern_enabled(PatternName.RAW_POINTER) or self.config.is_pattern_enabled(
                PatternName.CHAR_POINTER
            ):
                if node_type == ASTNodeType.POINTER_DECLARATOR.value:
                    self._detect_pointers(node, source_bytes, patterns, locations)

            if not self.config.is_pattern_enabled(PatternName.C_STYLE_CAST):
                pass
            elif node_type in {
                ASTNodeType.CAST_EXPRESSION.value,
                ASTNodeType.C_STYLE_CAST_EXPRESSION.value,
            }:
                patterns[str(PatternName.C_STYLE_CAST)] += 1
                self._record_location(locations, PatternName.C_STYLE_CAST, node)

            if not self.config.is_pattern_enabled(PatternName.INDEX_LOOP):
                pass
            elif node_type == ASTNodeType.FOR_STATEMENT.value:
                if self._is_index_based_for_loop(node, source_bytes):
                    patterns[str(PatternName.INDEX_LOOP)] += 1
                    self._record_location(locations, PatternName.INDEX_LOOP, node)

            if self.config.is_pattern_enabled(PatternName.PRINTF_USAGE) or self.config.is_pattern_enabled(
                PatternName.MALLOC_USAGE
            ) or self.config.is_pattern_enabled(PatternName.FREE_USAGE) or self.config.is_pattern_enabled(
                PatternName.MEMCPY_USAGE
            ):
                if node_type == ASTNodeType.CALL_EXPRESSION.value:
                    self._detect_function_calls(node, source_bytes, patterns, locations)

            if not self.config.is_pattern_enabled(PatternName.NULL_MACRO):
                pass
            elif node_type == ASTNodeType.IDENTIFIER.value:
                node_text = node.get_text(source_bytes)
                if node_text == "NULL":
                    patterns[str(PatternName.NULL_MACRO)] += 1
                    self._record_location(locations, PatternName.NULL_MACRO, node)

            if not self.config.is_pattern_enabled(PatternName.AUTO_PTR_USAGE):
                pass
            elif node_type in {
                ASTNodeType.TYPE_IDENTIFIER.value,
                ASTNodeType.QUALIFIED_IDENTIFIER.value,
                ASTNodeType.SCOPED_IDENTIFIER.value,
            }:
                node_text = node.get_text(source_bytes)
                if "std::auto_ptr" in node_text or "auto_ptr" in node_text:
                    patterns[str(PatternName.AUTO_PTR_USAGE)] += 1
                    self._record_location(locations, PatternName.AUTO_PTR_USAGE, node)

            if not self.config.is_pattern_enabled(PatternName.THROW_SPEC):
                pass
            elif node_type in {
                ASTNodeType.NOEXCEPT.value,
                ASTNodeType.THROW_SPECIFIER.value,
                ASTNodeType.DYNAMIC_EXCEPTION_SPECIFICATION.value,
            }:
                self._detect_throw_spec(node, source_bytes, patterns, locations)

        detected = [name for name, count in patterns.items() if count > 0]
        should_modernize = len(detected) > 0

        result = DetectionResult(
            counts=patterns,
            detected=detected,
            locations={name: rows for name, rows in locations.items() if rows},
            should_modernize=should_modernize,
        )

        # Cache the result
        if self.config.enable_cache:
            self._cache[cache_key] = result

        if self.config.debug:
            _log.debug(f"Detected {len(detected)} pattern(s): {detected}")

        return result

    def should_modernize(self, function_node: Optional[ASTNode], source_bytes: bytes) -> bool:
        """
        Quickly determine if a function should be modernized.

        Returns True if any pattern is detected, False otherwise.

        Args:
            function_node: The AST node of the function.
            source_bytes: The raw source code as UTF-8 bytes.

        Returns:
            True if modernization is recommended.
        """
        return self.detect_legacy_patterns(function_node, source_bytes).should_modernize

    # ==================== Helper Methods ====================

    def _iter_ast(self, root: Optional[ASTNode]) -> Iterable[ASTNode]:
        """
        Iterate over all nodes in the AST using depth-first traversal.

        Args:
            root: The root node to start traversal from.

        Yields:
            ASTNode instances in depth-first order.
        """
        if root is None:
            return
        stack: List[ASTNode] = [root]
        while stack:
            current = stack.pop()
            yield current
            for child in reversed(current.children):
                stack.append(child)

    def _record_location(
        self,
        bucket: Dict[str, List[Dict[str, int]]],
        pattern: PatternName,
        node: ASTNode,
    ) -> None:
        """
        Record the location of a detected pattern.

        Args:
            bucket: The location dictionary to append to.
            pattern: The pattern name.
            node: The AST node where the pattern was detected.
        """
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        bucket[str(pattern)].append(
            {
                "start_line": int(start_row) + 1,
                "start_col": int(start_col) + 1,
                "end_line": int(end_row) + 1,
                "end_col": int(end_col) + 1,
            }
        )

    def _detect_pointers(
        self,
        node: ASTNode,
        source_bytes: bytes,
        patterns: Dict[str, int],
        locations: Dict[str, List[Dict[str, int]]],
    ) -> None:
        """
        Detect raw and char pointers in pointer declarations.

        Excludes function parameters and const-qualified declarations.

        Args:
            node: The pointer_declarator node.
            source_bytes: The source code.
            patterns: Pattern count dictionary to update.
            locations: Location dictionary to update.
        """
        decl_text = self._pointer_declaration_text(node, source_bytes)
        lowered = decl_text.lower()
        is_parameter = node.ancestor_of_type(ASTNodeType.PARAMETER_DECLARATION.value) is not None
        is_local_decl = node.ancestor_of_type(ASTNodeType.DECLARATION.value) is not None
        is_const_qualified = bool(re.search(r"\bconst\b", lowered))

        # Count only mutable local pointers as raw pointers; parameters and
        # const-qualified declarations are intentionally excluded.
        if is_local_decl and not is_parameter and not is_const_qualified:
            patterns[str(PatternName.RAW_POINTER)] += 1
            self._record_location(locations, PatternName.RAW_POINTER, node)

        if is_local_decl and not is_parameter and "char" in lowered and not is_const_qualified:
            patterns[str(PatternName.CHAR_POINTER)] += 1
            self._record_location(locations, PatternName.CHAR_POINTER, node)

    def _detect_function_calls(
        self,
        node: ASTNode,
        source_bytes: bytes,
        patterns: Dict[str, int],
        locations: Dict[str, List[Dict[str, int]]],
    ) -> None:
        """
        Detect function calls to printf, malloc, free, and memcpy.

        Uses exact function name matching (not substring).

        Args:
            node: The call_expression node.
            source_bytes: The source code.
            patterns: Pattern count dictionary to update.
            locations: Location dictionary to update.
        """
        func_name = self._extract_function_name(node, source_bytes)

        if self.config.is_pattern_enabled(PatternName.PRINTF_USAGE) and func_name == "printf":
            patterns[str(PatternName.PRINTF_USAGE)] += 1
            self._record_location(locations, PatternName.PRINTF_USAGE, node)

        if self.config.is_pattern_enabled(PatternName.MALLOC_USAGE) and func_name == "malloc":
            patterns[str(PatternName.MALLOC_USAGE)] += 1
            self._record_location(locations, PatternName.MALLOC_USAGE, node)

        if self.config.is_pattern_enabled(PatternName.FREE_USAGE) and func_name == "free":
            patterns[str(PatternName.FREE_USAGE)] += 1
            self._record_location(locations, PatternName.FREE_USAGE, node)

        if self.config.is_pattern_enabled(PatternName.MEMCPY_USAGE) and func_name == "memcpy":
            patterns[str(PatternName.MEMCPY_USAGE)] += 1
            self._record_location(locations, PatternName.MEMCPY_USAGE, node)

    def _detect_throw_spec(
        self,
        node: ASTNode,
        source_bytes: bytes,
        patterns: Dict[str, int],
        locations: Dict[str, List[Dict[str, int]]],
    ) -> None:
        """
        Detect legacy dynamic exception specifications.

        Matches throw() and throw(...) constructs.

        Args:
            node: The throw specifier node.
            source_bytes: The source code.
            patterns: Pattern count dictionary to update.
            locations: Location dictionary to update.
        """
        node_text = node.get_text(source_bytes)
        # Dynamic exception specifications contain "throw" followed by parentheses
        if re.search(r"\bthrow\s*\(", node_text):
            patterns[str(PatternName.THROW_SPEC)] += 1
            self._record_location(locations, PatternName.THROW_SPEC, node)

    def _extract_function_name(self, call_expr: ASTNode, source_bytes: bytes) -> str:
        """
        Extract the exact function name from a call_expression.

        Attempts to find the 'function' child field; falls back to heuristics.

        Args:
            call_expr: The call_expression node.
            source_bytes: The source code.

        Returns:
            The function name, or empty string if extraction fails.
        """
        func_child = call_expr.child_by_field_name("function")
        if func_child is not None:
            func_text = func_child.get_text(source_bytes)
            # Extract the bare function name (last component after :: or .)
            if "::" in func_text:
                func_text = func_text.split("::")[-1]
            if "." in func_text:
                func_text = func_text.split(".")[-1]
            return func_text.strip()

        # Fallback: try to parse from the call_expression text
        call_text = call_expr.get_text(source_bytes)
        match = re.match(r"(?:\w+::)*(\w+)\s*\(", call_text)
        if match:
            return match.group(1)

        return ""

    def _pointer_declaration_text(self, pointer_node: ASTNode, source_bytes: bytes) -> str:
        """
        Extract the declaration text for a pointer declarator.

        Returns the full declaration statement or the node text.

        Args:
            pointer_node: The pointer_declarator node.
            source_bytes: The source code.

        Returns:
            The declaration text.
        """
        param_decl = pointer_node.ancestor_of_type(ASTNodeType.PARAMETER_DECLARATION.value)
        if param_decl is not None:
            return param_decl.get_text(source_bytes)

        decl = pointer_node.ancestor_of_type(ASTNodeType.DECLARATION.value)
        if decl is not None:
            return decl.get_text(source_bytes)

        return pointer_node.get_text(source_bytes)

    def _is_index_based_for_loop(self, for_node: ASTNode, source_bytes: bytes) -> bool:
        """
        Detect index-based for loops (e.g., for (int i = 0; i < n; ++i)).

        Limitations:
        - Assumes loop variable initialized to 0.
        - Requires standard loop structure (init, condition, update).
        - Does not detect if-based loops over containers.
        - Does not handle variables declared outside the loop.

        Args:
            for_node: The for_statement node.
            source_bytes: The source code.

        Returns:
            True if the loop appears to be index-based.
        """
        init = for_node.child_by_field_name("initializer")
        condition = for_node.child_by_field_name("condition")
        update = for_node.child_by_field_name("update")

        if init is None or condition is None or update is None:
            return False

        init_text = init.get_text(source_bytes)
        cond_text = condition.get_text(source_bytes)
        update_text = update.get_text(source_bytes)

        # Match initialization: type var = 0 (supports int, long, size_t, etc.)
        init_match = re.search(
            r"\b(?:int|long|short|size_t|std::size_t|unsigned|std::ptrdiff_t)\b[^;=]*\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\b",
            init_text,
        )
        if not init_match:
            return False

        index_var = init_match.group(1)

        # Match condition: var < n or var <= n
        cond_ok = re.search(rf"\b{re.escape(index_var)}\b\s*(?:<|<=)\s*", cond_text) is not None

        # Match update: ++var, var++, or var += 1
        update_ok = re.search(
            rf"(?:\+\+\s*{re.escape(index_var)}|{re.escape(index_var)}\s*\+\+|{re.escape(index_var)}\s*\+=\s*1)",
            update_text,
        ) is not None

        return cond_ok and update_ok

    def _compute_function_hash(self, source_bytes: bytes) -> str:
        """
        Compute a stable hash of the function source.

        Used for caching results. Includes only the first 4096 bytes to avoid
        hashing entire large functions.

        Args:
            source_bytes: The source code.

        Returns:
            A hex string hash of the source.
        """
        sample = source_bytes[:4096]
        return hashlib.sha256(sample).hexdigest()

    def _empty_result(self) -> DetectionResult:
        """
        Create an empty detection result with no patterns found.

        Returns:
            A DetectionResult with zero counts and no detected patterns.
        """
        return DetectionResult(
            counts={str(p): 0 for p in PatternName},
            detected=[],
            locations={},
            should_modernize=False,
        )
