"""
Parser utilities for the Code Modernization Engine.

Provides helpers to parse C++ code and extract function definitions using Tree-sitter.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any
from tree_sitter import Parser, Language


def _create_cpp_parser() -> Parser:
    """
    Creates and returns a Tree-sitter parser configured for C++ code.

    Returns:
        Parser: Tree-sitter parser for C++.

    Raises:
        RuntimeError: If the C++ grammar package is not installed.
    """
    try:
        import tree_sitter_cpp  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "C++ grammar package 'tree-sitter-cpp' is not installed. "
            "Please install it with 'pip install tree-sitter-cpp'."
        ) from exc
    
    cpp_language_handle = tree_sitter_cpp.language()
    cpp_language = Language(cpp_language_handle)
    parser = Parser(cpp_language)
    return parser


def _read_file_text(file_path: str) -> str:
    """
    Reads the entire contents of a text file as UTF-8 text.

    Args:
        file_path: Path to the file.

    Returns:
        str: File contents as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file_handle:
        file_text = file_handle.read()
    return file_text


def _collect_function_definitions(root_node: Any) -> List[Any]:
    """
    Traverses the AST starting at the root and collects all C++ function definition nodes.

    Args:
        root_node: Root node of the AST.

    Returns:
        List[Any]: List of function definition nodes.
    """
    function_nodes: List[Any] = []
    stack: List[Any] = [root_node]

    while stack:
        node = stack.pop()

        if node.type == "function_definition":
            function_nodes.append(node)

        for child in node.children:
            stack.append(child)

    return function_nodes


def _extract_function_signature(node: Any, source_text: str) -> str:
    """
    Extract a best-effort textual function signature, including return type and parameters.

    Heuristic: Extracts text from the start of the definition node up to the body block.

    Args:
        node: The function_definition AST node.
        source_text: The full source code string.

    Returns:
        str: The extracted signature text.
    """
    body_node = node.child_by_field_name("body")
    start_byte = node.start_byte
    
    if body_node is not None:
        end_byte = body_node.start_byte
    else:
        end_byte = node.end_byte

    source_bytes = source_text.encode("utf-8")
    header_bytes = source_bytes[start_byte:end_byte]
    signature_text = header_bytes.decode("utf-8", errors="replace").strip()
    return signature_text


def _extract_leading_comments(node: Any, source_text: str) -> str:
    """
    Extract any docstrings or comments that appear immediately before a function.

    Args:
        node: The function_definition AST node.
        source_text: The full source code string.

    Returns:
        str: Contiguous block of leading comment lines.
    """
    lines = source_text.splitlines()
    start_row, _ = node.start_point

    comment_lines: List[str] = []
    current_row = start_row - 1

    def _looks_like_comment_or_blank(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        return (stripped.startswith("//") or 
                stripped.startswith("/*") or 
                stripped.startswith("*") or 
                stripped.endswith("*/"))

    while current_row >= 0:
        candidate = lines[current_row]
        if not _looks_like_comment_or_blank(candidate):
            break
        comment_lines.append(candidate)
        current_row -= 1

    comment_lines.reverse()
    return "\n".join(comment_lines).strip()


def _extract_identifier_from_function(node: Any, source_bytes: bytes) -> str:
    """
    Recovers the function name (identifier) from a function_definition node.

    Args:
        node: The function_definition AST node.
        source_bytes: The source code encoded as bytes.

    Returns:
        str: The function identifier or '<anonymous_function>'.
    """
    declarator = node.child_by_field_name("declarator")
    # If declarator is present, search for identifier within it
    if declarator is not None:
        # Only return the first identifier child of declarator
        for child in declarator.children:
            if child.type == "identifier":
                start_byte = child.start_byte
                end_byte = child.end_byte
                return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
        # Fallback: search recursively in declarator
        stack: List[Any] = [declarator]
        while stack:
            current = stack.pop()
            if current.type == "identifier":
                start_byte = current.start_byte
                end_byte = current.end_byte
                return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
            for child in current.children:
                stack.append(child)
    # Fallback: search for identifier directly under node
    stack: List[Any] = [node]
    while stack:
        current = stack.pop()
        if current.type == "identifier":
            start_byte = current.start_byte
            end_byte = current.end_byte
            return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
        for child in current.children:
            stack.append(child)
    return "<anonymous_function>"


def _extract_function_body_source(node: Any, source_text: str) -> str:
    """
    Extract the source code that corresponds to the body of a function.

    Args:
        node: The function_definition AST node.
        source_text: The full source code string.

    Returns:
        str: The body source code including braces.
    """
    body_node = node.child_by_field_name("body")

    if body_node is None:
        return ""

    start_byte = body_node.start_byte
    end_byte = body_node.end_byte

    source_bytes = source_text.encode("utf-8")
    body_bytes = source_bytes[start_byte:end_byte]
    return body_bytes.decode("utf-8", errors="replace")


def _extract_call_target_name(call_node: Any, source_bytes: bytes) -> str:
    """
    Finds the name of the function being called inside a call_expression node.

    Args:
        call_node: The call_expression AST node.
        source_bytes: The source code encoded as bytes.

    Returns:
        str: The callee name or '<anonymous_call>'.
    """
    function_target = call_node.child_by_field_name("function")
    search_root = function_target if function_target is not None else call_node
    stack: List[Any] = [search_root]

    while stack:
        current = stack.pop()

        if current.type == "identifier":
            start_byte = current.start_byte
            end_byte = current.end_byte
            return source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")

        for child in current.children:
            stack.append(child)

    return "<anonymous_call>"


def _collect_function_calls(func_node: Any, source_bytes: bytes) -> List[str]:
    """
    Collects the names of all functions called within a function body.

    Args:
        func_node: The function_definition AST node.
        source_bytes: The source code encoded as bytes.

    Returns:
        List[str]: Distinct names of functions invoked.
    """
    body_node = func_node.child_by_field_name("body")
    if body_node is None:
        return []

    calls: List[str] = []
    stack: List[Any] = [body_node]

    while stack:
        node = stack.pop()

        if node.type == "call_expression":
            callee_name = _extract_call_target_name(node, source_bytes)
            if callee_name and callee_name != "<anonymous_call>" and callee_name not in calls:
                calls.append(callee_name)

        for child in node.children:
            stack.append(child)

    return calls


def extract_functions_from_cpp_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a C++ file and extract comprehensive function metadata.

    Args:
        file_path: Absolute path to the .cpp source file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing function metadata:
            - name (str): Function identifier.
            - body (str): Implementation source code.
            - calls (List[str]): Call graph dependencies.
            - start_byte (int): Starting byte offset.
            - end_byte (int): Ending byte offset.
            - signature (str): Textual function header.
            - comments (str): Leading documentation comments.
            - line_numbers (dict): 1-based start and end lines.
    
    Raises:
        FileNotFoundError: If the file_path is invalid.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"C++ file not found: {file_path}")

    parser = _create_cpp_parser()
    source_text = _read_file_text(file_path)
    source_bytes = source_text.encode("utf-8")

    tree = parser.parse(source_bytes)
    root_node = tree.root_node
    function_nodes = _collect_function_definitions(root_node)

    results: List[Dict[str, Any]] = []

    for func_node in function_nodes:
        name = _extract_identifier_from_function(func_node, source_bytes)
        body = _extract_function_body_source(func_node, source_text)
        calls = _collect_function_calls(func_node, source_bytes)
        start_byte = func_node.start_byte
        end_byte = func_node.end_byte
        signature = _extract_function_signature(func_node, source_text)
        comments = _extract_leading_comments(func_node, source_text)
        start_row, _ = func_node.start_point
        end_row, _ = func_node.end_point
        
        results.append({
            "name": name,
            "body": body,
            "calls": calls,
            "start_byte": start_byte,
            "end_byte": end_byte,
            "signature": signature,
            "comments": comments,
            "line_numbers": {"start": start_row + 1, "end": end_row + 1},
        })

    return results


class CppParser:
    """
    Interface for C++ source code parsing and metadata extraction.
    """

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a C++ source file to extract function definitions.

        Args:
            file_path: Path to the target file.

        Returns:
            List[Dict[str, Any]]: Extracted function metadata.
        """
        return extract_functions_from_cpp_file(file_path)


if __name__ == "__main__":
    import sys
    from pprint import pprint

    if len(sys.argv) != 2:
        print("Usage: python -m core.parser <path-to-file.cpp>")
        sys.exit(1)

    cpp_path = sys.argv[1]
    functions_info = extract_functions_from_cpp_file(cpp_path)
    pprint(functions_info)