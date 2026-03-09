"""
Parser utilities for the Code Modernization Engine.  # This line explains that this file holds helpers to understand and slice C++ code.
Uses Tree-sitter to build an AST (Abstract Syntax Tree) so the AI can see functions and their bodies.  # This line notes that we rely on Tree-sitter to understand structure, not just text.
"""  # This closing triple quote ends the human-readable description of this module.

from __future__ import annotations  # This import allows some modern typing features to work even on older Python versions.

import os  # This import lets us work with file paths and check whether a given path is a real file.
from typing import List, Dict, Any  # This import lets us describe return types like "list of dictionaries" for clarity.

from tree_sitter import (  # This grouped import brings in the core Tree-sitter classes we need.
    Parser,  # This class turns a language description into a parser that can build ASTs from source code.
    Language,  # This class wraps a compiled language grammar (like C++) that the parser will use.
)  # This closing parenthesis ends the grouped import from tree_sitter.


"""
WHAT IS AN AST (ABSTRACT SYNTAX TREE)?  # This heading introduces the concept in simple terms.

When a computer (or an AI helper) reads source code, it does not think in "lines of text" like humans do.  # This line explains that machines see structure, not just lines.
Instead, it turns the code into a TREE structure called an AST.  # This line introduces the abstract syntax tree image.

You can imagine the AST like an outline:  # This line compares the AST to a document outline.
  - At the top, there is one root node representing the entire file.  # This bullet explains the root node.
  - Under that, there are child nodes for things like classes, functions, and global variables.  # This bullet describes high-level nodes.
  - Under each function, there are nodes for the name, parameters, and body.  # This bullet shows how function details are represented.
  - Inside the body node, there are nodes for if-statements, loops, and individual expressions.  # This bullet shows deeper nesting.

Tree-sitter builds this AST for us from raw source code bytes.  # This line connects the AST concept to the Tree-sitter library.
Each node in the AST:  # This line introduces key properties.
  - has a TYPE name (for example "function_definition" for a C++ function),  # This bullet explains node types.
  - has CHILDREN (for example, a function has a child for its name and a child for its body),  # This bullet explains tree shape.
  - remembers the exact BYTE RANGE in the original source text that it covers.  # This bullet explains how we can map back to text.

Because of this, the AI can:  # This line explains why ASTs help.
  - reliably find all function definitions,  # This bullet says we can enumerate functions.
  - know exactly which characters belong to each function body,  # This bullet says we can slice bodies safely.
  - and work with structure instead of guessing by string search.  # This bullet highlights robustness.

In this file, we:  # This line summarizes the purpose of the helpers below.
  1. Create a C++ parser using Tree-sitter and the tree-sitter-cpp grammar.  # This item describes parser construction.
  2. Walk the AST to find all function_definition nodes.  # This item describes how we discover functions.
  3. Extract each function's name and body source code as plain text.  # This item describes what we return to the rest of the system.
"""  # This closing triple quote ends the long explanatory comment block.


def _create_cpp_parser() -> Parser:  # This helper function constructs and returns a Tree-sitter parser specialized for C++.
    """Create and return a Tree-sitter parser configured for C++ code."""  # This docstring summarizes the purpose of the helper.

    try:  # This try block tries to import the compiled C++ grammar package.
        import tree_sitter_cpp  # type: ignore[import]  # This import brings in the tree-sitter-cpp Python wheel that contains the C++ grammar.
    except ImportError as exc:  # This except block runs if the grammar package is not installed.
        raise RuntimeError(  # This line raises a clearer error so the user knows how to fix the problem.
            "C++ grammar package 'tree-sitter-cpp' is not installed. Please install it with 'pip install tree-sitter-cpp'."  # This message tells the user exactly what to install.
        ) from exc  # This part keeps the original ImportError attached for debugging.

    cpp_language_handle = tree_sitter_cpp.language()  # This line asks the grammar package for its internal Language handle.
    cpp_language = Language(cpp_language_handle)  # This line wraps that handle in a Language object that the Parser can understand.

    parser = Parser(cpp_language)  # This line creates a Parser configured to understand C++ code using the C++ Language object.
    return parser  # This line returns the ready-to-use parser to the caller.


def _read_file_text(file_path: str) -> str:  # This helper function reads an entire text file and returns its contents as a string.
    """Read the entire contents of a text file as UTF-8 text."""  # This docstring explains that we treat the file as UTF-8 encoded.

    with open(file_path, "r", encoding="utf-8") as file_handle:  # This line opens the file for reading with UTF-8 encoding.
        file_text = file_handle.read()  # This line reads the whole file into a single string named file_text.
    return file_text  # This line returns the text back to the caller.


def _collect_function_definitions(root_node: Any) -> List[Any]:  # This helper walks the AST and collects all nodes that represent full function definitions.
    """
    Traverse the AST starting at the root and collect all C++ function definition nodes.  # This docstring explains that we will walk the entire tree.

    In the Tree-sitter C++ grammar, whole function definitions usually have the type
    name "function_definition". We do a full depth-first traversal of the entire tree,
    so this will find free functions and member functions nested inside classes,
    structs, and namespaces as well.  # This line explains what node type we look for and that nesting is handled.
    """  # This closing triple quote ends the docstring.

    function_nodes: List[Any] = []  # This list will hold every function_definition node we discover.
    stack: List[Any] = [root_node]  # This stack is initialized with the root node so we can perform depth-first traversal.

    while stack:  # This loop runs until there are no more nodes left to inspect.
        node = stack.pop()  # This line removes the last node from the stack so we can examine it.

        if node.type == "function_definition":  # This condition checks whether the current node is a full function definition.
            function_nodes.append(node)  # This line records the function_definition node in our results list.

        for child in node.children:  # This loop goes through each direct child of the current node.
            stack.append(child)  # This line pushes each child onto the stack so we will visit it later.

    return function_nodes  # This line returns the list of all discovered function_definition nodes.


def _extract_function_signature(node: Any, source_text: str) -> str:  # This helper extracts an approximate full function signature line.
    """
    Extract a best-effort textual function signature, including return type and parameters.  # This docstring explains that we want the full logical "header" of the function.

    We take the bytes from the start of the function_definition node up to (but not
    including) the body block, and then trim leading/trailing whitespace. This usually
    yields code like: "int Foo::bar(int x, int y) const noexcept".  # This line explains the heuristic used.
    """  # This closing triple quote ends the docstring.

    body_node = node.child_by_field_name("body")  # This line asks Tree-sitter for the specific child that represents the function body block.

    start_byte = node.start_byte  # This line records the starting byte offset of the whole function definition.
    if body_node is not None:  # This condition handles the common case where the function has a body.
        end_byte = body_node.start_byte  # This line sets the end just before the body block, to avoid including the implementation.
    else:
        end_byte = node.end_byte  # This line falls back to the full node span for declarations without a body.

    source_bytes = source_text.encode("utf-8")  # This line encodes the entire source text as UTF-8 bytes so we can slice by byte positions.
    header_bytes = source_bytes[start_byte:end_byte]  # This line slices out the candidate signature region.
    signature_text = header_bytes.decode("utf-8", errors="replace").strip()  # This line decodes and trims the text region.
    return signature_text  # This line returns the extracted signature text.


def _extract_leading_comments(node: Any, source_text: str) -> str:  # This helper extracts any contiguous comments immediately preceding a function.
    """
    Extract any docstrings or comments that appear immediately before a function.  # This docstring explains that we want leading documentation comments.

    We:
      1. Look at the line where the function starts.
      2. Walk upward line-by-line, collecting lines that look like comments (//, /*, *,
         */) or are blank.
      3. Stop when we hit a non-comment, non-blank line.  # This bullet describes the stopping condition.

    The collected comment lines are returned in top-down order as a single string,
    separated by newlines.  # This line explains the return format.
    """  # This closing triple quote ends the docstring.

    lines = source_text.splitlines()  # This line splits the entire source into a list of lines without keeping newline characters.
    start_row, _ = node.start_point  # This line reads the zero-based row index where the function_definition starts.

    comment_lines: List[str] = []  # This list will hold the contiguous block of comment lines above the function.
    current_row = start_row - 1  # This integer points to the line immediately above the function header.

    def _looks_like_comment_or_blank(text: str) -> bool:  # This inner helper decides whether a line is part of the leading comment block.
        stripped = text.strip()
        if not stripped:
            return True  # Blank lines are allowed inside the doc block.
        return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*") or stripped.endswith("*/")

    while current_row >= 0:  # This loop walks upward from the function start to the top of the file.
        candidate = lines[current_row]
        if not _looks_like_comment_or_blank(candidate):  # This condition stops when we hit real code.
            break
        comment_lines.append(candidate)  # This line records the comment or blank line.
        current_row -= 1  # This line moves to the previous line.

    # The walk collected lines from bottom to top; reverse so they are in natural order.
    comment_lines.reverse()  # This line restores the original top-down order.
    return "\n".join(comment_lines).strip()  # This line joins and trims the block before returning it.


def _extract_identifier_from_function(node: Any, source_bytes: bytes) -> str:  # This helper tries to recover the function name (identifier) from a function_definition node.
    """
    Try to find the function's name (identifier) inside a "function_definition" node.  # This docstring explains our goal in plain English.

    We do this by:  # This line introduces the steps we follow.
      1. Looking for a child with field name "declarator" (this usually holds the function signature).  # This step explains where function names often live.
      2. Searching inside that subtree for a node of type "identifier".  # This step explains we expect the name to be an identifier node.
      3. Falling back to a generic search for "identifier" anywhere under the function node if needed.  # This step explains our fallback strategy.
    """  # This closing triple quote ends the docstring.

    declarator = node.child_by_field_name("declarator")  # This line asks Tree-sitter for the declarator child, which usually includes the function name.

    search_root = declarator if declarator is not None else node  # This line picks where to start searching: inside the declarator, or the whole function node.

    stack: List[Any] = [search_root]  # This stack starts with the chosen root so we can walk every descendant.

    while stack:  # This loop runs until we have checked all potential child nodes.
        current = stack.pop()  # This line takes the next node from the stack to inspect.

        if current.type == "identifier":  # This condition checks whether the current node is an identifier, which is often a function or variable name.
            start_byte = current.start_byte  # This line reads the start byte offset of the identifier text in the original source bytes.
            end_byte = current.end_byte  # This line reads the end byte offset of the identifier text in the original source bytes.
            identifier_text = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")  # This line slices and decodes that region to recover the name as text.
            return identifier_text  # This line returns the first identifier we find as the function name.

        for child in current.children:  # This loop walks through all children of the current node.
            stack.append(child)  # This line pushes each child onto the stack so we will inspect it later.

    return "<anonymous_function>"  # This fallback string is used if we fail to find any identifier at all, to avoid crashing.


def _extract_function_body_source(node: Any, source_text: str) -> str:  # This helper extracts the exact source code for the body of a function.
    """
    Extract the source code that corresponds to the body of a function.  # This docstring explains what text we will return.

    We:  # This line introduces the steps.
      1. Ask Tree-sitter for the child node with field name "body".  # This step gets the subtree representing the block of code in the function.
      2. Read its byte range and slice that section from the original text.  # This step turns AST positions back into text positions.
      3. Return that slice (which normally includes the braces and all internal statements).  # This step describes the content that comes back.
    """  # This closing triple quote ends the docstring.

    body_node = node.child_by_field_name("body")  # This line asks Tree-sitter for the specific child that represents the function body block.

    if body_node is None:  # This condition handles the case where a function has no body (for example, a pure declaration).
        return ""  # This line returns an empty string instead of raising an error, signalling that no body is present.

    start_byte = body_node.start_byte  # This line records where in the source bytes the body starts.
    end_byte = body_node.end_byte  # This line records where in the source bytes the body ends.

    source_bytes = source_text.encode("utf-8")  # This line encodes the entire source text as UTF-8 bytes so we can slice by byte positions.
    body_bytes = source_bytes[start_byte:end_byte]  # This line slices out just the bytes that belong to the function body.
    body_text = body_bytes.decode("utf-8", errors="replace")  # This line decodes those bytes back into a string representing the function body code.
    return body_text  # This line returns the extracted body code to the caller.


def _extract_call_target_name(call_node: Any, source_bytes: bytes) -> str:  # This helper tries to recover the callee name from a call_expression node.
    """
    Try to find the name of the function being called inside a "call_expression" node.  # This docstring explains our goal in plain English.

    In the Tree-sitter C++ grammar, a call_expression usually has a child with field
    name "function" that represents what is being invoked (an identifier, a scoped
    identifier, or a member access). We:
      1. Prefer to search under the "function" field if it exists.
      2. Fall back to searching the entire call_expression subtree.
    We then look for the first "identifier" node in that region and treat it as the
    callee name.
    """  # This closing triple quote ends the docstring.

    function_target = call_node.child_by_field_name("function")  # This line asks Tree-sitter for the function part of the call (what is being invoked).
    search_root = function_target if function_target is not None else call_node  # This line chooses where to start searching.

    stack: List[Any] = [search_root]  # This stack will walk the candidate subtree looking for an identifier.

    while stack:  # This loop runs until we have checked all nodes under the chosen root.
        current = stack.pop()  # This line retrieves the next node to inspect.

        if current.type == "identifier":  # This condition checks whether the current node is a bare identifier (the simplest form of a callee).
            start_byte = current.start_byte  # This line reads the start byte offset of the identifier text in the original source bytes.
            end_byte = current.end_byte  # This line reads the end byte offset of the identifier text in the original source bytes.
            identifier_text = source_bytes[start_byte:end_byte].decode("utf-8", errors="replace")  # This line slices and decodes that region to recover the text.
            return identifier_text  # This line returns the identifier we found as the callee name.

        for child in current.children:  # This loop walks through all children of the current node.
            stack.append(child)  # This line pushes each child so we eventually inspect the entire subtree.

    return "<anonymous_call>"  # This fallback string is used if we fail to find any identifier at all, to avoid crashing.


def _collect_function_calls(func_node: Any, source_bytes: bytes) -> List[str]:  # This helper walks a function body and collects all function call names.
    """
    Traverse the body of a function_definition node and collect the names of all
    functions it calls, based on "call_expression" nodes in the AST.  # This docstring explains that we are building a simple call graph.
    """  # This closing triple quote ends the docstring.

    body_node = func_node.child_by_field_name("body")  # This line asks Tree-sitter for the specific child that represents the function body block.
    if body_node is None:  # This condition handles the case where a function has no body (for example, a pure declaration).
        return []  # This line returns an empty list, signalling that there are no calls to record.

    calls: List[str] = []  # This list will hold the distinct names of functions that are called from within this function.
    stack: List[Any] = [body_node]  # This stack is initialized with the body node so we can perform depth-first traversal within the function body.

    while stack:  # This loop runs until we have visited every node inside the function body.
        node = stack.pop()  # This line removes the next node from the stack for inspection.

        if node.type == "call_expression":  # This condition checks whether the current node represents a function call.
            callee_name = _extract_call_target_name(node, source_bytes)  # This line extracts the name of the called function, if possible.
            if callee_name and callee_name != "<anonymous_call>" and callee_name not in calls:  # This condition filters out placeholders and duplicates.
                calls.append(callee_name)  # This line records the distinct callee name in our list.

        for child in node.children:  # This loop pushes all children so we eventually visit the entire body subtree.
            stack.append(child)  # This line adds the child node to the stack for later inspection.

    return calls  # This line returns the list of distinct callee names for this function.


def extract_functions_from_cpp_file(file_path: str) -> List[Dict[str, Any]]:  # This is the main function other modules call to get function names, bodies, and dependencies from a C++ file.
    """
    Parse a C++ file and extract all function names and their body code.  # This docstring explains in plain English what this function returns.

    The return value is a list of dictionaries, each with:  # This line introduces the shape of each result item.
      - "name": the function name as a string.  # This bullet explains the "name" key.
      - "body": the exact source code inside the function body.  # This bullet explains the "body" key.
      - "calls": a list of function names that this function calls (simple call graph).  # This bullet explains the "calls" key.
      - "start_byte": the starting byte offset of the whole function definition in the source file.  # This bullet explains the "start_byte" key.
      - "end_byte": the ending byte offset of the whole function definition in the source file.  # This bullet explains the "end_byte" key.
      - "signature": a best-effort textual function signature including return type and parameters.  # This bullet explains the "signature" key.
      - "comments": any contiguous comment lines immediately preceding the function (docstring-style).  # This bullet explains the "comments" key.
      - "line_numbers": a dictionary with "start" and "end" line numbers (1-based) for debugging.  # This bullet explains the "line_numbers" key.
    """  # This closing triple quote ends the docstring.

    if not os.path.isfile(file_path):  # This condition checks that the given path actually points to an existing regular file.
        raise FileNotFoundError(f"C++ file not found: {file_path}")  # This line raises a clear error so callers know the path is invalid.

    parser = _create_cpp_parser()  # This line obtains a ready-to-use C++ parser from our helper function.

    source_text = _read_file_text(file_path)  # This line reads the entire C++ source file into a single string.
    source_bytes = source_text.encode("utf-8")  # This line encodes that string into UTF-8 bytes for Tree-sitter to parse.

    tree = parser.parse(source_bytes)  # This line asks Tree-sitter to build an AST for the given source bytes.
    root_node = tree.root_node  # This line retrieves the root node of the AST, which represents the whole file.

    function_nodes = _collect_function_definitions(root_node)  # This line walks the AST to find all function_definition nodes in the file.

    results: List[Dict[str, Any]] = []  # This list will hold one dictionary per discovered function, with its name, body text, call dependencies, and metadata.

    for func_node in function_nodes:  # This loop processes each function_definition node we found.
        name = _extract_identifier_from_function(func_node, source_bytes)  # This line extracts the function name for the current node.
        body = _extract_function_body_source(func_node, source_text)  # This line extracts the function body text for the current node.
        calls = _collect_function_calls(func_node, source_bytes)  # This line collects the names of all functions called from inside this function.
        start_byte = func_node.start_byte  # This line records the starting byte offset of the whole function definition.
        end_byte = func_node.end_byte  # This line records the ending byte offset of the whole function definition.
        signature = _extract_function_signature(func_node, source_text)  # This line extracts a best-effort signature for the function.
        comments = _extract_leading_comments(func_node, source_text)  # This line extracts any immediately preceding comment block.
        start_row, _ = func_node.start_point  # This line reads the zero-based row index where the function starts.
        end_row, _ = func_node.end_point  # This line reads the zero-based row index where the function ends.
        line_numbers = {"start": start_row + 1, "end": end_row + 1}  # This dictionary stores 1-based start/end line numbers for debugging.
        results.append(  # This line appends a dictionary containing name, body, calls, and metadata to the results list.
            {
                "name": name,
                "body": body,
                "calls": calls,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "signature": signature,
                "comments": comments,
                "line_numbers": line_numbers,
            }
        )

    return results  # This line returns the full list of function information to the caller.


class CppParser:  # This lightweight wrapper class provides an explicit CppParser interface for the rest of the system.
    """
    Thin convenience wrapper around the Tree-sitter based helpers above.  # This docstring explains that this class uses the existing extract_functions_from_cpp_file logic.

    The intent is to make the call-site in the LangGraph nodes read cleanly as "CppParser"
    without forcing those nodes to know about the lower-level helper functions.
    """  # This closing triple quote ends the docstring.

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:  # This method parses a C++ source file and returns the rich function metadata list.
        """
        Parse a C++ file and return the same structure as extract_functions_from_cpp_file.  # This docstring explains that this method is a thin wrapper.
        """

        return extract_functions_from_cpp_file(file_path)  # This line delegates to the existing helper so we keep one source of truth.


if __name__ == "__main__":  # This condition checks whether this file is being run directly, not imported as a module.
    import sys  # This import lets us read command-line arguments when running the script directly.
    from pprint import pprint  # This import gives us a nicer way to print complex Python objects in a readable form.

    if len(sys.argv) != 2:  # This condition ensures the user passed exactly one argument, the path to a C++ file.
        print("Usage: python -m core.parser <path-to-file.cpp>")  # This line prints usage instructions for the user.
        sys.exit(1)  # This line exits with a non-zero status to indicate incorrect usage.

    cpp_path = sys.argv[1]  # This line reads the C++ file path from the command-line argument.
    functions_info = extract_functions_from_cpp_file(cpp_path)  # This line calls our main helper to extract functions from that file.
    pprint(functions_info)  # This line prints the list of functions and bodies so you can see what the parser discovered.