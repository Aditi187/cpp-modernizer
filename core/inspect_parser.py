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
	"""
	try:
		import tree_sitter_cpp  # type: ignore[import]
	except ImportError as exc:
		raise RuntimeError(
			"C++ grammar package 'tree-sitter-cpp' is not installed. Please install it with 'pip install tree-sitter-cpp'."
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
		File contents as a string.
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
		List of function definition nodes.
	"""
from tree_sitter import Parser
import inspect

print(Parser)
print(dir(Parser))
