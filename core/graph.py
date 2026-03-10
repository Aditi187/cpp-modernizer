
"""
Graph utilities for the Code Modernization Engine.

This module provides tools for building and analyzing function call graphs using networkx.
"""


from __future__ import annotations
from typing import Any, Dict, List, Tuple
import networkx as nx
import logging



def build_dependency_graph(
    functions_info: List[Dict[str, Any]]
) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
    """
    Builds a directed call graph from a list of function metadata.

    The implementation takes care to avoid adding duplicate edges and to
    ignore self-referential calls, which can arise when the parser includes a
    call expression inside the same function definition.

    Args:
        functions_info: List of dictionaries containing function metadata.

    Returns:
        Tuple containing:
            - graph: networkx.DiGraph instance.
            - dependency_map: Dictionary mapping function name to sorted unique list of functions it calls.
    """
    graph: nx.DiGraph = nx.DiGraph()
    defined_function_names: List[str] = []
    for function_metadata in functions_info:
        function_name = str(function_metadata.get("name") or "")
        if not function_name:
            continue
        if function_name not in graph:
            graph.add_node(function_name)
        defined_function_names.append(function_name)

    for function_metadata in functions_info:
        caller_name = str(function_metadata.get("name") or "")
        if not caller_name:
            continue
        called_functions = function_metadata.get("calls") or []
        # use a set to dedupe and avoid self edges
        seen: set[str] = set()
        for callee in called_functions:
            callee_name = str(callee or "")
            if not callee_name or callee_name == caller_name:
                # ignore empty names and self-recursion
                continue
            if callee_name in seen:
                continue
            seen.add(callee_name)
            if callee_name not in graph:
                graph.add_node(callee_name)
            graph.add_edge(caller_name, callee_name)

    dependency_map: Dict[str, List[str]] = {}
    for function_name in defined_function_names:
        neighbors = sorted({str(neighbor) for neighbor in graph.successors(function_name)})
        dependency_map[function_name] = neighbors
    return graph, dependency_map



def analyze_dependency_graph(
    graph: nx.DiGraph,
    defined_function_names: List[str]
) -> Dict[str, Any]:
    """
    Analyzes a directed call graph to identify orphan functions and cycles.

    Args:
        graph: networkx.DiGraph instance representing the call graph.
        defined_function_names: List of function names defined in the translation unit.

    Returns:
        Dictionary with keys:
            - "orphans": Sorted list of orphan function names.
            - "cycles": List of cycles (each cycle is a list of function names).
    """
    orphans: List[str] = []
    for function_name in defined_function_names:
        if graph.in_degree(function_name) == 0:
            orphans.append(function_name)
    cycles: List[List[str]] = []
    for cycle in nx.simple_cycles(graph):
        cycle_names = [str(name) for name in cycle]
        cycles.append(cycle_names)
    return {
        "orphans": sorted(orphans),
        "cycles": cycles,
    }



def build_analysis_report(
    functions_info: List[Dict[str, Any]],
    dependency_map: Dict[str, List[str]],
    orphans: List[str],
    cycles: List[List[str]]
) -> str:
    """
    Builds a concise, human-readable summary of the code structure in the translation unit.

    Args:
        functions_info: List of function metadata dictionaries.
        dependency_map: Dictionary mapping function name to list of callees.
        orphans: List of orphan function names.
        cycles: List of cycles (each cycle is a list of function names).

    Returns:
        Multi-line string summary.
    """
    total_functions = len(functions_info)
    lines: List[str] = []
    lines.append(f"Total functions: {total_functions}")
    lines.append(f"Orphan functions (no callers): {', '.join(sorted(orphans)) or 'none'}")
    if cycles:
        formatted_cycles = [" -> ".join(cycle) for cycle in cycles]
        lines.append("Circular recursion / cycles:")
        for cycle_str in formatted_cycles:
            lines.append(f"  - {cycle_str}")
    else:
        lines.append("Circular recursion / cycles: none")
    lines.append("Per-function call summary:")
    for function_name in sorted(dependency_map.keys()):
        callees = dependency_map.get(function_name, [])
        lines.append(f"  - {function_name} calls [{', '.join(callees) or 'none'}]")
    return "\n".join(lines)



class DependencyGraph:
    """
    Wrapper class for function call graph utilities.

    Builds a directed call graph from function metadata, exposes dependency map,
    analyzes for orphans and cycles, and exports nodes/edges for visualization.
    """
    def __init__(self, functions_info: List[Dict[str, Any]]) -> None:
        """
        Initializes DependencyGraph from a list of function metadata dictionaries.

        The constructor builds the internal directed graph and a lightweight
        dependency map.  It also records the set of names that were *defined*
        in this translation unit.  Note that "main" (or any other entry-point
        function) will appear just like an orphan because it has no callers, but
        the distinction between a genuine orphan and the entry point is left to
        callers of ``analyze`` (see below).

        Args:
            functions_info: List of function metadata dictionaries.
        """
        graph, dependency_map = build_dependency_graph(functions_info)
        self.graph: nx.DiGraph = graph
        self.dependency_map: Dict[str, List[str]] = dependency_map
        self.defined_function_names: List[str] = [
            str(function_metadata.get("name") or "")
            for function_metadata in functions_info
            if function_metadata.get("name")
        ]

    def analyze(self) -> Dict[str, Any]:
        """
        Analyzes the graph for orphans and cycles.

        Logical orphans are defined as functions that have no incoming edges in
        the call graph.  A program entry point such as ``main`` will also show up
        in this list, because by definition nobody calls it.  The calling code
        (e.g. the workflow or test harness) is responsible for treating ``main``
        specially if desired.  For example, the pruner node in the workflow
        explicitly filters out ``main`` when removing orphan functions.

        Returns:
            Dictionary with keys "orphans" and "cycles".
        """
        return analyze_dependency_graph(self.graph, self.defined_function_names)
    def to_dict(self) -> Dict[str, Any]:
        """
        Exports the call graph as a JSON-serializable dictionary.

        Returns:
            Dictionary with "nodes" (list of function names) and "edges" (list of caller/callee dicts).
        """
        nodes = [str(node) for node in self.graph.nodes]
        edges = [
            {"from": str(caller), "to": str(callee)}
            for caller, callee in self.graph.edges
        ]
        return {"nodes": nodes, "edges": edges}

    def get_impact_radius(self, function_name: str) -> int:
        """
        Calculate how many distinct functions are downstream of the given
        function, either directly or transitively.

        This provides a simple measure of the "impact radius" – how many other
        functions could be affected if `function_name` were changed.

        Args:
            function_name: Name of the function to inspect.

        Returns:
            int: Number of unique functions reachable from `function_name`.
        """
        if function_name not in self.graph:
            return 0
        visited: set[str] = set()
        stack = [function_name]
        while stack:
            current = stack.pop()
            for succ in self.graph.successors(current):
                succ_name = str(succ)
                if succ_name not in visited:
                    visited.add(succ_name)
                    stack.append(succ_name)
        return len(visited)
        