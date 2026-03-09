"""
Graph utilities for the Code Modernization Engine.  # This line explains that this file holds helpers for building and analyzing call graphs.

We use networkx to represent the relationships between functions in a translation
unit (which functions call which). This lets higher-level nodes reason about:
  - orphan functions (no callers),
  - circular recursions / cycles,
  - and a compact "knowledge graph" view of the local code structure.
"""  # This closing triple quote ends the human-readable description of this module.

from __future__ import annotations  # This import allows postponed evaluation of type annotations.

from typing import Any, Dict, List, Tuple  # This import lets us describe dictionaries and lists in type hints.

import networkx as nx  # This import brings in networkx, which we use to build directed graphs.


def build_dependency_graph(  # This function constructs a directed graph from the parsed function metadata.
    functions_info: List[Dict[str, Any]],
) -> Tuple[nx.DiGraph, Dict[str, List[str]]]:
    """
    Build a directed call graph (DependencyGraph) from the list of per-function metadata.  # This docstring explains what this helper produces.

    Each function definition becomes a node. For every call "f -> g" discovered by the
    parser, we add a directed edge from f (caller) to g (callee).

    Returns a tuple of:
      - graph: a networkx.DiGraph instance.
      - dependency_map: a plain dictionary mapping function name -> sorted unique list
        of functions it calls (for easy JSON serialization and LLM prompts).
    """  # This closing triple quote ends the docstring.

    graph: nx.DiGraph = nx.DiGraph()  # This line creates an empty directed graph to hold the call relationships.

    defined_function_names = []  # This list will record the names of all functions with definitions in this translation unit.

    for fn in functions_info:  # This loop processes each function metadata entry.
        name = str(fn.get("name") or "")  # This line normalizes the function name as a string.
        if not name:
            continue  # This line skips anonymous or malformed entries.
        if name not in graph:  # This condition ensures each defined function appears as a node.
            graph.add_node(name)  # This line adds a node for the function in the graph.
        defined_function_names.append(name)  # This line records that this function is defined here.

    for fn in functions_info:  # This loop adds edges based on the "calls" lists.
        caller = str(fn.get("name") or "")  # This line reads the caller function name.
        if not caller:
            continue  # This line skips entries without a usable name.
        raw_calls = fn.get("calls") or []  # This line reads the raw list of callees reported by the parser.
        for callee in raw_calls:  # This loop processes each callee name.
            callee_name = str(callee or "")  # This line normalizes the callee name as a string.
            if not callee_name:
                continue  # This line skips empty names.
            if callee_name not in graph:  # This condition ensures that the callee also appears as a node in the graph.
                graph.add_node(callee_name)  # This line adds a node for the callee (even if it is external to this translation unit).
            graph.add_edge(caller, callee_name)  # This line adds a directed edge from caller to callee.

    dependency_map: Dict[str, List[str]] = {}  # This dictionary will hold a stable, sorted adjacency list per function.
    for fn_name in defined_function_names:  # This loop builds a clean dependency list only for functions defined in this translation unit.
        neighbors = sorted({str(n) for n in graph.successors(fn_name)})  # This line collects and sorts all direct callees of fn_name.
        dependency_map[fn_name] = neighbors  # This line records the adjacency list for fn_name.

    return graph, dependency_map  # This line returns both the graph and the dependency_map.


def analyze_dependency_graph(  # This function analyzes the call graph to find orphans and cycles.
    graph: nx.DiGraph,
    defined_function_names: List[str],
) -> Dict[str, Any]:
    """
    Analyze a directed call graph to identify:  # This docstring explains what structural properties we compute.
      - orphan functions: defined functions that have no callers.
      - cycles: lists of function names that participate in circular recursion.
    """  # This closing triple quote ends the docstring.

    # Orphans: functions that are defined in this translation unit and have no
    # incoming edges (no callers). We intentionally do not special-case "main"
    # here; higher-level nodes can decide whether to prune it.
    orphans: List[str] = []  # This list will hold the names of functions with no callers.
    for fn_name in defined_function_names:  # This loop inspects each defined function.
        if graph.in_degree(fn_name) == 0:  # This condition checks whether any other function calls fn_name.
            orphans.append(fn_name)  # This line records fn_name as an orphan.

    # Cycles: networkx.simple_cycles returns a list of cycles, where each cycle is
    # a list of node names forming a directed loop.
    cycles: List[List[str]] = []  # This list will hold each simple cycle found in the graph.
    for cycle in nx.simple_cycles(graph):  # This loop iterates over all simple cycles in the graph.
        as_strings = [str(name) for name in cycle]  # This line normalizes node names as strings.
        cycles.append(as_strings)  # This line records the cycle.

    return {  # This dictionary summarizes the structural properties we discovered.
        "orphans": sorted(orphans),
        "cycles": cycles,
    }


def build_analysis_report(  # This function builds a human-readable summary of the local code graph.
    functions_info: List[Dict[str, Any]],
    dependency_map: Dict[str, List[str]],
    orphans: List[str],
    cycles: List[List[str]],
) -> str:
    """
    Build a concise, human-readable summary of the code structure in this translation unit.  # This docstring explains that the result is intended for logging and LLM context.
    """  # This closing triple quote ends the docstring.

    total_functions = len(functions_info)  # This line counts how many function definitions we have.
    lines: List[str] = []  # This list will accumulate lines of the report.

    lines.append(f"Total functions: {total_functions}")  # This line records the function count.
    lines.append(f"Orphan functions (no callers): {', '.join(sorted(orphans)) or 'none'}")  # This line records the orphan set.

    if cycles:  # This condition checks whether any cycles were discovered.
        formatted_cycles = [" -> ".join(cycle) for cycle in cycles]  # This line formats each cycle as a simple arrow chain.
        lines.append("Circular recursion / cycles:")  # This line introduces the cycle section.
        for cycle_str in formatted_cycles:  # This loop adds one line per cycle.
            lines.append(f"  - {cycle_str}")
    else:
        lines.append("Circular recursion / cycles: none")  # This line records that no cycles were found.

    # Optionally add a short per-function summary so the model sees the local neighborhood for each function.
    lines.append("Per-function call summary:")  # This line introduces the per-function section.
    for fn_name in sorted(dependency_map.keys()):  # This loop summarizes each function's outgoing edges.
        callees = dependency_map.get(fn_name, [])
        lines.append(f"  - {fn_name} calls [{', '.join(callees) or 'none'}]")  # This line records the callees for fn_name.

    return "\n".join(lines)  # This line joins all lines into a single multi-line string report.


class DependencyGraph:  # This lightweight wrapper class provides an explicit DependencyGraph abstraction for callers.
    """
    Object-oriented wrapper around the call graph utilities above.  # This docstring explains that this class encapsulates a networkx.DiGraph.

    It:
      - builds a directed call graph from a list of per-function metadata,
      - exposes a dependency_map adjacency list (function -> callees),
      - can analyze the graph for orphans and cycles,
      - and can export a simple nodes/edges dict for visualization or prompts.
    """  # This closing triple quote ends the docstring.

    def __init__(self, functions_info: List[Dict[str, Any]]) -> None:  # This initializer builds the underlying graph from function metadata.
        """
        Construct a DependencyGraph from a list of function metadata dictionaries.  # This docstring explains the constructor input.
        """

        graph, dependency_map = build_dependency_graph(functions_info)  # This line delegates to the functional helper to build the graph and adjacency list.
        self.graph: nx.DiGraph = graph  # This attribute stores the underlying networkx directed graph.
        self.dependency_map: Dict[str, List[str]] = dependency_map  # This attribute stores the caller -> callees adjacency list.
        self.defined_function_names: List[str] = [  # This attribute records the names of all functions with definitions.
            str(fn.get("name") or "")
            for fn in functions_info
            if fn.get("name")
        ]

    def analyze(self) -> Dict[str, Any]:  # This method analyzes the graph to find orphans and cycles.
        """
        Analyze the graph and return the same structure as analyze_dependency_graph.  # This docstring explains that this method is a thin wrapper.
        """

        return analyze_dependency_graph(self.graph, self.defined_function_names)  # This line delegates to the existing helper.

    def to_dict(self) -> Dict[str, Any]:  # This method exports the graph as simple nodes/edges data.
        """
        Export the call graph as a JSON-serializable dictionary with:
          - "nodes": list of function names
          - "edges": list of {"from": caller, "to": callee} dictionaries.  # This docstring explains the output shape.
        """

        nodes = [str(node) for node in self.graph.nodes]  # This line collects all node names as strings.
        edges = [  # This list comprehension builds a list of edge dictionaries.
            {"from": str(u), "to": str(v)}
            for u, v in self.graph.edges
        ]
        return {"nodes": nodes, "edges": edges}  # This line returns the combined structure.
        