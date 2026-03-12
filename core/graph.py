from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Polymorphism helpers
# ---------------------------------------------------------------------------

def _compute_signature_hash(parameters: List[Dict[str, Any]]) -> str:
    """Return a short deterministic hash of a function's parameter types.

    Used to distinguish overloaded functions that share the same simple name.
    """
    type_str = ",".join(str(p.get("type") or "") for p in parameters)
    return hashlib.md5(type_str.encode("utf-8")).hexdigest()[:8]


def _build_class_hierarchy(
    types_info: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Return a mapping of class/struct name → list of direct base class names."""
    hierarchy: Dict[str, List[str]] = {}
    for t in types_info:
        if t.get("type") in {"class", "struct"}:
            name = str(t.get("name") or "")
            if name:
                hierarchy[name] = [str(b) for b in (t.get("bases") or [])]
    return hierarchy


def _get_ancestors(
    class_name: str,
    hierarchy: Dict[str, List[str]],
) -> List[str]:
    """BFS walk up the inheritance graph and return all ancestor class names."""
    result: List[str] = []
    visited: set[str] = set()
    queue: List[str] = list(hierarchy.get(class_name, []))
    while queue:
        base = queue.pop(0)
        if base in visited:
            continue
        visited.add(base)
        result.append(base)
        queue.extend(hierarchy.get(base, []))
    return result


def _resolve_virtual_call_targets(
    method_name: str,
    caller_class: str,
    class_hierarchy: Dict[str, List[str]],
    method_to_fqns: Dict[str, List[str]],
) -> List[str]:
    """Resolve a virtual/polymorphic method call to a list of candidate FQNs.

    Resolution order:
    1. Exact match in the caller's own class (``caller_class::method_name``).
    2. Walk up the inheritance hierarchy to find the first ancestor that
       implements the method.
    3. If the caller class is unknown or not in the hierarchy, return all FQNs
       that define a method with that simple name (conservative over-approximation).

    Returns an empty list when *method_name* is not defined anywhere in the file
    (i.e., it is an external call).
    """
    candidates = method_to_fqns.get(method_name, [])
    if not candidates:
        return []  # External — let the caller handle it.

    if caller_class:
        exact = f"{caller_class}::{method_name}"
        if exact in candidates:
            return [exact]
        # Walk up the hierarchy to find an inherited implementation.
        for ancestor in _get_ancestors(caller_class, class_hierarchy):
            inherited = f"{ancestor}::{method_name}"
            if inherited in candidates:
                return [inherited]

    # Free-function context or no match found in hierarchy: be conservative.
    return candidates


def build_dependency_graph(
    functions_info: List[Dict[str, Any]],
    types_info: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[nx.DiGraph, Dict[str, List[str]], Dict[str, bool], List[str]]:
    graph: nx.DiGraph = nx.DiGraph()
    is_defined_in_file: Dict[str, bool] = {}
    defined_function_names: List[str] = []

    class_hierarchy = _build_class_hierarchy(types_info or [])

    # Build an index: simple method name → all FQNs in this file that define it.
    # Used for virtual/inherited call resolution.
    method_to_fqns: Dict[str, List[str]] = {}
    for fn in functions_info:
        fqn = str(fn.get("fqn") or fn.get("name") or "")
        simple = str(fn.get("name") or "")
        if simple and fqn:
            method_to_fqns.setdefault(simple, []).append(fqn)

    # Pass 1: add every defined function as a graph node with attributes.
    for fn in functions_info:
        name = str(fn.get("name") or "")
        if not name:
            continue
        parameters = fn.get("parameters") or []
        sig_hash = _compute_signature_hash(parameters)
        modifiers = fn.get("modifiers") or []
        is_virtual = "virtual" in modifiers
        if name not in graph:
            graph.add_node(name, signature_hash=sig_hash, is_virtual=is_virtual)
        else:
            graph.nodes[name]["signature_hash"] = sig_hash
            graph.nodes[name]["is_virtual"] = is_virtual
        is_defined_in_file[name] = True
        defined_function_names.append(name)

    # Pass 2: add call edges, using call_details for polymorphic resolution.
    for fn in functions_info:
        caller = str(fn.get("name") or "")
        if not caller:
            continue

        # Derive the class scope of this caller from its FQN so we can walk the
        # inheritance graph when resolving virtual method calls.
        caller_fqn = str(fn.get("fqn") or caller)
        fqn_parts = caller_fqn.split("::")
        caller_class = fqn_parts[-2] if len(fqn_parts) >= 2 else ""

        call_details = fn.get("call_details") or []
        if call_details:
            for detail in call_details:
                if not isinstance(detail, dict):
                    continue
                kind = str(detail.get("kind") or "")
                call_name = str(detail.get("name") or "")
                call_display = str(detail.get("display") or call_name)
                if not call_name:
                    continue

                if kind == "method":
                    # Use inheritance-aware resolution for virtual/polymorphic calls.
                    resolved_fqns = _resolve_virtual_call_targets(
                        call_name, caller_class, class_hierarchy, method_to_fqns
                    )
                    if resolved_fqns:
                        for target_fqn in resolved_fqns:
                            target_node = target_fqn.split("::")[-1]
                            if target_node not in graph:
                                graph.add_node(target_node)
                            if target_node not in is_defined_in_file:
                                is_defined_in_file[target_node] = False
                            graph.add_edge(caller, target_node)
                    else:
                        # External method call — record with display name.
                        if call_display not in graph:
                            graph.add_node(call_display)
                        if call_display not in is_defined_in_file:
                            is_defined_in_file[call_display] = False
                        graph.add_edge(caller, call_display)
                else:
                    # Local or scoped call.
                    callee_name = call_display if "::" in call_display else call_name
                    if not callee_name:
                        continue
                    if callee_name not in graph:
                        graph.add_node(callee_name)
                    if callee_name not in is_defined_in_file:
                        is_defined_in_file[callee_name] = False
                    graph.add_edge(caller, callee_name)
        else:
            # Legacy path: functions_info without call_details (simple calls list).
            raw_calls = fn.get("calls") or []
            for callee in raw_calls:
                callee_name = str(callee or "")
                if not callee_name:
                    continue
                if callee_name not in graph:
                    graph.add_node(callee_name)
                if callee_name not in is_defined_in_file:
                    is_defined_in_file[callee_name] = False
                graph.add_edge(caller, callee_name)

    dependency_map: Dict[str, List[str]] = {}
    for fn_name in defined_function_names:
        neighbors = sorted({str(n) for n in graph.successors(fn_name)})
        dependency_map[fn_name] = neighbors

    return graph, dependency_map, is_defined_in_file, defined_function_names


def analyze_dependency_graph(
    graph: nx.DiGraph,
    defined_function_names: List[str],
    is_defined_in_file: Dict[str, bool],
) -> Dict[str, Any]:
    orphans: List[str] = []
    entry_points: List[str] = []

    for fn_name in defined_function_names:
        if graph.in_degree(fn_name) == 0:
            if fn_name == "main":
                entry_points.append(fn_name)
            else:
                orphans.append(fn_name)

    cycles: List[List[str]] = []
    for cycle in nx.simple_cycles(graph):
        as_strings = [str(name) for name in cycle]
        if all(name in defined_function_names for name in as_strings):
            cycles.append(as_strings)

    return {
        "orphans": sorted(orphans),
        "cycles": cycles,
        "entry_points": sorted(entry_points),
    }


def build_analysis_report(
    functions_info: List[Dict[str, Any]],
    dependency_map: Dict[str, List[str]],
    orphans: List[str],
    cycles: List[List[str]],
) -> str:
    total_functions = len(functions_info)
    lines: List[str] = []

    lines.append(f"Total functions: {total_functions}")
    lines.append(
        f"Orphan functions (no callers): {', '.join(sorted(orphans)) or 'none'}"
    )

    if cycles:
        formatted_cycles = [" -> ".join(cycle) for cycle in cycles]
        lines.append("Circular recursion / cycles:")
        for cycle_str in formatted_cycles:
            lines.append(f"  - {cycle_str}")
    else:
        lines.append("Circular recursion / cycles: none")

    lines.append("Per-function call summary:")
    for fn_name in sorted(dependency_map.keys()):
        callees = dependency_map.get(fn_name, [])
        lines.append(f"  - {fn_name} calls [{', '.join(callees) or 'none'}]")

    return "\n".join(lines)


class DependencyGraph:
    def __init__(
        self,
        functions_info: List[Dict[str, Any]],
        types_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        (
            graph,
            dependency_map,
            is_defined_in_file,
            defined_function_names,
        ) = build_dependency_graph(functions_info, types_info=types_info)
        self.graph: nx.DiGraph = graph
        self._dependency_map: Dict[str, List[str]] = dependency_map
        self.is_defined_in_file: Dict[str, bool] = is_defined_in_file
        self.defined_function_names: List[str] = defined_function_names
        self._criticality_scores: Dict[str, float] = self._compute_criticality_scores()

    @property
    def dependency_map(self) -> Dict[str, List[str]]:
        return self._dependency_map

    def _compute_criticality_scores(self) -> Dict[str, float]:
        internal = {
            name for name, internal in self.is_defined_in_file.items() if internal
        }
        total_internal = len(internal)
        scores: Dict[str, float] = {
            str(node): 0.0 for node in self.graph.nodes
        }
        if total_internal == 0:
            return scores

        for fn_name in internal:
            downstream = nx.descendants(self.graph, fn_name)
            downstream_internal = len(downstream & internal)
            scores[fn_name] = downstream_internal / float(total_internal)

        return scores

    def analyze(self) -> Dict[str, Any]:
        return analyze_dependency_graph(
            self.graph, self.defined_function_names, self.is_defined_in_file
        )

    def get_impact_radius(self, function_name: str) -> float:
        return float(self._criticality_scores.get(function_name, 0.0))

    def get_bottlenecks(self) -> List[str]:
        internal_nodes = [
            name
            for name, internal in self.is_defined_in_file.items()
            if internal
        ]
        if not internal_nodes:
            return []

        in_degrees = {name: self.graph.in_degree(name) for name in internal_nodes}
        max_degree = max(in_degrees.values(), default=0)
        if max_degree == 0:
            return []

        return sorted(
            [name for name, degree in in_degrees.items() if degree == max_degree]
        )

    def get_modernization_order(self) -> List[str]:
        internal_nodes = [
            name
            for name, internal in self.is_defined_in_file.items()
            if internal
        ]
        if not internal_nodes:
            return []
        try:
            topo = list(nx.topological_sort(self.graph.reverse()))
            ordered = [name for name in topo if name in internal_nodes]
            remaining = [name for name in internal_nodes if name not in ordered]
            ordered.extend(sorted(remaining))
            return ordered
        except Exception:
            return sorted(internal_nodes)

    def to_dict(self) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        for node in self.graph.nodes:
            name = str(node)
            attrs = self.graph.nodes[name]
            nodes.append(
                {
                    "name": name,
                    "is_defined_in_file": bool(
                        self.is_defined_in_file.get(name, False)
                    ),
                    "criticality_score": float(
                        self._criticality_scores.get(name, 0.0)
                    ),
                    "signature_hash": str(attrs.get("signature_hash") or ""),
                    "is_virtual": bool(attrs.get("is_virtual", False)),
                }
            )

        edges = [
            {"from": str(u), "to": str(v)}
            for u, v in self.graph.edges
        ]
        return {"nodes": nodes, "edges": edges}


def get_modernization_order(dependency_map: Dict[str, List[str]]) -> List[str]:
    graph: nx.DiGraph = nx.DiGraph()

    for caller, callees in dependency_map.items():
        graph.add_node(caller)
        for callee in callees:
            callee_name = str(callee or "")
            if not callee_name:
                continue
            graph.add_node(callee_name)
            graph.add_edge(caller, callee_name)

    internal_nodes = set(dependency_map.keys())
    if not internal_nodes:
        return []

    try:
        topo = list(nx.topological_sort(graph.reverse()))
        ordered = [name for name in topo if name in internal_nodes]
        remaining = [name for name in sorted(internal_nodes) if name not in ordered]
        ordered.extend(remaining)
        return ordered
    except Exception:
        return sorted(internal_nodes)