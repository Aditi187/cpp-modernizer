from __future__ import annotations

from collections import deque
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


logger = logging.getLogger(__name__)
_SIGNATURE_HASH_LENGTH = 16


# ---------------------------------------------------------------------------
# Polymorphism helpers
# ---------------------------------------------------------------------------

def _compute_signature_hash(parameters: List[Dict[str, Any]]) -> str:
    """Return a deterministic hash of a function's parameter type list.

    Uses SHA-256 and keeps 16 hex chars to reduce collision risk in large code
    graphs while keeping node IDs compact.
    """
    type_str = ",".join(str(p.get("type") or "") for p in parameters)
    return hashlib.sha256(type_str.encode("utf-8")).hexdigest()[:_SIGNATURE_HASH_LENGTH]


def _make_node_id(unique_fqn: str, function_name: str, signature_hash: str) -> str:
    if unique_fqn:
        return unique_fqn
    return f"{function_name}#{signature_hash}"


def _node_display_name(graph: nx.DiGraph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    simple = str(attrs.get("name") or node_id)
    sig_hash = str(attrs.get("signature_hash") or "")
    if sig_hash:
        return f"{simple}#{sig_hash}"
    return simple


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
    queue: deque[str] = deque(hierarchy.get(class_name, []))
    while queue:
        base = queue.popleft()
        if base in visited:
            continue
        visited.add(base)
        result.append(base)
        queue.extend(str(item) for item in hierarchy.get(base, []))
    return result


def _call_argument_count(call_detail: Dict[str, Any]) -> Optional[int]:
    """Return argument count from call metadata when available.

    Parser schemas can vary; this helper supports common keys:
    - ``arguments``: list of parsed argument items
    - ``argument_count`` / ``arg_count``: explicit integer count
    """
    raw_arguments = call_detail.get("arguments")
    if isinstance(raw_arguments, list):
        return len(raw_arguments)

    for key in ("argument_count", "arg_count"):
        raw_count = call_detail.get(key)
        if isinstance(raw_count, int) and raw_count >= 0:
            return raw_count
    return None


def _matches_argument_count(
    graph: nx.DiGraph,
    node_id: str,
    argument_count: Optional[int],
) -> bool:
    """Return True when target function's parameter count matches the call-site."""
    if argument_count is None:
        return True
    node_param_count = graph.nodes[node_id].get("parameter_count")
    return isinstance(node_param_count, int) and node_param_count == argument_count


def _filter_nodes_by_arity(
    graph: nx.DiGraph,
    node_ids: List[str],
    argument_count: Optional[int],
) -> List[str]:
    """Filter candidate targets using call-site arity when available."""
    if argument_count is None:
        return list(node_ids)
    return [
        node_id
        for node_id in node_ids
        if _matches_argument_count(graph, node_id, argument_count)
    ]


def _resolve_virtual_call_targets(
    method_name: str,
    caller_class: str,
    class_hierarchy: Dict[str, List[str]],
    graph: nx.DiGraph,
    method_to_node_ids: Dict[str, List[str]],
    free_function_node_ids: Dict[str, List[str]],
    argument_count: Optional[int],
) -> List[str]:
    """Resolve a virtual/polymorphic method call to candidate node IDs.

    Resolution order:
    1. Exact match in the caller's own class.
    2. Walk up ancestors and collect inherited implementations.
    3. If caller class is unknown, fall back to free functions.

    Limitation: for multiple inheritance with ambiguous methods, this remains a
    conservative over-approximation and may return multiple callable candidates.
    That is intentional to avoid under-linking dependencies.

    Returns an empty list when *method_name* is not defined anywhere in the file
    (i.e., it is an external call).
    """
    if caller_class:
        candidates = method_to_node_ids.get(method_name, [])
        if not candidates:
            return []

        exact_prefix = f"{caller_class}::{method_name}"
        exact_matches = [
            node_id
            for node_id in candidates
            if str(graph.nodes[node_id].get("fqn") or "").startswith(exact_prefix)
            and _matches_argument_count(graph, node_id, argument_count)
        ]
        if exact_matches:
            return _ordered_unique(sorted(exact_matches))

        # Walk up the hierarchy to find inherited implementations.
        inherited_matches: List[str] = []
        for ancestor in _get_ancestors(caller_class, class_hierarchy):
            inherited_prefix = f"{ancestor}::{method_name}"
            for node_id in candidates:
                fqn = str(graph.nodes[node_id].get("fqn") or "")
                if not fqn.startswith(inherited_prefix):
                    continue
                if not _matches_argument_count(graph, node_id, argument_count):
                    continue
                inherited_matches.append(node_id)
        return _ordered_unique(inherited_matches)

    free_candidates = free_function_node_ids.get(method_name, [])
    return _filter_nodes_by_arity(graph, free_candidates, argument_count)


def _ordered_unique(values: List[str]) -> List[str]:
    """Preserve input order while removing duplicates."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _scc_modernization_order(graph: nx.DiGraph, internal_nodes: Set[str]) -> List[str]:
    """Return SCC-aware modernization order over internal nodes.

    Nodes are grouped by strongly connected component and topologically ordered
    on the reversed condensation DAG so dependencies are modernized first.
    """
    if not internal_nodes:
        return []

    internal_subgraph = graph.subgraph(internal_nodes).copy()
    if internal_subgraph.number_of_nodes() == 0:
        return []

    condensed = nx.condensation(internal_subgraph)
    topo_components = list(nx.topological_sort(condensed.reverse()))

    ordered_node_ids: List[str] = []
    for component in topo_components:
        members = condensed.nodes[component].get("members") or set()
        member_ids = sorted(str(node_id) for node_id in members)
        ordered_node_ids.extend(member_ids)
    return ordered_node_ids


def _is_malformed_function_record(entry: Any) -> bool:
    """Validate a function record shape before graph construction."""
    if not isinstance(entry, dict):
        return True
    if not str(entry.get("name") or ""):
        return True
    return False


def build_dependency_graph(
    functions_info: List[Dict[str, Any]],
    types_info: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[nx.DiGraph, Dict[str, List[str]], Dict[str, bool], List[str]]:
    """Build an overload-aware call graph from function metadata.

    Uses ``unique_fqn`` / ``signature_hash`` when available to prevent overload
    collisions in lookup maps.
    """
    graph: nx.DiGraph = nx.DiGraph()
    is_defined_in_file: Dict[str, bool] = {}
    defined_function_names: List[str] = []

    class_hierarchy = _build_class_hierarchy(types_info or [])

    # Build indexes used for overload-aware and polymorphic resolution.
    method_to_node_ids: Dict[str, List[str]] = {}
    free_function_node_ids_by_name: Dict[str, List[str]] = {}
    name_to_defined_node_ids: Dict[str, List[str]] = {}
    fqn_to_defined_node_ids: Dict[str, List[str]] = {}

    valid_functions: List[Dict[str, Any]] = []
    for idx, fn in enumerate(functions_info):
        if _is_malformed_function_record(fn):
            logger.warning("Skipping malformed function record at index %d", idx)
            continue
        valid_functions.append(fn)

    for fn in valid_functions:
        name = str(fn.get("name") or "")

        raw_parameters = fn.get("parameters")
        parameters = raw_parameters if isinstance(raw_parameters, list) else []
        parameter_count = len(parameters)
        sig_hash = str(fn.get("signature_hash") or "")
        if not sig_hash:
            sig_hash = _compute_signature_hash(parameters)
        fqn = str(fn.get("fqn") or name)
        unique_fqn = str(fn.get("unique_fqn") or f"{fqn}#{sig_hash}")
        node_id = _make_node_id(unique_fqn, name, sig_hash)
        modifiers = fn.get("modifiers") or []
        is_virtual = "virtual" in modifiers

        graph.add_node(
            node_id,
            node_id=node_id,
            display_name=name,
            name=name,
            fqn=fqn,
            unique_fqn=unique_fqn,
            signature_hash=sig_hash,
            parameter_count=parameter_count,
            is_virtual=is_virtual,
            is_defined_in_file=True,
            is_function_like=True,
        )

        name_to_defined_node_ids.setdefault(name, []).append(node_id)
        fqn_to_defined_node_ids.setdefault(fqn, []).append(node_id)
        is_defined_in_file[name] = True
        defined_function_names.append(name)

        if "::" in fqn:
            method_to_node_ids.setdefault(name, []).append(node_id)
        else:
            free_function_node_ids_by_name.setdefault(name, []).append(node_id)

    defined_function_names = _ordered_unique(defined_function_names)

    def _ensure_external_node(raw_name: str, *, is_function_like: bool) -> str:
        external_name = str(raw_name or "")
        if not external_name:
            return ""
        if not is_function_like:
            return ""
        if not graph.has_node(external_name):
            graph.add_node(
                external_name,
                node_id=external_name,
                display_name=external_name,
                name=external_name,
                fqn=external_name,
                unique_fqn=external_name,
                signature_hash="",
                is_virtual=False,
                is_defined_in_file=False,
                is_function_like=True,
            )
        else:
            attrs = graph.nodes[external_name]
            attrs["is_function_like"] = bool(attrs.get("is_function_like", False) or is_function_like)
        if external_name not in is_defined_in_file:
            is_defined_in_file[external_name] = False
        return external_name

    # Pass 2: add call edges, using call_details for polymorphic resolution.
    for fn in valid_functions:
        caller_name = str(fn.get("name") or "")
        if not caller_name:
            continue

        caller_parameters = fn.get("parameters") or []
        caller_sig_hash = _compute_signature_hash(caller_parameters)
        caller_fqn = str(fn.get("fqn") or caller_name)
        caller_unique_fqn = str(fn.get("unique_fqn") or f"{caller_fqn}#{caller_sig_hash}")
        caller_node_id = _make_node_id(caller_unique_fqn, caller_name, caller_sig_hash)
        if not graph.has_node(caller_node_id):
            continue

        # Derive the class scope of this caller from its FQN so we can walk the
        # inheritance graph when resolving virtual method calls.
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
                argument_count = _call_argument_count(detail)
                if not call_name:
                    continue
                is_function_like_call = kind in {
                    "method",
                    "local",
                    "scoped",
                    "function_pointer",
                    "functor",
                    "lambda",
                } or not kind

                if kind == "method":
                    # Use inheritance-aware resolution for virtual/polymorphic calls.
                    resolved_node_ids = _resolve_virtual_call_targets(
                        call_name,
                        caller_class,
                        class_hierarchy,
                        graph,
                        method_to_node_ids,
                        free_function_node_ids_by_name,
                        argument_count,
                    )
                    if resolved_node_ids:
                        for target_node in resolved_node_ids:
                            graph.add_edge(caller_node_id, target_node)
                    else:
                        target_external = _ensure_external_node(
                            call_display,
                            is_function_like=is_function_like_call,
                        )
                        if target_external:
                            graph.add_edge(caller_node_id, target_external)
                else:
                    # Local/scoped calls may refer to overloaded functions.
                    if "::" in call_display:
                        target_node_ids = _filter_nodes_by_arity(
                            graph,
                            fqn_to_defined_node_ids.get(call_display, []),
                            argument_count,
                        )
                        if target_node_ids:
                            for target_node_id in target_node_ids:
                                graph.add_edge(caller_node_id, target_node_id)
                        else:
                            target_external = _ensure_external_node(
                                call_display,
                                is_function_like=is_function_like_call,
                            )
                            if target_external:
                                graph.add_edge(caller_node_id, target_external)
                        continue

                    overload_targets = _filter_nodes_by_arity(
                        graph,
                        name_to_defined_node_ids.get(call_name, []),
                        argument_count,
                    )
                    if overload_targets:
                        for target_node_id in overload_targets:
                            graph.add_edge(caller_node_id, target_node_id)
                        continue

                    target_external = _ensure_external_node(
                        call_name,
                        is_function_like=is_function_like_call,
                    )
                    if target_external:
                        graph.add_edge(caller_node_id, target_external)
        else:
            # Legacy path: functions_info without call_details (simple calls list).
            raw_calls = fn.get("calls") or []
            for callee in raw_calls:
                callee_name = str(callee or "")
                if not callee_name:
                    continue

                overload_targets = name_to_defined_node_ids.get(callee_name, [])
                if overload_targets:
                    for target_node_id in overload_targets:
                        graph.add_edge(caller_node_id, target_node_id)
                else:
                    target_external = _ensure_external_node(callee_name, is_function_like=True)
                    if target_external:
                        graph.add_edge(caller_node_id, target_external)

    dependency_map: Dict[str, List[str]] = {}
    for fn_name in defined_function_names:
        neighbors: Set[str] = set()
        for caller_node_id in name_to_defined_node_ids.get(fn_name, []):
            for successor in graph.successors(caller_node_id):
                successor_attrs = graph.nodes[successor]
                successor_name = str(successor_attrs.get("name") or successor)
                if successor_name:
                    neighbors.add(successor_name)
        dependency_map[fn_name] = sorted(neighbors)

    return graph, dependency_map, is_defined_in_file, defined_function_names


def analyze_dependency_graph(
    graph: nx.DiGraph,
    defined_function_names: List[str],
    is_defined_in_file: Dict[str, bool],
) -> Dict[str, Any]:
    orphans: List[str] = []
    entry_points: List[str] = []

    name_to_nodes: Dict[str, List[str]] = {}
    for node_id in graph.nodes:
        attrs = graph.nodes[node_id]
        if bool(attrs.get("is_defined_in_file", False)):
            name = str(attrs.get("name") or "")
            if name:
                name_to_nodes.setdefault(name, []).append(str(node_id))

    for fn_name in defined_function_names:
        node_ids = name_to_nodes.get(fn_name, [])
        has_incoming = any(graph.in_degree(node_id) > 0 for node_id in node_ids)
        if not has_incoming:
            if fn_name == "main":
                entry_points.append(fn_name)
            else:
                orphans.append(fn_name)

    cycles: List[List[str]] = []
    for cycle in nx.simple_cycles(graph):
        if not cycle:
            continue
        if not all(bool(graph.nodes[node_id].get("is_defined_in_file", False)) for node_id in cycle):
            continue
        cycles.append([_node_display_name(graph, str(node_id)) for node_id in cycle])

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
    """Dependency graph built from one parser snapshot.

    Thread-safety: this class is not designed for concurrent mutation across
    threads. Create separate instances per request/run.
    """

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
        internal_node_ids = {
            str(node_id)
            for node_id in self.graph.nodes
            if bool(self.graph.nodes[node_id].get("is_defined_in_file", False))
        }
        total_internal = len(internal_node_ids)
        scores: Dict[str, float] = {str(node): 0.0 for node in self.graph.nodes}
        if total_internal == 0:
            return scores

        for node_id in internal_node_ids:
            downstream = nx.descendants(self.graph, node_id)
            downstream_internal = len({str(n) for n in downstream} & internal_node_ids)
            scores[node_id] = downstream_internal / float(total_internal)

        return scores

    def analyze(self) -> Dict[str, Any]:
        return analyze_dependency_graph(
            self.graph, self.defined_function_names, self.is_defined_in_file
        )

    def get_impact_radius(self, function_name: str) -> float:
        direct = self._criticality_scores.get(function_name)
        if direct is not None:
            return float(direct)

        node_scores: List[float] = []
        for node_id in self.graph.nodes:
            attrs = self.graph.nodes[node_id]
            if str(attrs.get("name") or "") == function_name and bool(attrs.get("is_defined_in_file", False)):
                node_scores.append(float(self._criticality_scores.get(str(node_id), 0.0)))
        if not node_scores:
            return 0.0
        return max(node_scores)

    def get_bottlenecks(self) -> List[str]:
        internal_node_ids = [
            str(node_id)
            for node_id in self.graph.nodes
            if bool(self.graph.nodes[node_id].get("is_defined_in_file", False))
        ]
        if not internal_node_ids:
            return []

        in_degrees = {node_id: self.graph.in_degree(node_id) for node_id in internal_node_ids}
        max_degree = max(in_degrees.values(), default=0)
        if max_degree == 0:
            return []

        names = [
            str(self.graph.nodes[node_id].get("name") or node_id)
            for node_id, degree in in_degrees.items()
            if degree == max_degree
        ]
        return sorted(_ordered_unique(names))

    def get_modernization_order(self) -> List[str]:
        """Return modernization order as internal node IDs.

        Important: IDs are expected to align with parser-level function keys
        (typically ``unique_fqn``). Downstream workflow code relies on this.
        """
        internal_node_ids = {
            str(node_id)
            for node_id in self.graph.nodes
            if bool(self.graph.nodes[node_id].get("is_defined_in_file", False))
        }
        if not internal_node_ids:
            return []

        ordered_node_ids = _scc_modernization_order(self.graph, internal_node_ids)
        if not ordered_node_ids:
            return sorted(_ordered_unique([str(node_id) for node_id in internal_node_ids]))

        return _ordered_unique([str(node_id) for node_id in ordered_node_ids])

    def get_dependency_levels(self) -> List[List[str]]:
        """Return dependency-respecting function layers for parallel modernization.

        Each inner list contains functions that can be modernized concurrently.
        Later layers depend on earlier layers.
        """
        internal_node_ids = {
            str(node_id)
            for node_id in self.graph.nodes
            if bool(self.graph.nodes[node_id].get("is_defined_in_file", False))
        }
        if not internal_node_ids:
            return []

        internal_subgraph = self.graph.subgraph(internal_node_ids).copy()
        if internal_subgraph.number_of_nodes() == 0:
            return []

        condensed = nx.condensation(internal_subgraph)
        schedule_graph = condensed.reverse()

        component_level: Dict[int, int] = {}
        for component in nx.topological_sort(schedule_graph):
            predecessors = list(schedule_graph.predecessors(component))
            # schedule_graph is reversed, so predecessors here are dependency
            # components that must execute earlier; therefore we use max(pred)+1.
            if not predecessors:
                component_level[int(component)] = 0
            else:
                component_level[int(component)] = max(component_level[int(pred)] for pred in predecessors) + 1

        levels_by_index: Dict[int, List[str]] = {}
        for component, level_idx in component_level.items():
            members = condensed.nodes[component].get("members") or set()
            member_names = sorted(
                {
                    str(internal_subgraph.nodes[str(node_id)].get("name") or node_id)
                    for node_id in members
                }
            )
            levels_by_index.setdefault(level_idx, []).extend(member_names)

        levels: List[List[str]] = []
        for level_idx in sorted(levels_by_index.keys()):
            level_names = sorted(_ordered_unique(levels_by_index[level_idx]))
            if level_names:
                levels.append(level_names)
        return levels

    def to_dict(self) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        for node in self.graph.nodes:
            node_id = str(node)
            attrs = self.graph.nodes[node_id]
            nodes.append(
                {
                    "node_id": str(attrs.get("node_id") or node_id),
                    "name": str(attrs.get("name") or node_id),
                    "display_name": str(attrs.get("display_name") or attrs.get("name") or node_id),
                    "fqn": str(attrs.get("fqn") or attrs.get("name") or node_id),
                    "signature_hash": str(attrs.get("signature_hash") or ""),
                    "is_virtual": bool(attrs.get("is_virtual", False)),
                    "is_defined_in_file": bool(attrs.get("is_defined_in_file", False)),
                    "criticality_score": float(
                        self._criticality_scores.get(node_id, 0.0)
                    ),
                }
            )

        edges = [
            {"from": str(u), "to": str(v)}
            for u, v in self.graph.edges
        ]
        return {"nodes": nodes, "edges": edges}


def get_modernization_order(dependency_map: Dict[str, List[str]]) -> List[str]:
    """Return modernization order from a name-level dependency map.

    Warning: this helper is NOT overload-aware because ``dependency_map`` keys
    are simple names. Prefer ``DependencyGraph.get_modernization_order`` when
    overload-level precision is required.
    """
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

    # External nodes are excluded from modernization ordering.
    internal_subgraph = nx.DiGraph()
    for caller, callees in dependency_map.items():
        internal_subgraph.add_node(caller)
        for callee in callees:
            callee_name = str(callee or "")
            if callee_name in internal_nodes:
                internal_subgraph.add_edge(caller, callee_name)

    try:
        condensed = nx.condensation(internal_subgraph)
        topo_components = list(nx.topological_sort(condensed.reverse()))
        ordered: List[str] = []
        for component in topo_components:
            members = condensed.nodes[component].get("members") or set()
            ordered.extend(sorted(str(member) for member in members if str(member) in internal_nodes))
        remaining = [name for name in sorted(internal_nodes) if name not in ordered]
        ordered.extend(remaining)
        return ordered
    except Exception:
        return sorted(internal_nodes)


if __name__ == "__main__":
    demo_functions = [
        {
            "name": "foo",
            "fqn": "A::foo",
            "unique_fqn": "A::foo#1111111111111111",
            "signature_hash": "1111111111111111",
            "parameters": [{"type": "int"}],
            "call_details": [{"kind": "method", "name": "bar", "arguments": [{"expr": "x"}]}],
        },
        {
            "name": "bar",
            "fqn": "A::bar",
            "unique_fqn": "A::bar#2222222222222222",
            "signature_hash": "2222222222222222",
            "parameters": [{"type": "int"}],
            "call_details": [],
        },
    ]
    dep = DependencyGraph(demo_functions, types_info=[])
    print(json.dumps(dep.to_dict(), indent=2))
    print("Modernization order:", dep.get_modernization_order())