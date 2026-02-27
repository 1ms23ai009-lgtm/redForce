"""Attack graph data structure using NetworkX DiGraph.

Represents the discovered attack surface of the target system with
nodes for system components and edges for exploitable transitions.
"""

import uuid
import json
from typing import Optional

import networkx as nx


# Valid node types
NODE_TYPES = frozenset([
    "system_prompt",
    "guardrail",
    "tool",
    "agent",
    "memory",
    "policy",
    "conversation_state",
    "user_input",
])

# Valid edge types
EDGE_TYPES = frozenset([
    "exploitable_transition",
    "information_flow",
    "trust_relationship",
    "constraint",
    "bypass_path",
])


def create_attack_graph() -> nx.DiGraph:
    """Create an empty attack graph."""
    return nx.DiGraph()


def add_node(
    graph: nx.DiGraph,
    name: str,
    node_type: str,
    message_id: int = 0,
    vulnerability: bool = False,
    vulnerability_severity: Optional[str] = None,
) -> str:
    """Add a node to the attack graph.

    Returns:
        The node_id of the added node.
    """
    if node_type not in NODE_TYPES:
        raise ValueError(f"Invalid node_type: {node_type}. Must be one of {NODE_TYPES}")

    node_id = str(uuid.uuid4())
    graph.add_node(
        node_id,
        name=name,
        node_type=node_type,
        vulnerability=vulnerability,
        vulnerability_severity=vulnerability_severity,
        message_id=message_id,
        bypass_attempts=0,
        bypass_successes=0,
    )
    return node_id


def add_edge(
    graph: nx.DiGraph,
    source_id: str,
    target_id: str,
    edge_type: str,
    effort_score: float = 0.5,
    confirmed: bool = False,
    attack_technique: Optional[str] = None,
) -> str:
    """Add an edge to the attack graph.

    Returns:
        The edge_id of the added edge.
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Invalid edge_type: {edge_type}. Must be one of {EDGE_TYPES}")
    if source_id not in graph:
        raise ValueError(f"Source node {source_id} not in graph")
    if target_id not in graph:
        raise ValueError(f"Target node {target_id} not in graph")

    edge_id = str(uuid.uuid4())
    graph.add_edge(
        source_id,
        target_id,
        edge_id=edge_id,
        edge_type=edge_type,
        effort_score=max(0.0, min(1.0, effort_score)),
        confirmed=confirmed,
        attack_technique=attack_technique,
    )
    return edge_id


def mark_node_vulnerable(
    graph: nx.DiGraph,
    node_id: str,
    severity: str,
) -> None:
    """Mark a node as having a confirmed vulnerability."""
    if node_id not in graph:
        raise ValueError(f"Node {node_id} not in graph")
    graph.nodes[node_id]["vulnerability"] = True
    graph.nodes[node_id]["vulnerability_severity"] = severity


def increment_bypass_attempt(graph: nx.DiGraph, node_id: str, success: bool = False) -> None:
    """Record a bypass attempt on a node."""
    if node_id not in graph:
        return
    graph.nodes[node_id]["bypass_attempts"] += 1
    if success:
        graph.nodes[node_id]["bypass_successes"] += 1


def confirm_edge(graph: nx.DiGraph, source_id: str, target_id: str) -> None:
    """Mark an edge as confirmed (transition was successfully exploited)."""
    if graph.has_edge(source_id, target_id):
        graph.edges[source_id, target_id]["confirmed"] = True


def get_vulnerable_nodes(graph: nx.DiGraph) -> list[str]:
    """Get all node IDs that have confirmed vulnerabilities."""
    return [
        nid for nid, data in graph.nodes(data=True)
        if data.get("vulnerability", False)
    ]


def get_attack_paths(graph: nx.DiGraph) -> list[list[str]]:
    """Get all simple paths from entry points to vulnerable nodes.

    Entry points are nodes with node_type 'user_input' or no predecessors.
    """
    vulnerable = get_vulnerable_nodes(graph)
    if not vulnerable:
        return []

    # Entry points: nodes with no predecessors or user_input type
    entry_points = [
        nid for nid in graph.nodes()
        if graph.in_degree(nid) == 0
        or graph.nodes[nid].get("node_type") == "user_input"
    ]

    paths = []
    for entry in entry_points:
        for vuln in vulnerable:
            if entry == vuln:
                paths.append([entry])
                continue
            try:
                for path in nx.all_simple_paths(graph, entry, vuln, cutoff=10):
                    paths.append(path)
            except nx.NetworkXError:
                continue
    return paths


def serialize_graph(graph: nx.DiGraph) -> dict:
    """Serialize the graph to a JSON-compatible dict."""
    return nx.node_link_data(graph)


def deserialize_graph(data: dict) -> nx.DiGraph:
    """Deserialize a graph from a JSON-compatible dict."""
    if not data or (not data.get("nodes") and not data.get("links")):
        return create_attack_graph()
    return nx.node_link_graph(data)


def remove_cycles(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove cycles from the graph to make it a DAG.

    Uses feedback arc set heuristic to remove minimum edges.
    """
    if nx.is_directed_acyclic_graph(graph):
        return graph

    # Find cycles and remove back edges
    try:
        cycles = list(nx.simple_cycles(graph))
        edges_to_remove = set()
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            # Remove the edge with highest effort score in each cycle
            worst_edge = None
            worst_effort = -1
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if graph.has_edge(u, v):
                    effort = graph.edges[u, v].get("effort_score", 0.5)
                    if effort > worst_effort:
                        worst_effort = effort
                        worst_edge = (u, v)
            if worst_edge:
                edges_to_remove.add(worst_edge)

        for u, v in edges_to_remove:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
    except Exception:
        pass

    return graph


def add_artificial_leaves(graph: nx.DiGraph) -> None:
    """Add artificial leaf nodes to vulnerable nodes (G-CTR post-processing)."""
    vulnerable = get_vulnerable_nodes(graph)
    for node_id in vulnerable:
        leaf_id = str(uuid.uuid4())
        severity = graph.nodes[node_id].get("vulnerability_severity", "medium")
        graph.add_node(
            leaf_id,
            name=f"exploit_outcome_{graph.nodes[node_id]['name']}",
            node_type="conversation_state",
            vulnerability=False,
            vulnerability_severity=None,
            message_id=graph.nodes[node_id].get("message_id", 0),
            bypass_attempts=0,
            bypass_successes=0,
        )
        graph.add_edge(
            node_id,
            leaf_id,
            edge_id=str(uuid.uuid4()),
            edge_type="exploitable_transition",
            effort_score=0.1,
            confirmed=True,
            attack_technique=f"confirmed_{severity}_exploit",
        )


def connect_components(graph: nx.DiGraph) -> None:
    """Connect disconnected components to a root node."""
    if graph.number_of_nodes() == 0:
        return

    components = list(nx.weakly_connected_components(graph))
    if len(components) <= 1:
        return

    # Find or create root
    root_candidates = [
        n for n in graph.nodes()
        if graph.nodes[n].get("node_type") == "user_input"
    ]

    if root_candidates:
        root_id = root_candidates[0]
    else:
        root_id = add_node(graph, "root_entry", "user_input")

    root_component = None
    for comp in components:
        if root_id in comp:
            root_component = comp
            break

    for comp in components:
        if comp is root_component:
            continue
        # Connect root to a node in this component
        target = next(iter(comp))
        add_edge(graph, root_id, target, "information_flow", effort_score=0.5)


def prune_nonvulnerable_leaves(graph: nx.DiGraph) -> None:
    """Remove non-vulnerable leaf nodes (G-CTR post-processing)."""
    changed = True
    while changed:
        changed = False
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        for leaf in leaves:
            if not graph.nodes[leaf].get("vulnerability", False):
                # Don't prune if it's an artificial exploit outcome node
                if graph.nodes[leaf].get("name", "").startswith("exploit_outcome_"):
                    continue
                graph.remove_node(leaf)
                changed = True


def get_graph_summary(graph: nx.DiGraph) -> dict:
    """Get a summary of the attack graph."""
    vulnerable = get_vulnerable_nodes(graph)
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "vulnerable_nodes": len(vulnerable),
        "confirmed_edges": sum(
            1 for _, _, d in graph.edges(data=True) if d.get("confirmed", False)
        ),
        "node_types": dict(
            sorted(
                {
                    t: sum(1 for _, d in graph.nodes(data=True) if d.get("node_type") == t)
                    for t in NODE_TYPES
                }.items()
            )
        ),
    }
