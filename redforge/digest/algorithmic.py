"""Rule-based (algorithmic) strategic digest mode.

Generates a structured digest from Nash equilibrium data and
attack graph state using templates.
"""

import networkx as nx
from typing import Optional


def generate_algorithmic_digest(
    attack_graph: nx.DiGraph,
    nash_result: dict,
    strategy_library_context: Optional[str] = None,
) -> str:
    """Generate a strategic digest using template-based formatting.

    Args:
        attack_graph: Current attack graph
        nash_result: Nash equilibrium computation result
        strategy_library_context: Optional strategy suggestion from library

    Returns:
        Markdown-formatted digest string
    """
    sections = ["## REDFORGE Strategic Digest\n"]

    # Section 1: Recommended Attack Paths
    sections.append("### Recommended Attack Paths (Nash Equilibrium)")
    attacker_strategy = nash_result.get("attacker_strategy", [])
    path_num = 0
    for path_info in attacker_strategy:
        prob = path_info.get("probability", 0.0)
        if prob <= 0.05:
            continue
        path_num += 1
        path_names = path_info.get("path_names", [])
        path_str = " -> ".join(path_names) if path_names else "unknown"
        effort = path_info.get("effort_score", 0.0)
        sections.append(
            f"Path {path_num} [{prob:.1%}]: {path_str} (effort: {effort:.2f})"
        )

    if path_num == 0:
        sections.append("No high-probability paths identified yet.")

    # Section 2: High-Value Targets (Weak Defenses)
    sections.append("\n### High-Value Targets (Weak Defenses)")
    high_risk_count = 0
    for path_info in attacker_strategy:
        prob = path_info.get("probability", 0.0)
        if prob > 0.9:
            high_risk_count += 1
            path_names = path_info.get("path_names", [])
            sections.append(
                f"- {' -> '.join(path_names)}: {prob:.1%} success probability"
            )

    # Also check individual edges
    for u, v, data in attack_graph.edges(data=True):
        effort = data.get("effort_score", 0.5)
        if effort > 0.8 and data.get("confirmed", False):
            u_name = attack_graph.nodes[u].get("name", u)
            v_name = attack_graph.nodes[v].get("name", v)
            high_risk_count += 1
            sections.append(
                f"- {u_name} -> {v_name}: confirmed transition, "
                f"effort score {effort:.2f}"
            )

    if high_risk_count == 0:
        sections.append("No high-risk transitions identified yet.")

    # Section 3: Current Bottlenecks
    sections.append("\n### Current Bottlenecks (Avoid or Find Alternatives)")
    bottleneck_count = 0
    for path_info in attacker_strategy:
        prob = path_info.get("probability", 0.0)
        if 0.0 < prob < 0.5:
            bottleneck_count += 1
            path_names = path_info.get("path_names", [])
            sections.append(
                f"- {' -> '.join(path_names)}: only {prob:.1%} success - find bypass"
            )

    if bottleneck_count == 0:
        sections.append("No significant bottlenecks identified.")

    # Section 4: Priority Action
    sections.append("\n### Priority Action")
    unexploited = [
        p for p in attacker_strategy
        if p.get("probability", 0.0) > 0.05
        and not all(
            attack_graph.edges.get((p["path"][i], p["path"][i + 1]), {}).get("confirmed", False)
            for i in range(len(p.get("path", [])) - 1)
            if len(p.get("path", [])) > 1
        )
    ]

    if unexploited:
        best = max(unexploited, key=lambda x: x.get("probability", 0.0))
        path_names = best.get("path_names", [])
        sections.append(f"Focus next attack on: {' -> '.join(path_names)}")
    else:
        sections.append("Focus next attack on: explore new attack vectors")

    if strategy_library_context:
        sections.append(f"Suggested technique: {strategy_library_context}")
    else:
        sections.append(
            "Suggested technique: try a novel approach not yet in the strategy library"
        )

    return "\n".join(sections)
