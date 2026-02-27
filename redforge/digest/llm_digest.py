"""LLM-interpreted strategic digest mode.

Uses a separate LLM to interpret Nash equilibrium data and produce
tactical guidance for specialist agents.
"""

import json
import logging
from typing import Optional

import networkx as nx
from langchain_core.messages import SystemMessage, HumanMessage

from redforge.graph.attack_graph import get_graph_summary

logger = logging.getLogger(__name__)

DIGEST_SYSTEM_PROMPT = """You are a strategic advisor for an authorized AI red-teaming system. \
Your role is to interpret game-theoretic analysis of an attack graph and produce \
clear, actionable tactical guidance for specialist attack agents.

You will receive:
1. Nash equilibrium data showing optimal attack/defense strategies
2. Attack graph summary showing discovered nodes and edges
3. Current engagement state

Produce a strategic digest with EXACTLY these 5 sections:

## Attack Paths
Describe the top recommended attack paths based on Nash equilibrium probabilities. \
For each path, explain what it targets and why it's promising.

## Bottlenecks
Identify where the attacker is struggling. Which transitions have low success \
probability? What alternative approaches should be considered?

## Critical Nodes
Which nodes in the graph are decision points? Where does the attacker's choice \
matter most? Highlight nodes with high variance in outcomes.

## High-Risk Transitions
Which transitions are near-certain successes? These represent weak defenses \
that should be exploited. Be specific about the vulnerability.

## Tactical Guidance
Concrete next steps. What specific technique should the next specialist agent \
use? Consider the attack history and what has/hasn't worked.

Keep the entire digest under 500 words. Be specific and actionable. \
Do not use generic advice."""


def generate_llm_digest(
    attack_graph: nx.DiGraph,
    nash_result: dict,
    aso_summary: Optional[dict] = None,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    provider: str = "openai",
    vertex_project: Optional[str] = None,
    vertex_location: str = "us-central1",
) -> str:
    """Generate a strategic digest using an LLM interpreter.

    Args:
        attack_graph: Current attack graph
        nash_result: Nash equilibrium computation result
        aso_summary: Optional summary of current ASO state
        model_name: LLM model to use for digest generation
        api_key: Optional API key override
        provider: LLM provider ("openai", "vertex", etc.)
        vertex_project: GCP project ID (for Vertex AI)
        vertex_location: GCP region (for Vertex AI)

    Returns:
        Markdown-formatted digest string
    """
    # Prepare context for the LLM
    context = _build_context(attack_graph, nash_result, aso_summary)

    try:
        from redforge.llm import create_langchain_llm
        llm = create_langchain_llm(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=0.3,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )
        messages = [
            SystemMessage(content=DIGEST_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.warning(f"LLM digest generation failed: {e}. Returning empty digest.")
        return ""


def _build_context(
    attack_graph: nx.DiGraph,
    nash_result: dict,
    aso_summary: Optional[dict] = None,
) -> str:
    """Build context string for the digest LLM."""
    parts = []

    # Nash equilibrium data
    parts.append("## Nash Equilibrium Data")
    parts.append(f"Equilibrium value: {nash_result.get('equilibrium_value', 'N/A')}")

    attacker_strategy = nash_result.get("attacker_strategy", [])
    parts.append(f"Number of attack paths: {len(attacker_strategy)}")
    for i, path in enumerate(attacker_strategy[:5]):
        names = path.get("path_names", [])
        prob = path.get("probability", 0.0)
        effort = path.get("effort_score", 0.0)
        parts.append(
            f"  Path {i+1}: {' -> '.join(names)} "
            f"(prob: {prob:.3f}, effort: {effort:.3f})"
        )

    defender = nash_result.get("defender_strategy", {})
    if defender:
        parts.append("Defender focus nodes:")
        sorted_defender = sorted(defender.items(), key=lambda x: x[1], reverse=True)
        for node_id, prob in sorted_defender[:5]:
            name = attack_graph.nodes[node_id].get("name", node_id) if node_id in attack_graph else node_id
            parts.append(f"  {name}: {prob:.3f}")

    # Graph summary
    parts.append("\n## Attack Graph Summary")
    summary = get_graph_summary(attack_graph)
    parts.append(json.dumps(summary, indent=2))

    # ASO summary
    if aso_summary:
        parts.append("\n## Current Engagement State")
        parts.append(json.dumps(aso_summary, indent=2, default=str))

    return "\n".join(parts)
