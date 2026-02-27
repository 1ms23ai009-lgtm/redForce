"""Strategic digest generator - main entry point.

Implements GenerateDigest() from Algorithm 1 of the G-CTR paper,
adapted for LLM attack context. Coordinates between algorithmic
and LLM digest modes with automatic fallback.
"""

import logging
from typing import Optional

import networkx as nx

from redforge.digest.algorithmic import generate_algorithmic_digest
from redforge.digest.llm_digest import generate_llm_digest
from redforge.graph.attack_graph import deserialize_graph
from redforge.core.state import AttackStateObject

logger = logging.getLogger(__name__)


def generate_digest(
    attack_graph: nx.DiGraph,
    nash_result: dict,
    mode: str = "llm",
    aso: Optional[AttackStateObject] = None,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    strategy_library_context: Optional[str] = None,
) -> str:
    """Generate a strategic digest for specialist agents.

    Algorithm:
    1. Extract attack paths with probabilities from nash_result
    2. Identify critical nodes (high variance decision points)
    3. Identify bottlenecks: paths with probability < 0.5
    4. Identify high-risk transitions: probability > 0.9
    5. If mode == "algorithmic": use template formatter
    6. If mode == "llm": send to judge LLM with structured prompt
    7. Return markdown digest string

    Automatic fallback to algorithmic mode on LLM failure.

    Args:
        attack_graph: Current attack graph (NetworkX DiGraph)
        nash_result: Nash equilibrium computation result
        mode: "llm" or "algorithmic"
        aso: Optional ASO for context
        model_name: LLM model for digest generation
        api_key: Optional API key
        strategy_library_context: Optional strategy suggestion

    Returns:
        Markdown-formatted strategic digest string
    """
    if not nash_result:
        return _empty_digest()

    if mode == "algorithmic":
        return generate_algorithmic_digest(
            attack_graph, nash_result, strategy_library_context
        )

    if mode == "llm":
        # Build ASO summary for context
        aso_summary = _build_aso_summary(aso) if aso else None

        digest = generate_llm_digest(
            attack_graph=attack_graph,
            nash_result=nash_result,
            aso_summary=aso_summary,
            model_name=model_name,
            api_key=api_key,
        )

        if digest:
            return digest

        # Automatic fallback to algorithmic mode
        logger.warning("LLM digest failed, falling back to algorithmic mode")
        return generate_algorithmic_digest(
            attack_graph, nash_result, strategy_library_context
        )

    raise ValueError(f"Invalid digest mode: {mode}. Must be 'llm' or 'algorithmic'.")


def generate_digest_from_aso(
    aso: AttackStateObject,
    mode: str = "llm",
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> str:
    """Convenience function to generate digest directly from ASO.

    Deserializes the attack graph and uses the stored Nash equilibrium.
    """
    graph = deserialize_graph(aso["attack_graph"])
    nash_result = aso.get("nash_equilibrium")

    if not nash_result:
        return _empty_digest()

    return generate_digest(
        attack_graph=graph,
        nash_result=nash_result,
        mode=mode,
        aso=aso,
        model_name=model_name,
        api_key=api_key,
    )


def _build_aso_summary(aso: AttackStateObject) -> dict:
    """Build a concise summary of the ASO for digest context."""
    return {
        "turns_completed": len(aso.get("attack_history", [])),
        "exploits_found": len(aso.get("confirmed_exploits", [])),
        "partial_bypasses": len(aso.get("partial_bypass_registry", [])),
        "budget_remaining": aso.get("query_budget_remaining", 0),
        "active_hypotheses_count": len(aso.get("active_hypotheses", [])),
        "coverage": {
            cat: state.get("tested", False)
            for cat, state in aso.get("coverage_state", {}).items()
        },
        "top_hypotheses": [
            {
                "description": h.get("description", ""),
                "confidence": h.get("confidence", 0.0),
                "recommended_agent": h.get("recommended_next_agent", ""),
            }
            for h in sorted(
                aso.get("active_hypotheses", []),
                key=lambda x: x.get("priority_score", 0.0),
                reverse=True,
            )[:3]
        ],
    }


def _empty_digest() -> str:
    """Return an empty digest when no Nash data is available."""
    return (
        "## REDFORGE Strategic Digest\n\n"
        "No Nash equilibrium data available yet. "
        "Continue exploring the target to build the attack graph."
    )
