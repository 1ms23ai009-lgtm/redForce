"""Exploit report generator.

Generates structured reports from engagement results including
executive summary, exploit details, attack graph analysis, and
OWASP coverage.
"""

import json
import logging
from datetime import datetime, timezone

from redforge.core.state import AttackStateObject, get_current_turn
from redforge.core.mdp import compute_coverage_percentage
from redforge.reporting.risk_scorer import (
    compute_composite_risk_score,
    compute_exploit_risk_score,
    compute_owasp_coverage,
)
from redforge.graph.attack_graph import deserialize_graph, get_graph_summary, get_attack_paths

logger = logging.getLogger(__name__)

REMEDIATION_TEMPLATES = {
    "jailbreak": "Strengthen system prompt boundaries and implement layered instruction hierarchy enforcement.",
    "prompt_injection_direct": "Implement input sanitization and instruction-data separation in prompt construction.",
    "prompt_injection_indirect": "Add content validation on RAG-retrieved documents and implement sandboxing of external content.",
    "cross_agent_injection": "Implement message authentication between agents and enforce strict inter-agent communication protocols.",
    "data_exfiltration": "Restrict system prompt visibility, implement output filtering, and sanitize tool configurations.",
    "tool_abuse": "Implement strict tool input validation, output sanitization, and least-privilege tool access policies.",
    "memory_poisoning": "Add content policy enforcement on memory write operations and implement memory isolation per user.",
    "social_engineering": "Implement conversation-level safety monitoring and rate-limit escalating request patterns.",
}


def generate_report(aso: AttackStateObject) -> dict:
    """Generate a comprehensive engagement report.

    Args:
        aso: Final attack state object

    Returns:
        Structured report dict
    """
    report = {
        "executive_summary": _build_executive_summary(aso),
        "confirmed_exploits": _build_exploit_details(aso),
        "partial_bypasses": _build_partial_bypasses(aso),
        "attack_graph_summary": _build_graph_summary(aso),
        "owasp_coverage": compute_owasp_coverage(aso),
        "methodology": _build_methodology(aso),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return report


def _build_executive_summary(aso: AttackStateObject) -> dict:
    """Build the executive summary section."""
    exploits = aso.get("confirmed_exploits", [])
    history = aso.get("attack_history", [])

    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for exploit in exploits:
        sev = exploit.get("severity", "medium")
        if sev in severity_counts:
            severity_counts[sev] += 1

    highest = "none"
    for sev in ["critical", "high", "medium", "low"]:
        if severity_counts.get(sev, 0) > 0:
            highest = sev
            break

    total_tokens = sum(e.get("tokens_used", 0) for e in history)
    total_cost = sum(e.get("cost_usd", 0.0) for e in history)

    # Compute duration from audit log
    audit_log = aso.get("audit_log", [])
    start_time = None
    end_time = None
    for entry in audit_log:
        if entry.get("event_type") == "engagement_start":
            start_time = entry.get("timestamp")
        if entry.get("event_type") == "engagement_end":
            end_time = entry.get("timestamp")

    duration = "unknown"
    if start_time and end_time:
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            delta = end - start
            duration = str(delta)
        except Exception:
            pass

    return {
        "target_description": (
            f"{aso['target_config'].get('model_name', 'unknown')} at "
            f"{aso['target_config'].get('endpoint_url', 'unknown')}"
        ),
        "engagement_duration": duration,
        "total_queries_used": len(history),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "exploits_found": severity_counts,
        "total_exploits": len(exploits),
        "partial_bypasses": len(aso.get("partial_bypass_registry", [])),
        "owasp_coverage_percentage": round(compute_coverage_percentage(aso), 1),
        "highest_severity_finding": highest,
        "overall_risk_score": round(compute_composite_risk_score(aso), 2),
    }


def _build_exploit_details(aso: AttackStateObject) -> list[dict]:
    """Build detailed exploit entries."""
    details = []
    for exploit in aso.get("confirmed_exploits", []):
        category = exploit.get("category", "unknown")
        details.append({
            "exploit_id": exploit.get("exploit_id", ""),
            "title": f"{category.replace('_', ' ').title()} Exploit",
            "severity": exploit.get("severity", "medium"),
            "owasp_categories": exploit.get("owasp_categories", []),
            "description": _get_exploit_description(exploit, aso),
            "attack_chain_length": len(exploit.get("attack_chain", [])),
            "reproducible_trace": exploit.get("reproducible_trace", []),
            "judge_verdicts": exploit.get("judge_verdicts", []),
            "risk_score": round(compute_exploit_risk_score(exploit), 2),
            "remediation_recommendation": REMEDIATION_TEMPLATES.get(
                category,
                "Review and strengthen the relevant guardrails and access controls.",
            ),
        })

    # Sort by risk score descending
    details.sort(key=lambda x: x["risk_score"], reverse=True)
    return details


def _build_partial_bypasses(aso: AttackStateObject) -> list[dict]:
    """Build partial bypass entries."""
    return [
        {
            "description": bp.get("description", ""),
            "affected_guardrail": bp.get("guardrail_affected", ""),
            "bypass_degree": bp.get("bypass_degree", 0.0),
            "potential_chaining_risk": bp.get("exploitable_by", []),
        }
        for bp in aso.get("partial_bypass_registry", [])
    ]


def _build_graph_summary(aso: AttackStateObject) -> dict:
    """Build attack graph summary."""
    try:
        graph = deserialize_graph(aso.get("attack_graph", {}))
        summary = get_graph_summary(graph)

        paths = get_attack_paths(graph)
        critical_paths = []
        for path in paths[:3]:
            path_names = [
                graph.nodes[n].get("name", n) for n in path if n in graph
            ]
            critical_paths.append(path_names)

        nash = aso.get("nash_equilibrium", {})
        eq_value = nash.get("equilibrium_value", 0.0) if nash else 0.0

        summary["critical_paths"] = critical_paths
        summary["nash_equilibrium_value"] = round(eq_value, 4)
        return summary
    except Exception as e:
        return {"error": str(e)}


def _build_methodology(aso: AttackStateObject) -> dict:
    """Build methodology section."""
    history = aso.get("attack_history", [])

    agents_deployed = set()
    strategies_tried = set()
    for entry in history:
        agents_deployed.add(entry.get("agent_name", "unknown"))
        strategies_tried.add(entry.get("strategy_category", "unknown"))

    novel_count = sum(
        1 for hit in aso.get("strategy_library_hits", [])
        if hit.get("similarity_score", 1.0) < 0.7
    )

    return {
        "agents_deployed": sorted(agents_deployed),
        "strategies_tried": sorted(strategies_tried),
        "total_attacks": len(history),
        "novel_exploits_count": novel_count,
        "strategy_library_entries_added": len(aso.get("strategy_library_hits", [])),
    }


def _get_exploit_description(exploit: dict, aso: AttackStateObject) -> str:
    """Generate a human-readable exploit description."""
    category = exploit.get("category", "unknown")
    severity = exploit.get("severity", "medium")
    chain_len = len(exploit.get("attack_chain", []))

    verdicts = exploit.get("judge_verdicts", [])
    explanation = ""
    if verdicts and isinstance(verdicts[0], dict):
        explanation = verdicts[0].get("explanation", "")

    return (
        f"A {severity}-severity {category.replace('_', ' ')} vulnerability was confirmed. "
        f"The exploit required {chain_len} step(s). "
        f"{explanation}"
    )
