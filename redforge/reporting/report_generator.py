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
    "jailbreak": {
        "title": "Jailbreak via Prompt Manipulation",
        "description": "The model's safety guardrails were bypassed through prompt manipulation techniques.",
        "fixes": [
            "Implement a layered instruction hierarchy with clear system/user message separation",
            "Add a second-pass safety classifier on model outputs before returning to users",
            "Use an LLM-based output filter specifically tuned for jailbreak detection",
            "Monitor for known jailbreak patterns (DAN, role-play, fictional framing) in input preprocessing",
            "Regularly update safety training data with newly discovered jailbreak techniques",
        ],
        "priority": "medium",
    },
    "prompt_injection_direct": {
        "title": "Direct Prompt Injection",
        "description": "Attacker instructions were injected directly via user input and overrode system instructions.",
        "fixes": [
            "Implement strict instruction-data separation in prompt construction",
            "Use delimiters (e.g., XML tags) to clearly delineate system instructions from user input",
            "Add input sanitization to strip or neutralize injection patterns",
            "Use a dedicated input classifier to detect injection attempts",
            "Consider using a separate system prompt channel that cannot be overridden by user messages",
        ],
        "priority": "high",
    },
    "prompt_injection_indirect": {
        "title": "Indirect Prompt Injection via External Content",
        "description": "Malicious instructions were embedded in retrieved documents/external content and executed by the model.",
        "fixes": [
            "Sandbox all external content — treat retrieved documents as untrusted data",
            "Add content validation and sanitization on RAG-retrieved documents before including in context",
            "Implement output monitoring to detect when model behavior changes after retrieving external content",
            "Use separate processing pipelines for system instructions vs. retrieved content",
            "Consider using a guardrail model to evaluate retrieved content before passing to the main model",
        ],
        "priority": "critical",
    },
    "cross_agent_injection": {
        "title": "Cross-Agent Prompt Injection",
        "description": "Attacker exploited inter-agent communication to inject instructions that propagated across the system.",
        "fixes": [
            "Implement message authentication and signing between agents",
            "Enforce strict schema validation on all inter-agent messages",
            "Add role-based access controls for agent-to-agent communication",
            "Monitor for anomalous inter-agent message patterns",
            "Implement privilege separation — each agent should only access its authorized scope",
        ],
        "priority": "critical",
    },
    "data_exfiltration": {
        "title": "Sensitive Information Disclosure",
        "description": "The model leaked sensitive information including system prompts, credentials, or private data.",
        "fixes": [
            "Never include API keys, credentials, or secrets in the model's context window",
            "Implement output filtering to detect and redact credential patterns (API keys, tokens, passwords)",
            "Separate system prompt from retrievable context — use a 'hidden' system prompt that the model cannot echo",
            "Add PII detection on model outputs and redact before returning to users",
            "Implement document-level access controls in RAG systems",
        ],
        "priority": "critical",
    },
    "tool_abuse": {
        "title": "Tool/Plugin Abuse",
        "description": "The LLM was manipulated into misusing its tools to execute unauthorized actions.",
        "fixes": [
            "Implement parameterized queries — never let the LLM construct raw SQL or system commands",
            "Add an allowlist of permitted tool operations and validate all inputs against it",
            "Implement least-privilege access — tools should only have the minimum permissions needed",
            "Add a confirmation step for destructive or sensitive tool operations",
            "Sanitize and validate all tool inputs server-side, not relying on the LLM's judgment",
            "Implement rate limiting and anomaly detection on tool usage patterns",
        ],
        "priority": "critical",
    },
    "rag_poisoning": {
        "title": "RAG Data Poisoning / Leakage",
        "description": "The retrieval system exposed sensitive documents or was manipulated to retrieve unauthorized content.",
        "fixes": [
            "Implement document-level access controls — filter retrievals based on user permissions",
            "Add metadata stripping to remove source paths, file names, and internal tags from retrieved content",
            "Sanitize retrieved documents for embedded instructions before including in context",
            "Implement query validation to prevent keyword manipulation attacks",
            "Add tenant isolation in multi-tenant RAG systems",
        ],
        "priority": "high",
    },
    "memory_poisoning": {
        "title": "Memory Manipulation / Injection",
        "description": "Attacker injected false context or instructions into the model's memory system.",
        "fixes": [
            "Apply content policy checks on memory write operations, not just conversation outputs",
            "Implement per-user memory isolation — prevent cross-user memory access",
            "Validate the role field server-side — users should not be able to inject 'system' or 'assistant' messages",
            "Add integrity checks on stored memory entries",
            "Implement memory expiration and periodic cleanup",
        ],
        "priority": "high",
    },
    "guardrail_bypass": {
        "title": "Safety Guardrail Bypass",
        "description": "Safety filters were bypassed using encoding tricks, multilingual content, or persona switching.",
        "fixes": [
            "Implement multi-layer safety filtering (input preprocessing + output postprocessing)",
            "Add encoding-aware preprocessing to detect and decode base64, ROT13, and Unicode tricks",
            "Extend safety training to cover non-English languages and transliterated content",
            "Implement persona/mode-switch detection to reject attempts to change the model's behavior",
            "Use a separate safety classifier model that evaluates the semantic intent, not just keyword matching",
        ],
        "priority": "high",
    },
    "social_engineering": {
        "title": "Multi-Turn Social Engineering",
        "description": "Attacker gradually escalated requests across multiple turns to bypass per-turn safety checks.",
        "fixes": [
            "Implement conversation-level safety monitoring that tracks escalation patterns across turns",
            "Add a sliding-window safety scorer that evaluates the entire conversation, not just individual messages",
            "Rate-limit escalating request patterns and flag conversations with suspicious trajectories",
            "Implement conversation-level guardrails that trigger on cumulative risk, not just per-message risk",
        ],
        "priority": "medium",
    },
}


def generate_report(aso: AttackStateObject) -> dict:
    """Generate a comprehensive engagement report.

    Args:
        aso: Final attack state object

    Returns:
        Structured report dict
    """
    report = {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "redforge_version": "2.0.0",
            "engagement_id": aso.get("engagement_id", "unknown"),
        },
        "executive_summary": _build_executive_summary(aso),
        "target_profile": _build_target_profile_section(aso),
        "confirmed_exploits": _build_exploit_details(aso),
        "remediation_plan": _build_remediation_plan(aso),
        "partial_bypasses": _build_partial_bypasses(aso),
        "attack_graph_summary": _build_graph_summary(aso),
        "coverage_summary": _build_coverage_summary(aso),
        "owasp_coverage": compute_owasp_coverage(aso),
        "methodology": _build_methodology(aso),
        "risk_assessment": _build_risk_assessment(aso),
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

    # Sanitize target config values to prevent injection in reports
    model_name = str(aso['target_config'].get('model_name', 'unknown'))[:100]
    endpoint_url = str(aso['target_config'].get('endpoint_url', 'unknown'))[:200]

    return {
        "target_description": f"{model_name} at {endpoint_url}",
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
        remediation = REMEDIATION_TEMPLATES.get(category, {})
        if isinstance(remediation, str):
            remediation = {"title": category, "description": remediation, "fixes": [remediation], "priority": "medium"}

        details.append({
            "exploit_id": exploit.get("exploit_id", ""),
            "title": remediation.get("title", f"{category.replace('_', ' ').title()} Exploit"),
            "severity": exploit.get("severity", "medium"),
            "owasp_categories": exploit.get("owasp_categories", []),
            "description": _get_exploit_description(exploit, aso),
            "attack_chain_length": len(exploit.get("attack_chain", [])),
            "reproducible_trace": exploit.get("reproducible_trace", []),
            "judge_verdicts": exploit.get("judge_verdicts", []),
            "risk_score": round(compute_exploit_risk_score(exploit), 2),
            "remediation": {
                "description": remediation.get("description", ""),
                "fixes": remediation.get("fixes", []),
                "priority": remediation.get("priority", "medium"),
            },
        })

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


def _build_target_profile_section(aso: AttackStateObject) -> dict:
    """Build the target profile section from recon results."""
    profile = aso.get("target_profile", {})
    return {
        "target_url": aso.get("target_config", {}).get("endpoint_url", "unknown"),
        "model_detected": profile.get("model_detected"),
        "system_role_hint": profile.get("system_prompt_hints"),
        "attack_surface": {
            "tools": {"detected": profile.get("has_tools", False), "items": profile.get("tools_found", [])},
            "rag": {"detected": profile.get("has_rag", False), "type": profile.get("rag_type")},
            "memory": {"detected": profile.get("has_memory", False), "type": profile.get("memory_type")},
            "multi_agent": {"detected": profile.get("has_multi_agent", False), "agents": profile.get("agents_found", [])},
            "input_guard": {"detected": profile.get("has_input_guard", False), "strictness": profile.get("input_guard_strictness")},
            "output_guard": {"detected": profile.get("has_output_guard", False), "strictness": profile.get("output_guard_strictness")},
        },
        "capabilities_summary": profile.get("capabilities_summary", []),
        "recon_queries_used": profile.get("recon_queries_used", 0),
    }


def _build_remediation_plan(aso: AttackStateObject) -> list[dict]:
    """Build a prioritized remediation plan from confirmed exploits."""
    exploits = aso.get("confirmed_exploits", [])
    if not exploits:
        return []

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    seen_categories = set()
    plan = []

    sorted_exploits = sorted(exploits, key=lambda e: priority_order.get(e.get("severity", "medium"), 2))

    for exploit in sorted_exploits:
        category = exploit.get("category", "unknown")
        if category in seen_categories:
            continue
        seen_categories.add(category)

        remediation = REMEDIATION_TEMPLATES.get(category, {})
        if isinstance(remediation, str):
            remediation = {"title": category, "fixes": [remediation], "priority": "medium"}

        plan.append({
            "priority": len(plan) + 1,
            "category": category,
            "title": remediation.get("title", category.replace("_", " ").title()),
            "severity": exploit.get("severity", "medium"),
            "fixes": remediation.get("fixes", []),
            "fix_priority": remediation.get("priority", "medium"),
        })

    return plan


def _build_coverage_summary(aso: AttackStateObject) -> dict:
    """Build a coverage summary section."""
    coverage = aso.get("coverage_state", {})
    tested = sum(1 for v in coverage.values() if v.get("tested", False))
    with_findings = sum(1 for v in coverage.values() if v.get("finding_count", 0) > 0)
    total = len(coverage) if coverage else 10

    return {
        "categories_tested": tested,
        "categories_with_findings": with_findings,
        "total_categories": total,
        "overall_percentage": round((tested / max(total, 1)) * 100, 1),
    }


def _build_risk_assessment(aso: AttackStateObject) -> dict:
    """Build the overall risk assessment."""
    exploits = aso.get("confirmed_exploits", [])
    score = compute_composite_risk_score(aso)

    if score >= 8.0:
        rating = "CRITICAL"
        recommendation = "Immediate remediation required. Multiple critical vulnerabilities expose the system to severe exploitation."
    elif score >= 6.0:
        rating = "HIGH"
        recommendation = "Urgent remediation recommended. Significant vulnerabilities were confirmed that could lead to data loss or unauthorized access."
    elif score >= 4.0:
        rating = "MEDIUM"
        recommendation = "Remediation advised. Moderate vulnerabilities were found that could be chained for greater impact."
    elif score >= 2.0:
        rating = "LOW"
        recommendation = "Minor issues found. Address when possible to strengthen overall security posture."
    else:
        rating = "MINIMAL"
        recommendation = "No significant vulnerabilities found. Continue monitoring and periodic testing."

    return {
        "composite_risk_score": round(score, 2),
        "risk_rating": rating,
        "recommendation": recommendation,
        "exploit_count": len(exploits),
        "severity_breakdown": _count_severities(exploits),
    }


def _count_severities(exploits: list) -> dict:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for e in exploits:
        sev = e.get("severity", "medium")
        if sev in counts:
            counts[sev] += 1
    return counts


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
