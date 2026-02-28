"""Attack State Object (ASO) - Central shared state for the REDFORGE system.

The ASO passes through the entire LangGraph graph. Every agent reads from
and writes to this object. Implemented as a TypedDict for LangGraph compatibility.
"""

import uuid
from typing import TypedDict, Optional
from datetime import datetime, timezone


class AttackHistoryEntry(TypedDict):
    turn_id: int
    agent_name: str
    strategy_category: str
    prompt_sent: str
    response_received: str
    judge_verdict: dict
    timestamp: str
    tokens_used: int
    cost_usd: float


class PartialBypassEntry(TypedDict):
    turn_id: int
    description: str
    guardrail_affected: str
    bypass_degree: float  # 0.0-1.0
    exploitable_by: list  # list of strategy categories


class HypothesisEntry(TypedDict):
    hypothesis_id: str
    description: str
    confidence: float  # 0.0-1.0
    supporting_evidence: list  # list of turn_ids
    recommended_next_agent: str
    priority_score: float


class CoverageCategoryState(TypedDict):
    tested: bool
    finding_count: int
    highest_severity: str


class ConfirmedExploitEntry(TypedDict):
    exploit_id: str
    category: str
    severity: str
    attack_chain: list  # list of turn_ids
    judge_verdicts: list
    reproducible_trace: list  # list of prompts
    risk_score: float
    owasp_categories: list


class AuditLogEntry(TypedDict):
    event_id: str
    event_type: str
    timestamp: str
    agent_name: str
    action_taken: str
    reasoning_chain: str
    aso_snapshot_hash: str
    previous_entry_hash: str
    entry_hash: str


class StrategyLibraryHit(TypedDict):
    strategy_id: str
    similarity_score: float
    was_successful: Optional[bool]


class TargetProfile(TypedDict):
    has_tools: bool
    tools_found: list
    has_rag: bool
    rag_type: Optional[str]
    has_memory: bool
    memory_type: Optional[str]
    has_multi_agent: bool
    agents_found: list
    has_input_guard: bool
    input_guard_strictness: Optional[str]
    has_output_guard: bool
    output_guard_strictness: Optional[str]
    system_prompt_hints: Optional[str]
    model_detected: Optional[str]
    capabilities_summary: list
    recon_complete: bool
    recon_queries_used: int


class AttackStateObject(TypedDict):
    engagement_id: str
    target_config: dict
    target_topology: dict
    target_profile: TargetProfile
    attack_history: list  # list of AttackHistoryEntry
    partial_bypass_registry: list  # list of PartialBypassEntry
    active_hypotheses: list  # list of HypothesisEntry
    coverage_state: dict  # maps OWASP category str -> CoverageCategoryState
    attack_graph: dict  # serialized NetworkX DiGraph adjacency data
    nash_equilibrium: Optional[dict]
    strategic_digest: Optional[str]
    strategy_library_hits: list  # list of StrategyLibraryHit
    confirmed_exploits: list  # list of ConfirmedExploitEntry
    query_budget_remaining: int
    kill_switch_triggered: bool
    audit_log: list  # list of AuditLogEntry


OWASP_LLM_CATEGORIES = [
    "LLM01_prompt_injection",
    "LLM02_insecure_output",
    "LLM03_training_data_poisoning",
    "LLM04_model_dos",
    "LLM05_supply_chain",
    "LLM06_sensitive_info",
    "LLM07_insecure_plugin",
    "LLM08_excessive_agency",
    "LLM09_overreliance",
    "LLM10_model_theft",
]


def create_initial_coverage_state() -> dict:
    """Create initial coverage state with all categories untested."""
    return {
        cat: {"tested": False, "finding_count": 0, "highest_severity": "none"}
        for cat in OWASP_LLM_CATEGORIES
    }


def create_empty_target_profile() -> TargetProfile:
    """Create an empty target profile for recon phase."""
    return TargetProfile(
        has_tools=False,
        tools_found=[],
        has_rag=False,
        rag_type=None,
        has_memory=False,
        memory_type=None,
        has_multi_agent=False,
        agents_found=[],
        has_input_guard=False,
        input_guard_strictness=None,
        has_output_guard=False,
        output_guard_strictness=None,
        system_prompt_hints=None,
        model_detected=None,
        capabilities_summary=[],
        recon_complete=False,
        recon_queries_used=0,
    )


def create_initial_aso(
    target_config: dict,
    query_budget: int = 100,
) -> AttackStateObject:
    """Create a fresh Attack State Object for a new engagement."""
    return AttackStateObject(
        engagement_id=str(uuid.uuid4()),
        target_config=target_config,
        target_topology={
            "agents": [],
            "tools": [],
            "memory_systems": [],
            "policies": [],
            "guardrails": [],
        },
        target_profile=create_empty_target_profile(),
        attack_history=[],
        partial_bypass_registry=[],
        active_hypotheses=[],
        coverage_state=create_initial_coverage_state(),
        attack_graph={"nodes": [], "adjacency": []},
        nash_equilibrium=None,
        strategic_digest=None,
        strategy_library_hits=[],
        confirmed_exploits=[],
        query_budget_remaining=query_budget,
        kill_switch_triggered=False,
        audit_log=[],
    )


def get_current_turn(aso: AttackStateObject) -> int:
    """Get the current turn number based on attack history length."""
    return len(aso["attack_history"])


def create_attack_history_entry(
    turn_id: int,
    agent_name: str,
    strategy_category: str,
    prompt_sent: str,
    response_received: str,
    judge_verdict: dict,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
) -> AttackHistoryEntry:
    """Create a new attack history entry."""
    return AttackHistoryEntry(
        turn_id=turn_id,
        agent_name=agent_name,
        strategy_category=strategy_category,
        prompt_sent=prompt_sent,
        response_received=response_received,
        judge_verdict=judge_verdict,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


def create_confirmed_exploit(
    category: str,
    severity: str,
    attack_chain: list,
    judge_verdicts: list,
    reproducible_trace: list,
    risk_score: float,
    owasp_categories: list,
) -> ConfirmedExploitEntry:
    """Create a confirmed exploit entry."""
    return ConfirmedExploitEntry(
        exploit_id=str(uuid.uuid4()),
        category=category,
        severity=severity,
        attack_chain=attack_chain,
        judge_verdicts=judge_verdicts,
        reproducible_trace=reproducible_trace,
        risk_score=risk_score,
        owasp_categories=owasp_categories,
    )
