"""Orchestrator agent - LangGraph StateGraph coordinating all agents.

Builds the main LangGraph StateGraph with all nodes and conditional edges
as specified in the REDFORGE architecture.
"""

import logging
import os
from typing import Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from redforge.config import (
    RedForgeConfig,
    STRATEGY_TO_AGENT,
    VERDICT_CONFIRMED,
    VERDICT_PARTIAL,
)
from redforge.core.state import (
    AttackStateObject,
    get_current_turn,
    create_confirmed_exploit,
)
from redforge.core.mdp import (
    check_engagement_done,
    update_coverage_state,
    compute_coverage_percentage,
)
from redforge.safety.audit import log_event
from redforge.safety.kill_switch import check_kill_switch
from redforge.target.connector import TargetConnector
from redforge.target.topology import TopologyDiscovery
from redforge.rl.high_level_policy import HighLevelPolicy
from redforge.rl.low_level_policy import LowLevelPolicy
from redforge.rl.reward import compute_reward, _strategy_to_owasp
from redforge.graph.attack_graph import (
    deserialize_graph,
    serialize_graph,
)
from redforge.graph.graph_extractor import extract_graph_from_logs
from redforge.graph.effort_scorer import update_graph_effort_scores
from redforge.graph.nash import compute_nash_equilibrium
from redforge.digest.generator import generate_digest
from redforge.judge.ensemble import JudgeEnsemble
from redforge.strategy_library.library import StrategyLibrary
from redforge.strategy_library.retriever import add_strategy
from redforge.reporting.report_generator import generate_report

from redforge.agents.jailbreak import jailbreak_specialist
from redforge.agents.prompt_injection import prompt_injection_specialist
from redforge.agents.cross_agent import cross_agent_specialist
from redforge.agents.exfiltration import exfiltration_specialist
from redforge.agents.tool_abuse import tool_abuse_specialist
from redforge.agents.memory_poison import memory_poison_specialist
from redforge.agents.social_engineering import social_engineering_specialist
from redforge.agents.rag_poisoning import rag_poisoning_specialist
from redforge.agents.guardrail_bypass import guardrail_bypass_specialist
from redforge.recon.profiler import ReconProfiler

logger = logging.getLogger(__name__)


def build_redforge_graph(
    config: RedForgeConfig,
    connector: TargetConnector,
    high_level_policy: HighLevelPolicy,
    low_level_policy: LowLevelPolicy,
    judge: JudgeEnsemble,
    strategy_library: StrategyLibrary,
) -> tuple[StateGraph, MemorySaver]:
    """Build the main LangGraph StateGraph for REDFORGE.

    Returns:
        (compiled_graph, checkpointer)
    """
    checkpointer = MemorySaver()
    topology_discovery = TopologyDiscovery(
        connector=connector,
        parser_model=config.graph_extractor_model,
    )
    recon_profiler = ReconProfiler(connector=connector)

    # --- Node functions ---

    def recon_node(state: AttackStateObject) -> AttackStateObject:
        """Phase 1: Reconnaissance — probe target to build a profile."""
        return recon_profiler.run(state)

    def topology_discovery_node(state: AttackStateObject) -> AttackStateObject:
        return topology_discovery.discover(state)

    def orchestrator_node(state: AttackStateObject) -> AttackStateObject:
        """High-level orchestrator: uses attack plan from recon, then RL policy."""
        attack_plan = state.get("_attack_plan", [])
        executed_agents = set()
        for entry in state.get("attack_history", []):
            executed_agents.add(entry.get("agent_name", ""))

        # Phase 2: if we have a recon-based attack plan, follow it first
        if attack_plan:
            for planned in attack_plan:
                agent_name = planned["agent"]
                if agent_name not in executed_agents:
                    strategy = _agent_to_strategy(agent_name)
                    state["_next_strategy"] = strategy
                    state["_last_log_prob"] = 0.0
                    state["_last_value"] = 0.0

                    log_event(
                        event_type="agent_action",
                        agent_name="orchestrator",
                        action_taken=f"Attack plan: {strategy} ({planned['reason']})",
                        reasoning_chain=f"Recon-driven selection: {agent_name}",
                        aso=state,
                    )
                    return state

        # All planned agents executed — fall back to RL policy
        strategy, log_prob, value = high_level_policy.select_action(state)
        state["_next_strategy"] = strategy
        state["_last_log_prob"] = log_prob
        state["_last_value"] = value

        log_event(
            event_type="agent_action",
            agent_name="orchestrator",
            action_taken=f"RL policy selected: {strategy}",
            reasoning_chain=f"Policy output: strategy={strategy}, log_prob={log_prob:.3f}",
            aso=state,
        )
        return state

    def jailbreak_node(state: AttackStateObject) -> AttackStateObject:
        return jailbreak_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def prompt_injection_node(state: AttackStateObject) -> AttackStateObject:
        strategy = state.get("_next_strategy", "prompt_injection_direct")
        mode = "indirect" if "indirect" in strategy else "direct"
        return prompt_injection_specialist(
            state, connector, low_level_policy, strategy_library,
            mode=mode,
            max_pair_iterations=config.max_pair_iterations,
            tap_branches=config.tap_branches,
        )

    def cross_agent_node(state: AttackStateObject) -> AttackStateObject:
        return cross_agent_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def exfiltration_node(state: AttackStateObject) -> AttackStateObject:
        return exfiltration_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def tool_abuse_node(state: AttackStateObject) -> AttackStateObject:
        return tool_abuse_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def memory_poison_node(state: AttackStateObject) -> AttackStateObject:
        return memory_poison_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def social_engineering_node(state: AttackStateObject) -> AttackStateObject:
        return social_engineering_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def rag_poisoning_node(state: AttackStateObject) -> AttackStateObject:
        return rag_poisoning_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def guardrail_bypass_node(state: AttackStateObject) -> AttackStateObject:
        return guardrail_bypass_specialist(
            state, connector, low_level_policy, strategy_library,
            config.max_pair_iterations, config.tap_branches,
        )

    def judge_evaluation_node(state: AttackStateObject) -> AttackStateObject:
        """Run three-layer judge on the latest attack response."""
        history = state.get("attack_history", [])
        if not history:
            return state

        # Find the latest entries with pending verdicts
        for entry in reversed(history):
            verdict = entry.get("judge_verdict", {})
            if isinstance(verdict, dict) and verdict.get("verdict") == "pending":
                result = judge.evaluate_exploit(
                    prompt_sent=entry["prompt_sent"],
                    response_received=entry["response_received"],
                    attack_context={
                        "category": entry.get("strategy_category", ""),
                        "agent": entry.get("agent_name", ""),
                    },
                    engagement_id=state.get("engagement_id", ""),
                    turn_id=entry.get("turn_id", 0),
                )
                entry["judge_verdict"] = result

                log_event(
                    event_type="judge_verdict",
                    agent_name="judge_ensemble",
                    action_taken=f"Verdict: {result.get('verdict', 'unknown')}",
                    reasoning_chain=result.get("explanation", ""),
                    aso=state,
                )

                # Update coverage state
                category = entry.get("strategy_category", "")
                owasp_cats = _strategy_to_owasp(category)
                for owasp_cat in owasp_cats:
                    update_coverage_state(
                        state, owasp_cat,
                        severity=result.get("severity") if result.get("verdict") == VERDICT_CONFIRMED else None,
                    )

                # Record partial bypass
                if result.get("verdict") == VERDICT_PARTIAL:
                    state["partial_bypass_registry"].append({
                        "turn_id": entry["turn_id"],
                        "description": result.get("explanation", "Partial bypass detected"),
                        "guardrail_affected": category,
                        "bypass_degree": result.get("confidence", 0.5),
                        "exploitable_by": [category],
                    })

        state["_has_confirmed_exploit"] = any(
            isinstance(e.get("judge_verdict"), dict) and e["judge_verdict"].get("verdict") == VERDICT_CONFIRMED
            for e in history[-10:]
        )
        return state

    def exploit_recorder_node(state: AttackStateObject) -> AttackStateObject:
        """Record confirmed exploits."""
        for entry in state.get("attack_history", []):
            verdict = entry.get("judge_verdict", {})
            if not isinstance(verdict, dict):
                continue
            if verdict.get("verdict") != VERDICT_CONFIRMED:
                continue

            # Check if already recorded
            exploit_turn_ids = set()
            for ex in state["confirmed_exploits"]:
                exploit_turn_ids.update(ex.get("attack_chain", []))

            if entry["turn_id"] in exploit_turn_ids:
                continue

            category = entry.get("strategy_category", "unknown")
            exploit = create_confirmed_exploit(
                category=category,
                severity=verdict.get("severity", "medium"),
                attack_chain=[entry["turn_id"]],
                judge_verdicts=[verdict],
                reproducible_trace=[entry["prompt_sent"]],
                risk_score=_severity_to_score(verdict.get("severity", "medium")),
                owasp_categories=_strategy_to_owasp(category),
            )
            state["confirmed_exploits"].append(exploit)

            log_event(
                event_type="exploit_confirmed",
                agent_name="exploit_recorder",
                action_taken=f"Exploit confirmed: {category} ({verdict.get('severity', 'medium')})",
                reasoning_chain=verdict.get("explanation", ""),
                aso=state,
            )

            # Add to strategy library
            try:
                add_strategy(
                    library=strategy_library,
                    exploit_result={
                        "category": category,
                        "technique_name": f"{category}_exploit",
                        "description": verdict.get("explanation", ""),
                        "reproducible_trace": [entry["prompt_sent"]],
                        "guardrail_trigger": "",
                        "success_indicators": [],
                        "failure_indicators": [],
                    },
                    aso=state,
                    novelty_threshold=config.novelty_threshold,
                )
            except Exception as e:
                logger.warning(f"Failed to add strategy to library: {e}")

        return state

    # Provider settings for LLM calls within graph nodes
    is_vertex = config.provider == "vertex"
    _provider = config.provider if is_vertex else "openai"
    _vertex_project = config.vertex_project if is_vertex else None
    _vertex_location = config.vertex_location if is_vertex else "us-central1"

    def graph_update_node(state: AttackStateObject) -> AttackStateObject:
        """Update attack graph from latest actions."""
        try:
            existing_graph = deserialize_graph(state["attack_graph"])
            graph_model = config.vertex_graph_model if is_vertex else config.graph_extractor_model
            updated_graph = extract_graph_from_logs(
                attack_history=state["attack_history"],
                model_name=graph_model,
                existing_graph=existing_graph,
                provider=_provider,
                vertex_project=_vertex_project,
                vertex_location=_vertex_location,
            )
            update_graph_effort_scores(updated_graph, state)
            state["attack_graph"] = serialize_graph(updated_graph)
        except Exception as e:
            logger.warning(f"Graph update failed: {e}")

        return state

    def nash_compute_node(state: AttackStateObject) -> AttackStateObject:
        """Recompute Nash equilibrium."""
        try:
            graph = deserialize_graph(state["attack_graph"])
            result = compute_nash_equilibrium(graph)
            if result:
                result["computed_at_turn"] = get_current_turn(state)
                state["nash_equilibrium"] = result
        except Exception as e:
            logger.warning(f"Nash computation failed: {e}")
        return state

    def digest_update_node(state: AttackStateObject) -> AttackStateObject:
        """Generate new strategic digest."""
        try:
            graph = deserialize_graph(state["attack_graph"])
            nash = state.get("nash_equilibrium")
            if nash:
                digest_model = config.vertex_digest_model if is_vertex else config.digest_model
                digest = generate_digest(
                    attack_graph=graph,
                    nash_result=nash,
                    mode=config.digest_mode,
                    aso=state,
                    model_name=digest_model,
                    provider=_provider,
                    vertex_project=_vertex_project,
                    vertex_location=_vertex_location,
                )
                state["strategic_digest"] = digest
        except Exception as e:
            logger.warning(f"Digest generation failed: {e}")
        return state

    def kill_switch_check_node(state: AttackStateObject) -> AttackStateObject:
        """Check kill switch conditions."""
        triggered, reason = check_kill_switch(
            state,
            config={"max_hours": config.default_max_hours},
        )
        # State is already updated by check_kill_switch if triggered
        return state

    def report_generator_node(state: AttackStateObject) -> AttackStateObject:
        """Generate final report."""
        try:
            report = generate_report(state)
            state["_final_report"] = report
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state["_final_report"] = {"error": str(e)}

        log_event(
            event_type="engagement_end",
            agent_name="report_generator",
            action_taken="Final report generated",
            reasoning_chain=f"Exploits: {len(state['confirmed_exploits'])}, Coverage: {compute_coverage_percentage(state):.1f}%",
            aso=state,
        )
        return state

    def engagement_end_node(state: AttackStateObject) -> AttackStateObject:
        """Terminal node."""
        return state

    # --- Build the graph ---
    graph = StateGraph(dict)  # Using dict since TypedDict works as dict

    # Add nodes
    graph.add_node("recon", recon_node)
    graph.add_node("topology_discovery", topology_discovery_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("jailbreak_specialist", jailbreak_node)
    graph.add_node("prompt_injection_specialist", prompt_injection_node)
    graph.add_node("cross_agent_specialist", cross_agent_node)
    graph.add_node("exfiltration_specialist", exfiltration_node)
    graph.add_node("tool_abuse_specialist", tool_abuse_node)
    graph.add_node("memory_poison_specialist", memory_poison_node)
    graph.add_node("social_engineering_specialist", social_engineering_node)
    graph.add_node("rag_poisoning_specialist", rag_poisoning_node)
    graph.add_node("guardrail_bypass_specialist", guardrail_bypass_node)
    graph.add_node("judge_evaluation", judge_evaluation_node)
    graph.add_node("exploit_recorder", exploit_recorder_node)
    graph.add_node("graph_update", graph_update_node)
    graph.add_node("nash_compute", nash_compute_node)
    graph.add_node("digest_update", digest_update_node)
    graph.add_node("kill_switch_check", kill_switch_check_node)
    graph.add_node("report_generator", report_generator_node)
    graph.add_node("engagement_end", engagement_end_node)

    # --- Edges ---

    # START -> recon -> topology_discovery -> orchestrator
    graph.set_entry_point("recon")
    graph.add_edge("recon", "topology_discovery")
    graph.add_edge("topology_discovery", "orchestrator")

    # Orchestrator -> specialist (conditional)
    def route_orchestrator(state: dict) -> str:
        strategy = state.get("_next_strategy", "jailbreak")
        if strategy == "end_engagement":
            return "report_generator"
        agent = STRATEGY_TO_AGENT.get(strategy, "jailbreak_specialist")
        return agent

    graph.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "jailbreak_specialist": "jailbreak_specialist",
            "prompt_injection_specialist": "prompt_injection_specialist",
            "cross_agent_specialist": "cross_agent_specialist",
            "exfiltration_specialist": "exfiltration_specialist",
            "tool_abuse_specialist": "tool_abuse_specialist",
            "memory_poison_specialist": "memory_poison_specialist",
            "social_engineering_specialist": "social_engineering_specialist",
            "rag_poisoning_specialist": "rag_poisoning_specialist",
            "guardrail_bypass_specialist": "guardrail_bypass_specialist",
            "report_generator": "report_generator",
        },
    )

    # All specialists -> judge_evaluation
    for specialist in [
        "jailbreak_specialist", "prompt_injection_specialist",
        "cross_agent_specialist", "exfiltration_specialist",
        "tool_abuse_specialist", "memory_poison_specialist",
        "social_engineering_specialist", "rag_poisoning_specialist",
        "guardrail_bypass_specialist",
    ]:
        graph.add_edge(specialist, "judge_evaluation")

    # judge_evaluation -> conditional
    def route_after_judge(state: dict) -> str:
        if state.get("_has_confirmed_exploit"):
            return "exploit_recorder"
        return "graph_update"

    graph.add_conditional_edges(
        "judge_evaluation",
        route_after_judge,
        {
            "exploit_recorder": "exploit_recorder",
            "graph_update": "graph_update",
        },
    )

    # exploit_recorder -> graph_update
    graph.add_edge("exploit_recorder", "graph_update")

    # graph_update -> kill_switch_check
    graph.add_edge("graph_update", "kill_switch_check")

    # kill_switch_check -> conditional
    def route_after_kill_switch(state: dict) -> str:
        if state.get("kill_switch_triggered"):
            return "engagement_end"
        if state.get("query_budget_remaining", 0) <= 0:
            return "report_generator"
        turn = get_current_turn(state)
        if turn > 0 and turn % config.nash_update_frequency == 0:
            return "nash_compute"
        return "orchestrator"

    graph.add_conditional_edges(
        "kill_switch_check",
        route_after_kill_switch,
        {
            "engagement_end": "engagement_end",
            "report_generator": "report_generator",
            "nash_compute": "nash_compute",
            "orchestrator": "orchestrator",
        },
    )

    # nash_compute -> digest_update -> orchestrator
    graph.add_edge("nash_compute", "digest_update")
    graph.add_edge("digest_update", "orchestrator")

    # report_generator -> engagement_end
    graph.add_edge("report_generator", "engagement_end")

    # engagement_end -> END
    graph.add_edge("engagement_end", END)

    # Compile
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, checkpointer


def _severity_to_score(severity: str) -> float:
    """Convert severity to risk score."""
    return {
        "low": 2.5,
        "medium": 5.0,
        "high": 7.5,
        "critical": 10.0,
    }.get(severity, 5.0)


def _agent_to_strategy(agent_name: str) -> str:
    """Reverse-map agent name to strategy category."""
    agent_to_strat = {v: k for k, v in STRATEGY_TO_AGENT.items()}
    return agent_to_strat.get(agent_name, "jailbreak")
