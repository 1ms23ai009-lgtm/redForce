"""Reconnaissance Profiler — orchestrates all probes and builds a TargetProfile.

This is the Phase 1 node in the REDFORGE v2 LangGraph.
"""

import logging
from typing import Optional, Callable

from redforge.core.state import AttackStateObject, TargetProfile
from redforge.safety.audit import log_event
from redforge.target.connector import TargetConnector
from redforge.recon.probes import PROBE_REGISTRY

logger = logging.getLogger(__name__)


class ReconProfiler:
    """Runs all reconnaissance probes against a target and builds a profile."""

    def __init__(self, connector: TargetConnector, broadcast_fn: Optional[Callable] = None):
        self._connector = connector
        self._broadcast = broadcast_fn

    def _send(self, message: str) -> dict:
        return self._connector.send_message(message)

    def run(self, aso: AttackStateObject) -> AttackStateObject:
        """Execute all recon probes and populate target_profile in the ASO.

        This is designed to be used as a LangGraph node function.
        """
        log_event(
            event_type="recon_start",
            agent_name="recon_profiler",
            action_taken="Starting Phase 1: Reconnaissance",
            reasoning_chain="Probing target to build capability profile before attacking",
            aso=aso,
        )

        self._emit("phase", {"phase": "recon", "message": "Phase 1: Reconnaissance — probing target..."})

        profile = aso["target_profile"]
        total_queries = 0
        probe_names = list(PROBE_REGISTRY.keys())
        total_probes = len(probe_names)

        for idx, probe_name in enumerate(probe_names):
            probe_info = PROBE_REGISTRY[probe_name]
            probe_fn = probe_info["fn"]
            cost = probe_info["queries_cost"]

            if aso["query_budget_remaining"] < cost:
                logger.warning(f"Skipping probe {probe_name}: insufficient budget ({aso['query_budget_remaining']} < {cost})")
                continue

            self._emit("recon_probe", {
                "probe": probe_name,
                "index": idx,
                "total": total_probes,
                "description": probe_info["description"],
            })

            try:
                result = probe_fn(self._send)
                self._merge_probe_result(profile, probe_name, result)
                aso["query_budget_remaining"] -= cost
                total_queries += cost

                self._emit("recon_result", {
                    "probe": probe_name,
                    "index": idx,
                    "result": _sanitize_for_broadcast(result),
                })

                log_event(
                    event_type="recon_probe",
                    agent_name="recon_profiler",
                    action_taken=f"Probe '{probe_name}' complete",
                    reasoning_chain=str(_sanitize_for_broadcast(result))[:300],
                    aso=aso,
                )

            except Exception as e:
                logger.error(f"Probe '{probe_name}' failed: {e}")

        profile["recon_complete"] = True
        profile["recon_queries_used"] = total_queries
        profile["capabilities_summary"] = self._build_capabilities_summary(profile)

        self._update_topology_from_profile(aso, profile)

        attack_plan = self._build_attack_plan(profile)

        self._emit("recon_complete", {
            "profile": _sanitize_profile(profile),
            "attack_plan": attack_plan,
        })

        log_event(
            event_type="recon_complete",
            agent_name="recon_profiler",
            action_taken=f"Recon complete: {len(profile['capabilities_summary'])} capabilities discovered",
            reasoning_chain=f"Plan: {', '.join(a['agent'] for a in attack_plan)}",
            aso=aso,
        )

        aso["target_profile"] = profile
        aso["_attack_plan"] = attack_plan
        return aso

    def _merge_probe_result(self, profile: TargetProfile, probe_name: str, result: dict):
        if probe_name == "tools":
            profile["has_tools"] = result.get("has_tools", False)
            profile["tools_found"] = result.get("tools_found", [])

        elif probe_name == "memory":
            profile["has_memory"] = result.get("has_memory", False)
            profile["memory_type"] = result.get("memory_type")

        elif probe_name == "identity":
            profile["system_prompt_hints"] = result.get("system_prompt_hints")
            profile["model_detected"] = result.get("model_detected")

        elif probe_name == "rag":
            profile["has_rag"] = result.get("has_rag", False)
            profile["rag_type"] = result.get("rag_type")

        elif probe_name == "multi_agent":
            profile["has_multi_agent"] = result.get("has_multi_agent", False)
            profile["agents_found"] = result.get("agents_found", [])

        elif probe_name == "input_guard":
            profile["has_input_guard"] = result.get("has_input_guard", False)
            profile["input_guard_strictness"] = result.get("input_guard_strictness")

        elif probe_name == "output_guard":
            profile["has_output_guard"] = result.get("has_output_guard", False)
            profile["output_guard_strictness"] = result.get("output_guard_strictness")

    def _update_topology_from_profile(self, aso: AttackStateObject, profile: TargetProfile):
        """Sync topology from the richer profile data."""
        topo = aso["target_topology"]

        for tool_name in profile.get("tools_found", []):
            entry = {"name": tool_name, "description": "Discovered via recon"}
            if entry not in topo["tools"]:
                topo["tools"].append(entry)

        if profile.get("has_memory") and profile.get("memory_type"):
            if profile["memory_type"] not in topo["memory_systems"]:
                topo["memory_systems"].append(profile["memory_type"])

        for agent_name in profile.get("agents_found", []):
            entry = {"name": agent_name, "role": "Discovered via recon"}
            if entry not in topo["agents"]:
                topo["agents"].append(entry)

        if profile.get("has_input_guard"):
            guard_desc = f"input_guard: {profile.get('input_guard_strictness', 'unknown')}"
            if guard_desc not in topo["guardrails"]:
                topo["guardrails"].append(guard_desc)

        if profile.get("has_output_guard"):
            guard_desc = f"output_guard: {profile.get('output_guard_strictness', 'unknown')}"
            if guard_desc not in topo["guardrails"]:
                topo["guardrails"].append(guard_desc)

    def _build_capabilities_summary(self, profile: TargetProfile) -> list[str]:
        caps = []
        if profile.get("has_tools"):
            caps.append(f"Tools: {', '.join(profile['tools_found'][:5])}")
        if profile.get("has_rag"):
            caps.append(f"RAG: {profile.get('rag_type', 'detected')}")
        if profile.get("has_memory"):
            caps.append(f"Memory: {profile.get('memory_type', 'detected')}")
        if profile.get("has_multi_agent"):
            caps.append(f"Multi-agent: {', '.join(profile['agents_found'][:3])}")
        if profile.get("has_input_guard"):
            caps.append(f"Input guard: {profile.get('input_guard_strictness', 'detected')}")
        if profile.get("has_output_guard"):
            caps.append(f"Output guard: {profile.get('output_guard_strictness', 'detected')}")
        if profile.get("model_detected"):
            caps.append(f"Model: {profile['model_detected']}")
        if profile.get("system_prompt_hints"):
            caps.append(f"Role: {profile['system_prompt_hints'][:80]}")
        return caps

    def _build_attack_plan(self, profile: TargetProfile) -> list[dict]:
        """Determine which agents to deploy based on recon results."""
        plan = []

        plan.append({
            "agent": "jailbreak_specialist",
            "reason": "Always run: core PAIR jailbreak",
            "priority": 1,
        })
        plan.append({
            "agent": "exfiltration_specialist",
            "reason": "Always run: system prompt extraction",
            "priority": 2,
        })
        plan.append({
            "agent": "social_engineering_specialist",
            "reason": "Always run: multi-turn escalation",
            "priority": 3,
        })

        if profile.get("has_tools"):
            plan.append({
                "agent": "tool_abuse_specialist",
                "reason": f"Tools detected: {', '.join(profile['tools_found'][:5])}",
                "priority": 1,
            })

        if profile.get("has_rag"):
            plan.append({
                "agent": "rag_poisoning_specialist",
                "reason": f"RAG detected: {profile.get('rag_type', 'unknown')}",
                "priority": 1,
            })

        if profile.get("has_memory"):
            plan.append({
                "agent": "memory_poison_specialist",
                "reason": f"Memory detected: {profile.get('memory_type', 'unknown')}",
                "priority": 1,
            })

        if profile.get("has_input_guard") or profile.get("has_output_guard"):
            plan.append({
                "agent": "guardrail_bypass_specialist",
                "reason": f"Guards detected — input: {profile.get('input_guard_strictness', 'none')}, output: {profile.get('output_guard_strictness', 'none')}",
                "priority": 2,
            })

        if profile.get("has_multi_agent"):
            plan.append({
                "agent": "cross_agent_specialist",
                "reason": f"Multi-agent detected: {', '.join(profile['agents_found'][:3])}",
                "priority": 2,
            })

        plan.append({
            "agent": "prompt_injection_specialist",
            "reason": "Always run: direct/indirect prompt injection",
            "priority": 3,
        })

        plan.sort(key=lambda x: x["priority"])
        return plan

    def _emit(self, event_type: str, data: dict):
        if self._broadcast:
            try:
                self._broadcast({"type": event_type, "data": data})
            except Exception:
                pass


def _sanitize_for_broadcast(result: dict) -> dict:
    """Remove raw_responses to keep broadcast payloads small."""
    return {k: v for k, v in result.items() if k != "raw_responses"}


def _sanitize_profile(profile) -> dict:
    """Produce a JSON-safe summary of the profile."""
    return {
        "has_tools": profile.get("has_tools", False),
        "tools_found": profile.get("tools_found", []),
        "has_rag": profile.get("has_rag", False),
        "rag_type": profile.get("rag_type"),
        "has_memory": profile.get("has_memory", False),
        "memory_type": profile.get("memory_type"),
        "has_multi_agent": profile.get("has_multi_agent", False),
        "agents_found": profile.get("agents_found", []),
        "has_input_guard": profile.get("has_input_guard", False),
        "input_guard_strictness": profile.get("input_guard_strictness"),
        "has_output_guard": profile.get("has_output_guard", False),
        "output_guard_strictness": profile.get("output_guard_strictness"),
        "system_prompt_hints": profile.get("system_prompt_hints"),
        "model_detected": profile.get("model_detected"),
        "capabilities_summary": profile.get("capabilities_summary", []),
    }
