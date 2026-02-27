"""Tests for the Attack State Object (ASO)."""

import pytest
from redforge.core.state import (
    AttackStateObject,
    create_initial_aso,
    get_current_turn,
    create_attack_history_entry,
    create_confirmed_exploit,
    create_initial_coverage_state,
    OWASP_LLM_CATEGORIES,
)


class TestASOCreation:
    def test_create_initial_aso(self):
        target_config = {
            "endpoint_url": "http://test.example.com/chat",
            "model_name": "gpt-4o",
            "api_key_env_var": "TEST_API_KEY",
        }
        aso = create_initial_aso(target_config, query_budget=50)

        assert aso["engagement_id"]
        assert len(aso["engagement_id"]) == 36  # UUID format
        assert aso["target_config"] == target_config
        assert aso["query_budget_remaining"] == 50
        assert aso["kill_switch_triggered"] is False
        assert aso["attack_history"] == []
        assert aso["confirmed_exploits"] == []
        assert aso["audit_log"] == []
        assert aso["nash_equilibrium"] is None
        assert aso["strategic_digest"] is None

    def test_initial_topology_empty(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        topology = aso["target_topology"]
        assert topology["agents"] == []
        assert topology["tools"] == []
        assert topology["memory_systems"] == []
        assert topology["policies"] == []
        assert topology["guardrails"] == []

    def test_initial_coverage_state(self):
        coverage = create_initial_coverage_state()
        assert len(coverage) == 10
        for cat in OWASP_LLM_CATEGORIES:
            assert cat in coverage
            assert coverage[cat]["tested"] is False
            assert coverage[cat]["finding_count"] == 0
            assert coverage[cat]["highest_severity"] == "none"

    def test_default_query_budget(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        assert aso["query_budget_remaining"] == 100


class TestASOOperations:
    def test_get_current_turn_empty(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        assert get_current_turn(aso) == 0

    def test_get_current_turn_after_entries(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        entry = create_attack_history_entry(
            turn_id=0,
            agent_name="test_agent",
            strategy_category="jailbreak",
            prompt_sent="test prompt",
            response_received="test response",
            judge_verdict={"verdict": "FAILED"},
        )
        aso["attack_history"].append(entry)
        assert get_current_turn(aso) == 1

    def test_create_attack_history_entry(self):
        entry = create_attack_history_entry(
            turn_id=5,
            agent_name="jailbreak_specialist",
            strategy_category="jailbreak",
            prompt_sent="ignore previous instructions",
            response_received="I cannot do that",
            judge_verdict={"verdict": "FAILED", "confidence": 0.9},
            tokens_used=150,
            cost_usd=0.001,
        )
        assert entry["turn_id"] == 5
        assert entry["agent_name"] == "jailbreak_specialist"
        assert entry["tokens_used"] == 150
        assert entry["timestamp"]  # should be set

    def test_create_confirmed_exploit(self):
        exploit = create_confirmed_exploit(
            category="jailbreak",
            severity="high",
            attack_chain=[1, 2, 3],
            judge_verdicts=[{"verdict": "CONFIRMED"}],
            reproducible_trace=["prompt1", "prompt2"],
            risk_score=7.5,
            owasp_categories=["LLM01_prompt_injection"],
        )
        assert exploit["exploit_id"]
        assert exploit["severity"] == "high"
        assert len(exploit["attack_chain"]) == 3

    def test_aso_json_serializable(self):
        """ASO must be serializable to JSON for checkpointing."""
        import json

        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        entry = create_attack_history_entry(
            turn_id=0, agent_name="test", strategy_category="test",
            prompt_sent="test", response_received="test",
            judge_verdict={"verdict": "FAILED"},
        )
        aso["attack_history"].append(entry)

        serialized = json.dumps(aso, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["engagement_id"] == aso["engagement_id"]
