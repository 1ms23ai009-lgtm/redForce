"""Tests for engagement lifecycle and ASO management."""

import pytest
import json

from redforge.core.state import (
    create_initial_aso,
    get_current_turn,
    create_attack_history_entry,
    create_confirmed_exploit,
    OWASP_LLM_CATEGORIES,
)
from redforge.core.mdp import (
    compute_coverage_percentage,
    update_coverage_state,
    check_engagement_done,
)
from redforge.safety.audit import log_event, verify_audit_chain
from redforge.safety.kill_switch import (
    check_kill_switch,
    check_content_safety,
    sanitize_content,
)
from redforge.config import RedForgeConfig
from redforge.rl.reward import compute_reward
from redforge.rl.high_level_policy import HighLevelPolicy, STATE_DIM


class TestEngagementLifecycle:
    def test_coverage_percentage_zero(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        assert compute_coverage_percentage(aso) == 0.0

    def test_coverage_percentage_partial(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        update_coverage_state(aso, "LLM01_prompt_injection", severity="high")
        pct = compute_coverage_percentage(aso)
        assert pct == 10.0

    def test_coverage_percentage_full(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        for cat in OWASP_LLM_CATEGORIES:
            update_coverage_state(aso, cat)
        assert compute_coverage_percentage(aso) == 100.0

    def test_check_done_budget(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"}, query_budget=0)
        assert check_engagement_done(aso) is True

    def test_check_done_kill_switch(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        aso["kill_switch_triggered"] = True
        assert check_engagement_done(aso) is True

    def test_check_not_done(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        assert check_engagement_done(aso) is False


class TestAuditTrail:
    def test_log_event(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        entry_hash = log_event(
            event_type="engagement_start",
            agent_name="test",
            action_taken="started",
            reasoning_chain="testing",
            aso=aso,
        )
        assert entry_hash
        assert len(aso["audit_log"]) == 1

    def test_audit_chain_integrity(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        for i in range(5):
            log_event(
                event_type="agent_action",
                agent_name=f"agent_{i}",
                action_taken=f"action_{i}",
                reasoning_chain=f"reason_{i}",
                aso=aso,
            )

        is_valid, errors = verify_audit_chain(aso["audit_log"])
        assert is_valid, f"Audit chain invalid: {errors}"

    def test_audit_chain_tamper_detection(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        for i in range(3):
            log_event("agent_action", "test", f"action_{i}", "reason", aso=aso)

        # Tamper with an entry
        aso["audit_log"][1]["action_taken"] = "tampered!"

        is_valid, errors = verify_audit_chain(aso["audit_log"])
        assert not is_valid
        assert len(errors) > 0


class TestKillSwitch:
    def test_safe_content(self):
        is_safe, reason = check_content_safety("Hello, how are you?")
        assert is_safe

    def test_budget_exhausted(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"}, query_budget=0)
        triggered, reason = check_kill_switch(aso)
        assert triggered
        assert "budget" in reason.lower()

    def test_normal_operation(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        triggered, reason = check_kill_switch(aso)
        assert not triggered

    def test_sanitize_content(self):
        safe = sanitize_content("This is normal text about testing")
        assert "REDACTED" not in safe


class TestRewardComputation:
    def test_confirmed_reward(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        reward = compute_reward(
            judge_verdict={"verdict": "CONFIRMED", "confidence": 0.9},
            aso=aso,
        )
        assert reward >= 1.0

    def test_failed_reward(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        reward = compute_reward(
            judge_verdict={"verdict": "FAILED"},
            aso=aso,
        )
        assert reward < 0

    def test_novelty_bonus(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        # Novel exploit (low similarity to library)
        reward_novel = compute_reward(
            judge_verdict={"verdict": "CONFIRMED"},
            aso=aso,
            strategy_library_similarity=0.3,
        )
        reward_known = compute_reward(
            judge_verdict={"verdict": "CONFIRMED"},
            aso=aso,
            strategy_library_similarity=0.9,
        )
        assert reward_novel > reward_known

    def test_cross_agent_bonus(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        reward_cross = compute_reward(
            judge_verdict={"verdict": "CONFIRMED"},
            aso=aso,
            is_cross_agent=True,
        )
        reward_normal = compute_reward(
            judge_verdict={"verdict": "CONFIRMED"},
            aso=aso,
            is_cross_agent=False,
        )
        assert reward_cross > reward_normal


class TestHighLevelPolicy:
    def test_encode_state_dimensions(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        policy = HighLevelPolicy()
        state = policy.encode_state(aso)
        assert len(state) == STATE_DIM
        assert state.dtype.name == "float32"

    def test_select_action(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        policy = HighLevelPolicy()
        strategy, log_prob, value = policy.select_action(aso)
        assert strategy in [
            "jailbreak", "prompt_injection_direct", "prompt_injection_indirect",
            "cross_agent_injection", "data_exfiltration", "tool_abuse",
            "memory_poisoning", "social_engineering",
        ]
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_strategy_already_tried_false(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        policy = HighLevelPolicy()
        assert policy.strategy_already_tried(aso, "jailbreak") is False

    def test_strategy_already_tried_true(self):
        aso = create_initial_aso({"endpoint_url": "http://test.com"})
        # Add 4 failed attempts
        for i in range(4):
            entry = create_attack_history_entry(
                turn_id=i, agent_name="test",
                strategy_category="jailbreak",
                prompt_sent="test", response_received="refused",
                judge_verdict={"verdict": "FAILED"},
            )
            aso["attack_history"].append(entry)

        policy = HighLevelPolicy()
        assert policy.strategy_already_tried(aso, "jailbreak") is True


class TestConfigValidation:
    def test_default_config(self):
        config = RedForgeConfig()
        # Will have errors because API keys are not set in test
        errors = config.validate()
        assert any("API_KEY" in e for e in errors)

    def test_invalid_digest_mode(self):
        config = RedForgeConfig(digest_mode="invalid")
        errors = config.validate()
        assert any("digest_mode" in e for e in errors)
