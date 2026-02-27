"""Kill switch protocol for REDFORGE safety enforcement.

The kill switch runs BEFORE every target interaction. When triggered,
all agents halt immediately.
"""

import re
import time
from datetime import datetime, timezone

from redforge.core.state import AttackStateObject
from redforge.safety.audit import log_event


# Patterns that trigger immediate halt - these represent content
# that must never be generated even in red-team context
PROHIBITED_CONTENT_PATTERNS = [
    # Weapons/explosives step-by-step
    re.compile(
        r"step[- ]by[- ]step.{0,50}(synthesize|manufacture|build|assemble).{0,50}"
        r"(explosive|bomb|weapon|nerve agent|chemical weapon)",
        re.IGNORECASE,
    ),
    # CSAM indicators
    re.compile(
        r"(child|minor|underage).{0,30}(sexual|explicit|nude|pornograph)",
        re.IGNORECASE,
    ),
    # Targeted individual doxing
    re.compile(
        r"(home address|phone number|social security|SSN).{0,50}(of|for|belonging to)\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
        re.IGNORECASE,
    ),
]


def check_content_safety(content: str) -> tuple[bool, str]:
    """Check if content contains prohibited patterns.

    Returns:
        (is_safe, reason) - is_safe=True means content is OK to send.
    """
    for pattern in PROHIBITED_CONTENT_PATTERNS:
        if pattern.search(content):
            return False, f"Prohibited content pattern detected: {pattern.pattern[:60]}..."
    return True, ""


def check_kill_switch(
    aso: AttackStateObject,
    config: dict | None = None,
    prompt_to_send: str | None = None,
) -> tuple[bool, str]:
    """Check all kill switch conditions.

    Args:
        aso: Current attack state object
        config: Engagement configuration (for max_hours, authorized scope)
        prompt_to_send: If provided, check this prompt for prohibited content

    Returns:
        (triggered, reason) - triggered=True means HALT.
    """
    if config is None:
        config = {}

    # Condition 1: Check generated prompt for prohibited content
    if prompt_to_send:
        is_safe, reason = check_content_safety(prompt_to_send)
        if not is_safe:
            _trigger_kill_switch(aso, f"Content safety violation: {reason}")
            return True, reason

    # Condition 2: Query budget exceeded
    if aso["query_budget_remaining"] <= 0:
        reason = "Query budget exhausted"
        _trigger_kill_switch(aso, reason)
        return True, reason

    # Condition 3: Engagement duration exceeded
    max_hours = config.get("max_hours", 2.0)
    if aso["audit_log"]:
        start_events = [
            e for e in aso["audit_log"] if e["event_type"] == "engagement_start"
        ]
        if start_events:
            start_time = datetime.fromisoformat(start_events[0]["timestamp"])
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600
            if elapsed > max_hours:
                reason = f"Engagement duration exceeded: {elapsed:.1f}h > {max_hours}h"
                _trigger_kill_switch(aso, reason)
                return True, reason

    # Condition 4: Authorization scope exceeded
    auth_scope = aso["target_config"].get("authorization_scope", {})
    authorized_categories = auth_scope.get("categories", None)
    if authorized_categories:
        for exploit in aso["confirmed_exploits"]:
            if exploit["category"] not in authorized_categories:
                reason = (
                    f"Exploit in unauthorized category: {exploit['category']}"
                )
                _trigger_kill_switch(aso, reason)
                return True, reason

    # Condition 5: Critical severity in unauthorized category
    authorized_categories = auth_scope.get("categories", None)
    if authorized_categories:
        for exploit in aso["confirmed_exploits"]:
            if (
                exploit["severity"] == "critical"
                and exploit["category"] not in authorized_categories
            ):
                reason = (
                    f"Critical exploit in unauthorized category: {exploit['category']}"
                )
                _trigger_kill_switch(aso, reason)
                return True, reason

    # Log successful check
    log_event(
        event_type="kill_switch_check",
        agent_name="kill_switch",
        action_taken="Kill switch check passed",
        reasoning_chain="All conditions clear",
        aso=aso,
    )
    return False, ""


def _trigger_kill_switch(aso: AttackStateObject, reason: str) -> None:
    """Trigger the kill switch and log the event."""
    aso["kill_switch_triggered"] = True
    log_event(
        event_type="kill_switch_triggered",
        agent_name="kill_switch",
        action_taken=f"Kill switch triggered: {reason}",
        reasoning_chain=reason,
        aso=aso,
    )


def sanitize_content(content: str) -> str:
    """Remove prohibited content patterns from logs."""
    sanitized = content
    for pattern in PROHIBITED_CONTENT_PATTERNS:
        sanitized = pattern.sub("[REDACTED - PROHIBITED CONTENT]", sanitized)
    return sanitized
