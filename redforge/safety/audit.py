"""Immutable audit trail logger with hash-chaining for tamper detection."""

import hashlib
import json
import uuid
from datetime import datetime, timezone

from redforge.core.state import AttackStateObject


def _compute_aso_hash(aso: AttackStateObject) -> str:
    """Compute a SHA-256 hash of the current ASO state (excluding audit_log to avoid recursion)."""
    aso_copy = dict(aso)
    aso_copy.pop("audit_log", None)
    serialized = json.dumps(aso_copy, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _compute_entry_hash(entry: dict) -> str:
    """Compute SHA-256 hash of an audit entry (without entry_hash field)."""
    entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
    serialized = json.dumps(entry_copy, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def get_previous_entry_hash(aso: AttackStateObject) -> str:
    """Get the hash of the last audit log entry, or genesis hash if empty."""
    if aso["audit_log"]:
        return aso["audit_log"][-1]["entry_hash"]
    return hashlib.sha256(b"REDFORGE_GENESIS").hexdigest()


def log_event(
    event_type: str,
    agent_name: str,
    action_taken: str,
    reasoning_chain: str,
    aso: AttackStateObject,
) -> str:
    """Append an immutable audit entry to the ASO audit log.

    Args:
        event_type: One of "agent_action", "judge_verdict", "exploit_confirmed",
                    "kill_switch_check", "kill_switch_triggered",
                    "engagement_start", "engagement_end", "escalation_created"
        agent_name: Name of the agent performing the action
        action_taken: Description of the action
        reasoning_chain: Reasoning behind the action
        aso: The current Attack State Object (mutated in-place)

    Returns:
        The entry hash string.
    """
    previous_hash = get_previous_entry_hash(aso)
    aso_hash = _compute_aso_hash(aso)

    entry = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "action_taken": action_taken,
        "reasoning_chain": reasoning_chain,
        "aso_snapshot_hash": aso_hash,
        "previous_entry_hash": previous_hash,
        "entry_hash": "",  # placeholder
    }

    entry["entry_hash"] = _compute_entry_hash(entry)
    aso["audit_log"].append(entry)
    return entry["entry_hash"]


def verify_audit_chain(audit_log: list[dict]) -> tuple[bool, list[str]]:
    """Verify the integrity of the audit chain.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    if not audit_log:
        return True, []

    genesis_hash = hashlib.sha256(b"REDFORGE_GENESIS").hexdigest()

    for i, entry in enumerate(audit_log):
        # Verify entry hash
        expected_hash = _compute_entry_hash(entry)
        if entry["entry_hash"] != expected_hash:
            errors.append(f"Entry {i} ({entry['event_id']}): hash mismatch")

        # Verify chain linkage
        if i == 0:
            if entry["previous_entry_hash"] != genesis_hash:
                errors.append(f"Entry 0: previous_entry_hash is not genesis hash")
        else:
            if entry["previous_entry_hash"] != audit_log[i - 1]["entry_hash"]:
                errors.append(
                    f"Entry {i} ({entry['event_id']}): chain broken, "
                    f"previous_entry_hash does not match entry {i-1}"
                )

    return len(errors) == 0, errors


VALID_EVENT_TYPES = frozenset(
    [
        "agent_action",
        "judge_verdict",
        "exploit_confirmed",
        "kill_switch_check",
        "kill_switch_triggered",
        "engagement_start",
        "engagement_end",
        "escalation_created",
    ]
)
