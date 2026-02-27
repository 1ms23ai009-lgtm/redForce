"""Similarity-based strategy retrieval and strategy addition logic."""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from redforge.strategy_library.entry import StrategyEntry
from redforge.strategy_library.library import StrategyLibrary
from redforge.config import STRATEGY_CATEGORIES

logger = logging.getLogger(__name__)


def retrieve_relevant_strategies(
    library: StrategyLibrary,
    category: str,
    target_description: str,
    current_aso: dict,
    top_k: int = 3,
) -> list[StrategyEntry]:
    """Retrieve relevant strategies from the library.

    Uses semantic similarity on technique_description + guardrail_trigger_pattern.
    Filters by: category match, success_rate > 0.1, not recently failed on similar target.

    Args:
        library: The strategy library instance
        category: Attack category to filter by
        target_description: Description of the current target
        current_aso: Current ASO state dict
        top_k: Number of strategies to return

    Returns:
        List of top_k most relevant StrategyEntry objects
    """
    # Build query text from target and context
    query = f"Category: {category}. Target: {target_description}."

    # Include recent failure info for filtering
    recent_failures = set()
    for entry in current_aso.get("attack_history", [])[-20:]:
        verdict = entry.get("judge_verdict", {})
        if isinstance(verdict, dict) and verdict.get("verdict") == "FAILED":
            recent_failures.add(entry.get("strategy_category", ""))

    results = library.search(query, n_results=top_k * 3, category=category)

    # Filter and rank
    filtered = []
    for entry, similarity in results:
        # Skip low success rate
        if entry.success_rate < 0.1 and entry.times_used > 3:
            continue
        # Deprioritize recently failed categories
        if entry.category in recent_failures:
            similarity *= 0.5
        filtered.append((entry, similarity))

    # Sort by similarity
    filtered.sort(key=lambda x: x[1], reverse=True)

    return [entry for entry, _ in filtered[:top_k]]


def add_strategy(
    library: StrategyLibrary,
    exploit_result: dict,
    aso: dict,
    novelty_threshold: float = 0.7,
) -> StrategyEntry:
    """Add a new strategy or update existing one based on an exploit result.

    Checks similarity against existing entries:
    - similarity < threshold: add as new entry (novel strategy)
    - similarity >= threshold: update existing entry's success metrics

    Args:
        library: The strategy library instance
        exploit_result: Dict with exploit details (from confirmed exploit)
        aso: Current ASO state
        novelty_threshold: Threshold for considering a strategy novel

    Returns:
        The created or updated StrategyEntry
    """
    # Build a candidate entry from the exploit result
    candidate = StrategyEntry(
        category=exploit_result.get("category", "unknown"),
        technique_name=exploit_result.get("technique_name", "unnamed"),
        technique_description=exploit_result.get("description", ""),
        prompt_structure=_extract_prompt_structure(exploit_result),
        guardrail_trigger_pattern=exploit_result.get("guardrail_trigger", ""),
        success_indicators=exploit_result.get("success_indicators", []),
        failure_indicators=exploit_result.get("failure_indicators", []),
        success_rate=1.0,  # just succeeded
        times_used=1,
        times_succeeded=1,
        model_types_tested=[aso.get("target_config", {}).get("model_name", "unknown")],
        novelty_score=1.0,
    )

    # Check for similar existing entry
    match = library.find_similar(candidate, threshold=novelty_threshold)

    if match:
        existing, similarity = match
        existing.update_success_metrics(succeeded=True)
        # Add model type if new
        model = aso.get("target_config", {}).get("model_name", "unknown")
        if model not in existing.model_types_tested:
            existing.model_types_tested.append(model)
        # Decrease novelty as it's matching an existing pattern
        existing.novelty_score = max(0.0, 1.0 - similarity)
        library.update(existing)
        logger.info(
            f"Updated existing strategy {existing.strategy_id} "
            f"(similarity: {similarity:.3f})"
        )
        return existing
    else:
        # Novel strategy
        library.add(candidate)
        logger.info(
            f"Added novel strategy {candidate.strategy_id}: "
            f"{candidate.technique_name}"
        )
        return candidate


def _extract_prompt_structure(exploit_result: dict) -> str:
    """Extract an abstract prompt structure from an exploit result.

    Abstracts away specific content to capture the technique pattern.
    """
    trace = exploit_result.get("reproducible_trace", [])
    if not trace:
        return "unknown structure"

    # Use the first prompt as the structural template
    first_prompt = trace[0] if isinstance(trace[0], str) else str(trace[0])

    # Abstract common patterns
    structure = first_prompt
    # Replace specific names/values with placeholders
    if len(structure) > 200:
        structure = structure[:200] + "..."

    return f"Multi-step ({len(trace)} turns): {structure}"
