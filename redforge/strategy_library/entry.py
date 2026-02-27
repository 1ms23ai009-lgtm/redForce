"""Strategy entry data model for the REDFORGE strategy library."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StrategyEntry:
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""
    technique_name: str = ""
    technique_description: str = ""
    prompt_structure: str = ""  # abstract structure, NOT the specific prompt
    guardrail_trigger_pattern: str = ""
    success_indicators: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)
    success_rate: float = 0.0
    times_used: int = 0
    times_succeeded: int = 0
    model_types_tested: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_used_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    novelty_score: float = 1.0  # 1.0 if completely novel, 0.0 if well-known

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "category": self.category,
            "technique_name": self.technique_name,
            "technique_description": self.technique_description,
            "prompt_structure": self.prompt_structure,
            "guardrail_trigger_pattern": self.guardrail_trigger_pattern,
            "success_indicators": self.success_indicators,
            "failure_indicators": self.failure_indicators,
            "success_rate": self.success_rate,
            "times_used": self.times_used,
            "times_succeeded": self.times_succeeded,
            "model_types_tested": self.model_types_tested,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "novelty_score": self.novelty_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyEntry":
        """Create a StrategyEntry from a dictionary."""
        return cls(
            strategy_id=data.get("strategy_id", str(uuid.uuid4())),
            category=data.get("category", ""),
            technique_name=data.get("technique_name", ""),
            technique_description=data.get("technique_description", ""),
            prompt_structure=data.get("prompt_structure", ""),
            guardrail_trigger_pattern=data.get("guardrail_trigger_pattern", ""),
            success_indicators=data.get("success_indicators", []),
            failure_indicators=data.get("failure_indicators", []),
            success_rate=data.get("success_rate", 0.0),
            times_used=data.get("times_used", 0),
            times_succeeded=data.get("times_succeeded", 0),
            model_types_tested=data.get("model_types_tested", []),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_used_at=data.get("last_used_at", datetime.now(timezone.utc).isoformat()),
            novelty_score=data.get("novelty_score", 1.0),
        )

    def update_success_metrics(self, succeeded: bool) -> None:
        """Update the entry after a use attempt."""
        self.times_used += 1
        if succeeded:
            self.times_succeeded += 1
        self.success_rate = self.times_succeeded / self.times_used if self.times_used > 0 else 0.0
        self.last_used_at = datetime.now(timezone.utc).isoformat()

    def get_embedding_text(self) -> str:
        """Get text representation for embedding/similarity computation."""
        return (
            f"{self.technique_name}: {self.technique_description}. "
            f"Trigger pattern: {self.guardrail_trigger_pattern}. "
            f"Category: {self.category}."
        )
