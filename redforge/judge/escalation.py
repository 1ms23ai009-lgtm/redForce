"""Layer 3: Human escalation queue.

Manages cases where Layer 1 and Layer 2 judges disagree or have
low confidence. Uses SQLite for persistent storage.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, Session, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class EscalationRecord(Base):
    __tablename__ = "escalations"

    escalation_id = Column(String, primary_key=True)
    engagement_id = Column(String, nullable=False, index=True)
    turn_id = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    layer1_verdict = Column(String)
    layer2_verdict = Column(String)
    layer1_confidence = Column(Float)
    layer2_confidence = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime, nullable=True)
    human_verdict = Column(String, nullable=True)
    resolver_notes = Column(Text, nullable=True)


class EscalationQueue:
    """Manages the human escalation queue for ambiguous judge results."""

    def __init__(self, db_path: str = "data/escalations.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine)

    def escalate(
        self,
        turn_data: dict,
        layer1_result: dict,
        layer2_result: dict,
    ) -> str:
        """Add a case to the escalation queue.

        Args:
            turn_data: Dict with engagement_id, turn_id, prompt, response
            layer1_result: Layer 1 classifier result
            layer2_result: Layer 2 LLM judge result

        Returns:
            The escalation_id
        """
        escalation_id = str(uuid.uuid4())

        record = EscalationRecord(
            escalation_id=escalation_id,
            engagement_id=turn_data.get("engagement_id", "unknown"),
            turn_id=str(turn_data.get("turn_id", "")),
            prompt=turn_data.get("prompt", ""),
            response=turn_data.get("response", ""),
            layer1_verdict=layer1_result.get("label", "unknown"),
            layer2_verdict=layer2_result.get("verdict", "unknown"),
            layer1_confidence=layer1_result.get("confidence", 0.0),
            layer2_confidence=layer2_result.get("confidence", 0.0),
        )

        with self._SessionFactory() as session:
            session.add(record)
            session.commit()

        logger.info(f"Escalation created: {escalation_id}")
        return escalation_id

    def resolve(
        self,
        escalation_id: str,
        human_verdict: str,
        notes: str = "",
    ) -> bool:
        """Resolve an escalated case with a human verdict.

        Args:
            escalation_id: The escalation to resolve
            human_verdict: The human's verdict (CONFIRMED, PARTIAL, etc.)
            notes: Optional resolver notes

        Returns:
            True if successfully resolved
        """
        with self._SessionFactory() as session:
            record = session.query(EscalationRecord).filter_by(
                escalation_id=escalation_id
            ).first()

            if not record:
                logger.warning(f"Escalation {escalation_id} not found")
                return False

            record.human_verdict = human_verdict
            record.resolver_notes = notes
            record.resolved_at = datetime.now(timezone.utc)
            session.commit()

        logger.info(f"Escalation {escalation_id} resolved: {human_verdict}")
        return True

    def get_pending(self) -> list[dict]:
        """Get all unresolved escalations.

        Returns:
            List of escalation dicts
        """
        with self._SessionFactory() as session:
            records = session.query(EscalationRecord).filter(
                EscalationRecord.human_verdict.is_(None)
            ).all()

            return [
                {
                    "escalation_id": r.escalation_id,
                    "engagement_id": r.engagement_id,
                    "turn_id": r.turn_id,
                    "prompt": r.prompt,
                    "response": r.response,
                    "layer1_verdict": r.layer1_verdict,
                    "layer2_verdict": r.layer2_verdict,
                    "layer1_confidence": r.layer1_confidence,
                    "layer2_confidence": r.layer2_confidence,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]

    def get_resolved(self, engagement_id: Optional[str] = None) -> list[dict]:
        """Get resolved escalations, optionally filtered by engagement.

        Returns:
            List of resolved escalation dicts
        """
        with self._SessionFactory() as session:
            query = session.query(EscalationRecord).filter(
                EscalationRecord.human_verdict.isnot(None)
            )
            if engagement_id:
                query = query.filter_by(engagement_id=engagement_id)

            records = query.all()
            return [
                {
                    "escalation_id": r.escalation_id,
                    "engagement_id": r.engagement_id,
                    "turn_id": r.turn_id,
                    "human_verdict": r.human_verdict,
                    "resolver_notes": r.resolver_notes,
                    "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                }
                for r in records
            ]
