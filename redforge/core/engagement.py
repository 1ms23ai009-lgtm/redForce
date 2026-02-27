"""Engagement manager and lifecycle control."""

import logging
import os
import uuid
from typing import Optional

from redforge.config import RedForgeConfig
from redforge.core.state import (
    AttackStateObject,
    create_initial_aso,
    get_current_turn,
)
from redforge.core.mdp import compute_coverage_percentage
from redforge.safety.audit import log_event
from redforge.target.connector import TargetConnector
from redforge.rl.high_level_policy import HighLevelPolicy
from redforge.rl.low_level_policy import LowLevelPolicy
from redforge.judge.ensemble import JudgeEnsemble
from redforge.strategy_library.library import StrategyLibrary
from redforge.agents.orchestrator import build_redforge_graph

logger = logging.getLogger(__name__)


class EngagementManager:
    """Manages the lifecycle of a REDFORGE red-team engagement."""

    def __init__(
        self,
        target_config: dict,
        customer_config: Optional[dict] = None,
        config: Optional[RedForgeConfig] = None,
    ):
        """Initialize the engagement.

        Args:
            target_config: Target system configuration dict with:
                endpoint_url, model_name, api_key_env_var, etc.
            customer_config: Optional customer settings (budget, depth, etc.)
            config: Optional REDFORGE configuration
        """
        self.config = config or RedForgeConfig()
        customer_config = customer_config or {}

        # Validate authorization scope
        self._validate_authorization(target_config, customer_config)

        # Apply customer overrides
        query_budget = customer_config.get(
            "query_budget", self.config.default_query_budget
        )
        depth = customer_config.get("depth", self.config.default_authorized_depth)

        # Ensure authorization scope is in target config
        if "authorization_scope" not in target_config:
            target_config["authorization_scope"] = {
                "depth": depth,
                "categories": customer_config.get("categories"),
                "query_budget": query_budget,
            }

        # Initialize ASO
        self.aso = create_initial_aso(
            target_config=target_config,
            query_budget=query_budget,
        )

        # Initialize components
        self.connector = TargetConnector(target_config)

        self.high_level_policy = HighLevelPolicy(
            model_path=(
                self.config.rl_policy_path
                if self.config.use_pretrained_policy
                else None
            ),
        )

        self.low_level_policy = LowLevelPolicy(
            model_name=self.config.attacker_model,
            api_key=self.config.openai_api_key,
            max_pair_iterations=self.config.max_pair_iterations,
            tap_branches=self.config.tap_branches,
        )

        self.judge = JudgeEnsemble(
            judge_model=self.config.judge_model,
            judge_api_key=self.config.anthropic_api_key,
        )

        self.strategy_library = StrategyLibrary(
            persist_dir=self.config.chroma_persist_dir
        )

        # Build LangGraph
        self.graph, self.checkpointer = build_redforge_graph(
            config=self.config,
            connector=self.connector,
            high_level_policy=self.high_level_policy,
            low_level_policy=self.low_level_policy,
            judge=self.judge,
            strategy_library=self.strategy_library,
        )

        # Log engagement start
        log_event(
            event_type="engagement_start",
            agent_name="engagement_manager",
            action_taken=f"Engagement initialized: {self.aso['engagement_id']}",
            reasoning_chain=(
                f"Target: {target_config.get('endpoint_url', 'unknown')}, "
                f"Budget: {query_budget}, Depth: {depth}"
            ),
            aso=self.aso,
        )

    def run(self) -> dict:
        """Execute the LangGraph graph for the full engagement.

        Returns:
            Final report dict
        """
        logger.info(f"Starting engagement {self.aso['engagement_id']}")

        thread_config = {
            "configurable": {"thread_id": self.aso["engagement_id"]}
        }

        try:
            # Run the graph
            final_state = self.graph.invoke(self.aso, config=thread_config)

            # Extract report
            report = final_state.get("_final_report", {})
            if not report:
                from redforge.reporting.report_generator import generate_report
                report = generate_report(final_state)

            # Save report
            self._save_report(report)

            logger.info(
                f"Engagement {self.aso['engagement_id']} complete. "
                f"Exploits: {len(final_state.get('confirmed_exploits', []))}"
            )
            return report

        except Exception as e:
            logger.error(f"Engagement failed: {e}")
            log_event(
                event_type="engagement_end",
                agent_name="engagement_manager",
                action_taken=f"Engagement failed: {str(e)}",
                reasoning_chain="Unhandled error during graph execution",
                aso=self.aso,
            )
            raise
        finally:
            self.connector.close()

    def get_status(self) -> dict:
        """Return current engagement status summary."""
        return {
            "engagement_id": self.aso["engagement_id"],
            "turns_completed": get_current_turn(self.aso),
            "exploits_found": len(self.aso.get("confirmed_exploits", [])),
            "partial_bypasses": len(self.aso.get("partial_bypass_registry", [])),
            "budget_remaining": self.aso.get("query_budget_remaining", 0),
            "kill_switch": self.aso.get("kill_switch_triggered", False),
            "coverage_percentage": compute_coverage_percentage(self.aso),
            "active_hypotheses": len(self.aso.get("active_hypotheses", [])),
            "strategy_library_size": self.strategy_library.count(),
        }

    def _validate_authorization(
        self, target_config: dict, customer_config: dict
    ) -> None:
        """Validate that authorization scope is properly defined."""
        endpoint = target_config.get("endpoint_url")
        if not endpoint:
            raise ValueError("target_config must include endpoint_url")

    def _save_report(self, report: dict) -> None:
        """Save the engagement report to disk."""
        import json

        os.makedirs(self.config.reports_dir, exist_ok=True)
        report_path = os.path.join(
            self.config.reports_dir,
            f"report_{self.aso['engagement_id']}.json",
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")
