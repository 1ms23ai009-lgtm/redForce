"""Target topology discovery - coordinates probing and maps the target system."""

import logging
from typing import Optional

from redforge.target.connector import TargetConnector
from redforge.target.probe import TargetProbe
from redforge.core.state import AttackStateObject
from redforge.safety.audit import log_event

logger = logging.getLogger(__name__)


class TopologyDiscovery:
    """Discovers and maps the target system's topology."""

    def __init__(
        self,
        connector: TargetConnector,
        parser_model: str = "gpt-4o-mini",
        parser_api_key: Optional[str] = None,
    ):
        self._probe = TargetProbe(
            connector=connector,
            parser_model=parser_model,
            parser_api_key=parser_api_key,
        )

    def discover(self, aso: AttackStateObject) -> AttackStateObject:
        """Run topology discovery and update the ASO.

        This is the LangGraph node function for topology_discovery.

        Args:
            aso: Current attack state object

        Returns:
            Updated ASO with target_topology populated
        """
        log_event(
            event_type="agent_action",
            agent_name="topology_discovery",
            action_taken="Starting passive topology probing",
            reasoning_chain="Initial discovery phase to map target before attacking",
            aso=aso,
        )

        try:
            topology = self._probe.run_all_probes()

            # Merge with any existing topology data
            existing = aso["target_topology"]
            merged = {
                "agents": _merge_lists(existing.get("agents", []), topology.get("agents", [])),
                "tools": _merge_lists(existing.get("tools", []), topology.get("tools", [])),
                "memory_systems": _merge_lists(
                    existing.get("memory_systems", []),
                    topology.get("memory_systems", []),
                ),
                "policies": _merge_lists(
                    existing.get("policies", []), topology.get("policies", [])
                ),
                "guardrails": _merge_lists(
                    existing.get("guardrails", []), topology.get("guardrails", [])
                ),
            }
            aso["target_topology"] = merged

            # Consume budget for probe queries
            probes_used = len([k for k in topology if topology[k]])
            aso["query_budget_remaining"] = max(
                0, aso["query_budget_remaining"] - 6  # 6 probes
            )

            log_event(
                event_type="agent_action",
                agent_name="topology_discovery",
                action_taken=(
                    f"Discovery complete: {len(merged['agents'])} agents, "
                    f"{len(merged['tools'])} tools, "
                    f"{len(merged['memory_systems'])} memory systems, "
                    f"{len(merged['policies'])} policies, "
                    f"{len(merged['guardrails'])} guardrails"
                ),
                reasoning_chain="Topology data merged into ASO",
                aso=aso,
            )

        except Exception as e:
            logger.error(f"Topology discovery failed: {e}")
            log_event(
                event_type="agent_action",
                agent_name="topology_discovery",
                action_taken=f"Discovery failed: {str(e)}",
                reasoning_chain="Error during probing, continuing with limited topology",
                aso=aso,
            )

        return aso


def _merge_lists(existing: list, new: list) -> list:
    """Merge two lists, avoiding duplicates."""
    result = list(existing)
    for item in new:
        if item not in result:
            result.append(item)
    return result
