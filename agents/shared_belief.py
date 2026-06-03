"""Shared map array and navigation-target registry for cooperative agents.

All CoopRoleHumanAgents on the same team point their ``_known_map`` at the
same ``SharedBeliefMap.known_map`` numpy array.  Because it is shared by
reference, any observation written by one agent is immediately visible to
every other agent — no copying or message passing required for map data.

In addition, each agent registers its current frontier target here so the
cooperative BFS can skip cells already claimed by a teammate, naturally
partitioning exploration across the team.
"""
from __future__ import annotations

import numpy as np


class SharedBeliefMap:
    """Mutable belief map shared by reference between cooperative agents.

    Each CoopRoleHumanAgent points its own _known_map at shared_map.known_map,
    so any observation write is immediately visible to all teammates.

    Additionally, agents register their current navigation target here so
    that the frontier BFS can prefer cells not already claimed by a teammate.
    """

    UNKNOWN: int = -1  # sentinel for unobserved cells, matches agent UNKNOWN constants

    def __init__(self, shape: tuple[int, int]) -> None:
        self.known_map: np.ndarray = np.full(shape, self.UNKNOWN, dtype=np.int16)  # shared tile map
        self._positions: dict[int, tuple[int, int]] = {}       # agent_id → current position
        self._targets: dict[int, tuple[int, int] | None] = {}  # agent_id → current frontier target

    # ── Agent state registration ──────────────────────────────────────────────

    def set_position(self, agent_id: int, pos: tuple[int, int]) -> None:
        """Update the registered position for an agent (called each step)."""
        self._positions[agent_id] = pos

    def set_target(self, agent_id: int, target: tuple[int, int] | None) -> None:
        """Register the frontier cell this agent is currently heading toward."""
        self._targets[agent_id] = target

    def clear_agent(self, agent_id: int) -> None:
        """Remove all stored state for an agent.

        Called when an agent is captured or escapes so stale target claims
        do not block teammates from exploring those frontiers.
        """
        self._targets.pop(agent_id, None)
        self._positions.pop(agent_id, None)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def is_targeted_by_other(self, pos: tuple[int, int], my_id: int) -> bool:
        """True if any other agent has registered pos as their current frontier target."""
        # Used by BFS to skip cells already claimed, partitioning exploration across the team.
        return any(t == pos for aid, t in self._targets.items() if aid != my_id)

    def other_positions(self, my_id: int) -> set[tuple[int, int]]:
        """Positions of all agents except my_id — used by the decoy noise proximity check."""
        # Lets the decoy avoid emitting noise when a teammate is standing nearby.
        return {p for aid, p in self._positions.items() if aid != my_id}
