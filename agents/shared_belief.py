from __future__ import annotations

import numpy as np


class SharedBeliefMap:
    """Mutable belief map shared by reference between cooperative agents.

    Each CoopRoleHumanAgent points its own _known_map at shared_map.known_map,
    so any observation write is immediately visible to all teammates.

    Additionally, agents register their current navigation target here so
    that the frontier BFS can prefer cells not already claimed by a teammate.
    """

    UNKNOWN: int = -1

    def __init__(self, shape: tuple[int, int]) -> None:
        self.known_map: np.ndarray = np.full(shape, self.UNKNOWN, dtype=np.int16)
        self._positions: dict[int, tuple[int, int]] = {}
        self._targets: dict[int, tuple[int, int] | None] = {}

    # ── Agent state registration ──────────────────────────────────────────────

    def set_position(self, agent_id: int, pos: tuple[int, int]) -> None:
        self._positions[agent_id] = pos

    def set_target(self, agent_id: int, target: tuple[int, int] | None) -> None:
        self._targets[agent_id] = target

    # ── Query helpers ─────────────────────────────────────────────────────────

    def is_targeted_by_other(self, pos: tuple[int, int], my_id: int) -> bool:
        """True if any other agent has registered pos as their current frontier target."""
        return any(
            t == pos for aid, t in self._targets.items() if aid != my_id
        )

    def other_positions(self, my_id: int) -> set[tuple[int, int]]:
        """Positions of all agents except my_id."""
        return {p for aid, p in self._positions.items() if aid != my_id}
