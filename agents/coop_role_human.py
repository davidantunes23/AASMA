"""Cooperative role-aware human agent with shared map knowledge.

Extends RoleHumanAgent with two cooperative mechanisms:

1. **Shared map** — all CoopRoleHumanAgents on a team alias their ``_known_map``
   to the same ``SharedBeliefMap.known_map`` array.  Any tile observed by one
   agent is instantly visible to all teammates at zero cost — no messages needed
   for map data.

2. **Claimed frontiers** — each agent registers its current exploration target
   in the shared map.  The cooperative BFS skips frontiers already claimed by a
   teammate (Pass 1), falling back to any frontier only when all are claimed
   (Pass 2).  This naturally partitions exploration across the team.

3. **Noise suppression** — the DECOY role suppresses loud-noise bursts when a
   teammate is within TEAMMATE_NOISE_BLOCK_RADIUS cells, preventing the alien
   from being drawn toward the whole team instead of away from missions.
"""
from __future__ import annotations

from collections import deque

from agents.base import Direction
from agents.role_human import RoleHumanAgent
from agents.coord_bus import CoordType, CoordMessage
from agents.shared_belief import SharedBeliefMap


class CoopRoleHumanAgent(RoleHumanAgent):
    """RoleHumanAgent extended with a shared belief map and cooperative frontier BFS."""

    # Decoy suppresses noise if any teammate is closer than this (Manhattan cells).
    # Prevents the alien being attracted toward the whole group instead of away from missions.
    TEAMMATE_NOISE_BLOCK_RADIUS = 10

    def __init__(
        self,
        start_pos: tuple[int, int],
        start_dir: Direction = Direction.NORTH,
        view_length: int = 6,
    ) -> None:
        super().__init__(start_pos, start_dir, view_length)
        self.shared_map: SharedBeliefMap | None = None  # wired up by the simulation before first observe()

    # ── Memory initialisation ─────────────────────────────────────────────────

    def _init_memory(self, obs) -> None:
        """Alias _known_map to the shared array the first time observe() is called.

        After aliasing, all tile writes via _integrate_observation go directly into
        the shared array without any extra copying or messaging.
        """
        if (
            self.shared_map is not None
            and self.shared_map.known_map.shape == obs.shape
        ):
            if self._known_map is not self.shared_map.known_map:
                prev_known_exit = self._known_exit
                self._known_map = self.shared_map.known_map  # alias — not a copy
                self._known_exit = prev_known_exit           # preserve already-known exit
                self._observed_aliens = set()
            return
        # Solo agent or shape mismatch: fall back to an independent map.
        super()._init_memory(obs)

    # ── Observation ───────────────────────────────────────────────────────────

    def _integrate_observation(self, obs) -> None:
        """Write visible tiles into the shared map and update position registry."""
        super()._integrate_observation(obs)
        if self.shared_map is not None:
            self.shared_map.set_position(self.agent_id, self.pos)

    def receive_coords(self, messages: list[CoordMessage]) -> None:
        """Accept discovery messages and mirror mission-active claims into the shared map."""
        super().receive_coords(messages)

        if self.shared_map is None:
            return
        for msg in messages:
            try:
                if msg.coord_type == CoordType.MISSION_ACTIVE:
                    # Register teammate's mission target so our BFS avoids it.
                    self.shared_map.set_target(self.agent_id, msg.pos)
                elif msg.coord_type == CoordType.MISSION_DONE:
                    # Clear any target pointing to the completed mission.
                    cur = getattr(self.shared_map, "_targets", {}).get(self.agent_id)
                    if cur == msg.pos:
                        self.shared_map.set_target(self.agent_id, None)
            except Exception:
                pass

    # ── Noise suppression ─────────────────────────────────────────────────────

    def _teammates_too_close_for_noise(self) -> bool:
        """True if any teammate is within TEAMMATE_NOISE_BLOCK_RADIUS cells."""
        if self.shared_map is None:
            return False
        for oy, ox in self.shared_map.other_positions(self.agent_id):
            if abs(self.pos[0] - oy) + abs(self.pos[1] - ox) <= self.TEAMMATE_NOISE_BLOCK_RADIUS:
                return True
        return False

    def _update_decoy_loud_noise(self, can_start_noise: bool) -> bool:
        if self._teammates_too_close_for_noise():
            can_start_noise = False  # suppress: would draw alien toward the group
        return super()._update_decoy_loud_noise(can_start_noise)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        _player_pos: tuple[int, int],
        _heard_pos: tuple[int, int] | None = None,
        _step_num: int = 0,
    ) -> tuple[int, int]:
        result = super().step(_player_pos, _heard_pos, _step_num)
        if self.shared_map is not None:
            self.shared_map.set_position(self.agent_id, result)  # keep registry current after move
        return result

    # ── Cooperative frontier BFS ──────────────────────────────────────────────

    def _next_step_to_nearest_frontier(self) -> tuple[int, int] | None:
        return self._coop_bfs_frontier(self._is_frontier)

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        return self._coop_bfs_frontier(self._is_floor_frontier)

    def _coop_bfs_frontier(self, is_frontier) -> tuple[int, int] | None:
        """BFS to nearest frontier, preferring cells unclaimed by teammates.

        Pass 1 — skip frontiers already targeted by another agent.
        Pass 2 (fallback) — accept any frontier, preventing deadlock when all
                            reachable frontiers are claimed by teammates.

        Registers the chosen destination in shared_map so teammates see it.
        """
        if self.shared_map is None:
            return self._bfs_next_step(is_frontier)

        # Pass 1: nearest unclaimed frontier
        first_step, dest = self._bfs_frontier_with_dest(is_frontier, prefer_unclaimed=True)
        if first_step is not None:
            self.shared_map.set_target(self.agent_id, dest)
            return first_step

        # Pass 2: any frontier (ignoring claims)
        first_step, dest = self._bfs_frontier_with_dest(is_frontier, prefer_unclaimed=False)
        self.shared_map.set_target(self.agent_id, dest)
        return first_step

    def _bfs_frontier_with_dest(
        self,
        is_frontier,
        prefer_unclaimed: bool,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """BFS returning (first_step, destination).

        When prefer_unclaimed is True, only stops at frontiers not targeted by
        another agent. Returns (None, None) if no matching cell is reachable.
        """
        start = self.pos
        if not self._in_bounds(*start):
            return None, None

        queue = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()
            if current != start and is_frontier(current):
                if prefer_unclaimed and self.shared_map is not None:
                    if self.shared_map.is_targeted_by_other(current, self.agent_id):
                        # Keep BFS going — don't stop at a claimed frontier.
                        for neighbor, _ in self._walkable_neighbors(current):
                            if neighbor not in parents:
                                parents[neighbor] = current
                                queue.append(neighbor)
                        continue
                first_step = self._first_step_from_path(current, parents)
                return first_step, current
            for neighbor, _ in self._walkable_neighbors(current):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)

        return None, None
