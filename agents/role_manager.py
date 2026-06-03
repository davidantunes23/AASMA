"""Role assignment functions for the human team.

Roles are reassigned event-driven by the simulation: when a mission is
discovered, completed, a worker is captured, or the exit unlocks.  Each
function is pure — it selects a candidate and returns a label; the caller
(simulation) commits the role mutation on the chosen agent spec.

Assignment strategy:
  WORKER  — agent closest (Manhattan) to any uncompleted mission.
  DECOY   — agent farthest (Manhattan) from all missions, so it draws the
             alien away from the work area.
  EXPLORER  — residual agent left after Worker and Decoy are assigned.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from agents.base import TeamRole


# ── Worker assignment ─────────────────────────────────────────────────────────


def assign_worker_greedy(
    agent_specs: Sequence[object],
    mission_positions: Iterable[tuple[int, int]],
) -> str | None:
    """Return the label of the human agent closest to any mission tile.

    Does NOT mutate agent state — the caller commits the role change.
    Returns None if there are no missions or no human agents.
    """
    missions = list(mission_positions)
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    if not missions or not human_specs:
        return None

    best = None
    best_dist = None
    for s in human_specs:
        pos = getattr(s.agent, "pos", None)
        if pos is None:
            continue
        min_d = min(abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in missions)  # nearest mission
        if best is None or min_d < best_dist:
            best = s
            best_dist = min_d

    return getattr(best, "label", None) if best is not None else None


# ── Role clearing ─────────────────────────────────────────────────────────────


def clear_roles(agent_specs: Sequence[object]) -> None:
    """Reset team_role to NONE on all human agents."""
    for s in agent_specs:
        if getattr(s, "role", None) == "human":
            try:
                setattr(s.agent, "team_role", TeamRole.NONE)
            except Exception:
                pass


# ── Decoy assignment ──────────────────────────────────────────────────────────


def assign_decoy_farthest(
    agent_specs: Sequence[object],
    mission_positions: Iterable[tuple[int, int]],
) -> str | None:
    """Return the label of the human agent farthest from all mission tiles.

    Also pushes the current mission list into each agent's mission_positions
    attribute so agents' local scoring heuristics stay up to date.
    Does NOT mutate agent.team_role — the caller commits the role change.
    Returns None if there are no missions or no human agents.
    """
    missions = list(mission_positions)
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    if not missions or not human_specs:
        return None

    # Sync mission list into each agent so their internal heuristics stay current.
    for s in human_specs:
        if hasattr(s.agent, "mission_positions"):
            setattr(s.agent, "mission_positions", missions)

    best = None
    best_score = -1  # any real distance will exceed this sentinel
    for s in human_specs:
        pos = getattr(s.agent, "pos", None)
        if pos is None:
            continue
        min_d = min(abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in missions)  # closest mission
        if min_d > best_score:  # decoy should be far from ALL missions
            best = s
            best_score = min_d

    return getattr(best, "label", None) if best is not None else None


# ── Omniscient worker assignment ──────────────────────────────────────────────


def assign_workers_omniscient(
    agent_specs: Sequence[object],
    mission_positions: Iterable[tuple[int, int]],
) -> list[tuple[object, tuple[int, int]]]:
    """Return an optimal greedy (agent_spec, mission) pairing.

    Each iteration picks the globally closest (agent, mission) pair and
    removes both from the remaining pools, so no two workers share a mission.
    Used by the omniscient agent which knows all mission positions upfront.

    Does NOT mutate agent state — the caller commits roles and assignments.
    Returns a list of (spec, mission_pos) pairs in assignment order.
    """
    missions = list(mission_positions)
    available_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    if not missions or not available_specs:
        return []

    n_workers = min(len(missions), len(available_specs))  # can't assign more workers than missions
    available_missions = list(missions)
    pairs: list[tuple[object, tuple[int, int]]] = []

    for _ in range(n_workers):
        if not available_specs or not available_missions:
            break
        best_spec = None
        best_mission = None
        best_dist = float("inf")
        # Find the globally closest (agent, mission) pair across all remaining combinations.
        for s in available_specs:
            pos = getattr(s.agent, "pos", None)
            if pos is None:
                continue
            for m in available_missions:
                d = abs(pos[0] - m[0]) + abs(pos[1] - m[1])
                if d < best_dist:
                    best_dist = d
                    best_spec = s
                    best_mission = m
        if best_spec is None:
            break
        pairs.append((best_spec, best_mission))
        available_specs.remove(best_spec)      # agent is now assigned
        available_missions.remove(best_mission)  # mission is now claimed

    return pairs


# ── Explorer assignment ─────────────────────────────────────────────────────────


def assign_explorer_residual(agent_specs: Sequence[object]) -> str | None:
    """Return the label of the first available human agent as the EXPLORER.

    Called after WORKER and DECOY are assigned — whoever is left becomes
    the Explorer. Ignores position; role geometry is handled by the agent itself.
    """
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]
    if not human_specs:
        return None
    return getattr(human_specs[0], "label", None)
