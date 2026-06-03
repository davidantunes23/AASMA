"""Lightweight coordination message bus used by role-aware human agents.

Agents queue CoordMessages in their outbox during observe(); the simulation
relays them to all teammates each step via flush_outbox() / receive_coords().
This gives agents shared awareness of missions, the exit, and task claims
without any direct agent-to-agent coupling.
"""
from dataclasses import dataclass
from enum import Enum, auto


class CoordType(Enum):
    MISSION        = auto()  # a new mission tile was spotted
    MISSION_ACTIVE = auto()  # this agent is currently heading to / working a mission
    MISSION_DONE   = auto()  # a mission tile has been completed; remove from all lists
    EXIT           = auto()  # the exit tile was spotted for the first time


@dataclass
class CoordMessage:
    """A single coordination message broadcast to all teammates."""
    coord_type: CoordType
    pos: tuple[int, int]  # (y, x) tile position the message refers to
    sender_id: int        # agent_id of the sender — used for deduplication and tiebreaking
