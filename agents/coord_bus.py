from dataclasses import dataclass
from enum import Enum, auto


class CoordType(Enum):
    MISSION   = auto()   # new mission tile found
    MISSION_ACTIVE = auto()   # mission currently being worked by a teammate
    MISSION_DONE = auto()   # mission tile completed
    EXIT      = auto()   # exit tile found


@dataclass
class CoordMessage:
    coord_type: CoordType
    pos: tuple[int, int]
    sender_id: int          # agent index, for dedup / tiebreak