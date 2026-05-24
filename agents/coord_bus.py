from dataclasses import dataclass, field
from enum import Enum, auto


class CoordType(Enum):
    MISSION   = auto()   # new mission tile found
    EXIT      = auto()   # exit tile found


@dataclass
class CoordMessage:
    coord_type: CoordType
    pos: tuple[int, int]
    sender_id: int          # agent index, for dedup / tiebreak