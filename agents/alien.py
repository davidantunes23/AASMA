from enum import Enum

class Direction(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4

class Action(Enum):
    WALK = 1

class AlienAgent:
    def __init__(self, start_pos: tuple, start_dir: Direction):
        self.position = start_pos
        self.direction = start_dir

    def _act(self, obs) -> tuple:
        return (Action.WALK, self.direction)