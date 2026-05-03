#!/usr/bin/env python3
"""
map_generator.py  —  Alien Isolation-inspired 2D grid map generator.

Tile legend:
    0 WALL          impassable
    1 FLOOR         passable by both agents
    2 VENT          alien-only shortcut (passable for alien, blocked for player)
    3 HIDE          hiding spot (player can hide; blocks alien LOS)
    4 PLAYER_START  player spawn
    5 ALIEN_START   alien spawn
    6 EXIT          player goal

Alpha parameter  alpha in [-1, +1]:
    alpha < 0  -> player-favoured  (more hiding spots per room, fewer vents)
    alpha = 0  -> balanced
    alpha > 0  -> alien-favoured   (more vents, fewer hiding spots per room)
"""

import random
import json
import os
import argparse
import math
from enum import IntEnum
from collections import deque

import numpy as np

# ── Tile constants ──────────────────────────────────────────────────────────────
class Tile(IntEnum):
    WALL = 0
    FLOOR = 1
    VENT = 2
    HIDE = 3
    PLAYER_START = 4
    ALIEN_START = 5
    EXIT = 6

TILE_CHAR = {
    Tile.WALL:         "##",
    Tile.FLOOR:        "  ",
    Tile.VENT:         "VV",
    Tile.HIDE:         "HH",
    Tile.PLAYER_START: "PP",
    Tile.ALIEN_START:  "AA",
    Tile.EXIT:         "EE",
}

TILE_NAME = {
    Tile.WALL: "wall", Tile.FLOOR: "floor", Tile.VENT: "vent",
    Tile.HIDE: "hide", Tile.PLAYER_START: "player", Tile.ALIEN_START: "alien", Tile.EXIT: "exit",
}


# ── MapGenerator ───────────────────────────────────────────────────────────────
class MapGenerator:
    """
    BSP-style room placement + L-shaped corridor carving.
    After carving, special tiles (vents, hiding spots, spawns, exit) are
    scattered according to the alpha advantage parameter.
    """

    def __init__(
        self,
        width: int = 60,
        height: int = 40,
        alpha: float = 0.0,
        seed: int | None = None,
        min_room_size: int | None = None,
        max_room_size: int | None = None,
        max_rooms: int | None = None,
        max_hides_per_room: int | None = None,
    ):
        if not -1.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [-1, +1]")
        self.width = width
        self.height = height
        self.alpha = alpha
        self.seed = seed if seed is not None else random.randint(0, 2**31)

        scale = max(width / 60.0, height / 40.0)
        if min_room_size is None:
            min_room_size = max(3, int(round(3 * scale)))
        if max_room_size is None:
            max_room_size = max(min_room_size + 2, int(round(8 * scale)))
        if max_rooms is None:
            max_rooms = max(12, int(round(12 * scale)))

        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.max_rooms = max_rooms
        if max_hides_per_room is None:
            self.max_hides_per_room = int(max_room_size * max_room_size * 0.10)
        else:
            if max_hides_per_room < 0:
                raise ValueError("max_hides_per_room must be >= 0")
            self.max_hides_per_room = max_hides_per_room

        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        self.grid: np.ndarray | None = None
        self.rooms: list[tuple] = []
        self.player_pos: tuple | None = None
        self.alien_pos:  tuple | None = None
        self.exit_pos:   tuple | None = None
        self.metadata:   dict = {}

    # ── Tile behaviour from alpha ───────────────────────────────────────────────
    @property
    def vent_probability(self) -> float:
        """Fraction of floor tiles converted to vents. Increases with alpha."""
        return 0.5 + 0.5 * self.alpha

    def hide_count_distribution(self, room_max_hides: int) -> list[float]:
        """
        Probability of placing k hiding spots per room for k in
        [0, room_max_hides].

        alpha < 0 shifts probability toward higher k (player-favoured).
        alpha > 0 shifts probability toward lower k (alien-favoured).
        alpha = 0 produces a uniform distribution.
        """
        if room_max_hides <= 0:
            return [1.0]

        counts = list(range(room_max_hides + 1))
        tilt = -2.2 * self.alpha
        logits = [tilt * k for k in counts]
        max_logit = max(logits)
        weights = [math.exp(v - max_logit) for v in logits]
        total = sum(weights)
        return [w / total for w in weights]

    def _room_hide_max(self, room: tuple, floor_cells_available: int) -> int:
        """
        Scale each room's hide cap by room area, clipped by global max.
        Larger rooms can host more hiding spots.
        """
        if self.max_hides_per_room <= 0 or floor_cells_available <= 0:
            return 0

        _, _, w, h = room
        room_area = w * h
        max_room_area = max(1, self.max_room_size * self.max_room_size)
        scaled = math.ceil(self.max_hides_per_room * (room_area / max_room_area))
        return max(1, min(self.max_hides_per_room, scaled, floor_cells_available))

    # ── Public API ──────────────────────────────────────────────────────────────
    def generate(self) -> "np.ndarray":
        """Generate and return a copy of the grid."""
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.rooms = []

        self._place_rooms()
        self._connect_rooms()
        self._place_special_tiles()
        self._validate_connectivity()
        self._compute_metadata()

        return self.grid.copy()

    def render_ascii(self) -> str:
        if self.grid is None:
            return "<not generated>"
        rows = []
        for row in self.grid:
            rows.append("".join(TILE_CHAR[int(t)] for t in row))
        return chr(10).join(rows)

    def print_map(self):
        print(self.render_ascii())
        m = self.metadata
        print(
            f"  seed={m.get('seed')}  alpha={m.get('alpha'):+.2f}  "
            f"rooms={m.get('n_rooms')}  "
            f"vents={m.get('vent_ratio', 0):.3f}  "
            f"hides={m.get('hide_number', 0):.3f}"
        )
        print(
            f"  dist P->exit={m.get('dist_player_exit')}  "
            f"dist A->exit={m.get('dist_alien_exit')}  "
            f"dist A->P={m.get('dist_alien_player')}"
        )

    def to_dict(self) -> dict:
        return {"grid": self.grid.tolist(), "metadata": self.metadata}

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "MapGenerator":
        with open(path) as f:
            data = json.load(f)
        m = data["metadata"]
        gen = cls(width=m["width"], height=m["height"],
                  alpha=m["alpha"], seed=m["seed"],
                  max_hides_per_room=m.get("hide_max_per_room", 3))
        gen.grid = np.array(data["grid"], dtype=np.int8)
        gen.metadata = m
        gen.player_pos = tuple(m["player_start"])
        gen.alien_pos  = tuple(m["alien_start"])
        gen.exit_pos   = tuple(m["exit_pos"])
        return gen

    # ── Room placement ──────────────────────────────────────────────────────────
    def _place_rooms(self):
        attempts = 0
        while len(self.rooms) < self.max_rooms and attempts < 400:
            attempts += 1
            w = self.rng.randint(self.min_room_size, self.max_room_size)
            h = self.rng.randint(self.min_room_size, self.max_room_size)
            x = self.rng.randint(1, self.width  - w - 1)
            y = self.rng.randint(1, self.height - h - 1)
            if not self._overlaps(x, y, w, h):
                self.grid[y:y+h, x:x+w] = Tile.FLOOR
                self.rooms.append((x, y, w, h))

    def _overlaps(self, x, y, w, h) -> bool:
        pad = 1
        for (rx, ry, rw, rh) in self.rooms:
            if (x < rx + rw + pad and x + w + pad > rx and
                    y < ry + rh + pad and y + h + pad > ry):
                return True
        return False

    def _room_centre(self, room) -> tuple:
        x, y, w, h = room
        return (x + w // 2, y + h // 2)

    # ── Corridor carving ────────────────────────────────────────────────────────
    def _connect_rooms(self):
        if len(self.rooms) < 2:
            return
        centres = [self._room_centre(r) for r in self.rooms]
        for i in range(1, len(centres)):
            x1, y1 = centres[i - 1]
            x2, y2 = centres[i]
            if self.rng.random() < 0.5:
                self._h_corridor(y1, x1, x2)
                self._v_corridor(x2, y1, y2)
            else:
                self._v_corridor(x1, y1, y2)
                self._h_corridor(y2, x1, x2)

    def _h_corridor(self, y, x1, x2):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < x < self.width - 1 and 0 < y < self.height - 1:
                if self.grid[y, x] == Tile.WALL:
                    self.grid[y, x] = Tile.FLOOR

    def _v_corridor(self, x, y1, y2):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < x < self.width - 1 and 0 < y < self.height - 1:
                if self.grid[y, x] == Tile.WALL:
                    self.grid[y, x] = Tile.FLOOR

    # ── Special tile placement ──────────────────────────────────────────────────
    def _place_special_tiles(self):
        if len(self.rooms) < 3:
            raise RuntimeError("Too few rooms to place all special tiles.")

        # Place spawns and exit at room centres, maximising distance
        centres = [self._room_centre(r) for r in self.rooms]

        px, py = centres[0]
        ax, ay = centres[-1]
        mid    = len(centres) // 2
        ex, ey = centres[mid]

        self.grid[py, px] = Tile.PLAYER_START;  self.player_pos = (px, py)
        self.grid[ay, ax] = Tile.ALIEN_START;   self.alien_pos  = (ax, ay)
        self.grid[ey, ex] = Tile.EXIT;          self.exit_pos   = (ex, ey)

        # Track special tile positions to avoid overwriting them
        special_tiles = {(px, py), (ax, ay), (ex, ey)}

        # Vents

        for room in self.rooms:
            if self.rng.random() >= self.vent_probability:
                continue
            x, y, w, h = room
            vx = self.rng.randint(x, x + w - 1)
            vy = self.rng.randint(y, y + h - 1)
            # Don't overwrite special tiles (spawns, exit)
            if (vx, vy) not in special_tiles:
                self.grid[vy, vx] = Tile.VENT

        # Hiding spots: sample per-room count from alpha-dependent distribution.
        for room in self.rooms:
            x, y, w, h = room
            # Only place hides on the room border so they stay accessible but do not
            # occupy the room interior or interfere with corridor flow.
            room_border_floor = []
            for ry in range(y, y + h):
                for rx in range(x, x + w):
                    if self.grid[ry, rx] != Tile.FLOOR:
                        continue
                    if rx not in {x, x + w - 1} and ry not in {y, y + h - 1}:
                        continue
                    room_border_floor.append((ry, rx))

            if not room_border_floor:
                continue

            room_max_hides = self._room_hide_max(room, len(room_border_floor))
            hide_counts = list(range(room_max_hides + 1))
            hide_probs = self.hide_count_distribution(room_max_hides)
            n_hide_room = int(self.np_rng.choice(hide_counts, p=hide_probs))
            if n_hide_room <= 0:
                continue

            for hy, hx in self.rng.sample(room_border_floor, min(n_hide_room, len(room_border_floor))):
                # Don't overwrite special tiles (spawns, exit)
                if (hx, hy) not in special_tiles:
                    self.grid[hy, hx] = Tile.HIDE

    # ── Connectivity validation ─────────────────────────────────────────────────
    def _validate_connectivity(self):
        passable = {Tile.FLOOR, Tile.VENT, Tile.HIDE, Tile.PLAYER_START, Tile.ALIEN_START, Tile.EXIT}
        reachable = self._bfs_reachable(self.player_pos, passable)

        for target, name in [(self.exit_pos, "exit"), (self.alien_pos, "alien start")]:
            if target not in reachable:
                print(f"[WARNING] {name} not reachable — carving fallback corridor.")
                self._h_corridor(self.player_pos[1], self.player_pos[0], target[0])
                self._v_corridor(target[0], self.player_pos[1], target[1])

    def _bfs_reachable(self, start, passable) -> set:
        visited = {start}
        queue   = deque([start])
        dirs    = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] in passable:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    def _bfs_distance(self, start, goal, passable) -> int | None:
        if start is None or goal is None:
            return None
        visited = {start}
        queue   = deque([(start, 0)])
        dirs    = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while queue:
            (cx, cy), d = queue.popleft()
            if (cx, cy) == goal:
                return d
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] in passable:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), d + 1))
        return None
            
    # ── Metadata ────────────────────────────────────────────────────────────────
    def _compute_metadata(self):
        total = self.width * self.height
        uniq, cnts = np.unique(self.grid, return_counts=True)
        counts = {int(k): int(v) for k, v in zip(uniq, cnts)}

        passable = {Tile.FLOOR, Tile.VENT, Tile.HIDE, Tile.PLAYER_START, Tile.ALIEN_START, Tile.EXIT}
        d_pe = self._bfs_distance(self.player_pos, self.exit_pos,  passable)
        d_ae = self._bfs_distance(self.alien_pos,  self.exit_pos,  passable)
        d_ap = self._bfs_distance(self.alien_pos,  self.player_pos, passable)

        open_tiles = sum(counts.get(t, 0) for t in passable)

        self.metadata = {
            "seed":              self.seed,
            "alpha":             self.alpha,
            "width":             self.width,
            "height":            self.height,
            "n_rooms":           len(self.rooms),
            "hide_max_per_room": self.max_hides_per_room,
            "hide_distribution":  [round(p, 4) for p in self.hide_count_distribution(self.max_hides_per_room)],
            "hide_room_max_mode": "scaled_by_room_area",
            "tile_counts":       {TILE_NAME[k]: v for k, v in counts.items() if k in TILE_NAME},
            "open_ratio":        round(open_tiles / total, 4),
            "vent_ratio":        round(counts.get(Tile.VENT, 0) / len(self.rooms), 4),
            "hide_number":        counts.get(Tile.HIDE, 0),
            "player_start":      list(self.player_pos),
            "alien_start":       list(self.alien_pos),
            "exit_pos":          list(self.exit_pos),
            "dist_player_exit":  d_pe,
            "dist_alien_exit":   d_ae,
            "dist_alien_player": d_ap,
            "computed_alpha":    round(
                (counts.get(Tile.VENT, 0) - counts.get(Tile.HIDE, 0)) / max(total, 1), 4
            ),
        }


# ── MapPool ────────────────────────────────────────────────────────────────────
class MapPool:
    """
    Batch-generates maps across an alpha range.
    Ideal for curriculum learning or evaluation suites.
    """

    def __init__(
        self,
        n_maps: int = 10,
        width: int = 60,
        height: int = 40,
        alpha_range: tuple = (-1.0, 1.0),
        **kwargs,
    ):
        self.n_maps = n_maps
        self.width = width
        self.height = height
        self.alpha_range = alpha_range
        self.kwargs = kwargs
        self.maps: list[MapGenerator] = []

    def generate_all(self, seed_offset: int = 0) -> list:
        self.maps = []
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], self.n_maps)
        for i, alpha in enumerate(alphas):
            gen = MapGenerator(
                width=self.width, height=self.height,
                alpha=float(alpha), seed=seed_offset + i,
                **self.kwargs,
            )
            gen.generate()
            self.maps.append(gen)
        return self.maps

    def save_all(self, folder: str = "maps"):
        os.makedirs(folder, exist_ok=True)
        for i, gen in enumerate(self.maps):
            name = f"map_{i:03d}_alpha{gen.alpha:+.2f}.json"
            gen.save(os.path.join(folder, name))

    def summary(self) -> list:
        return [g.metadata for g in self.maps]


# ── Pygame visualiser ──────────────────────────────────────────────────────────
def visualise_pygame(gen: MapGenerator, cell: int = 24):
    """
    Open a Pygame window showing the map.
    Press ESC or close the window to exit.
    Requires: pip install pygame
    """
    try:
        import pygame
    except ImportError:
        print("pygame not installed — run: pip install pygame")
        return

    COLORS = {
        Tile.WALL:         ( 20,  20,  30),
        Tile.FLOOR:        ( 55,  55,  75),
        Tile.VENT:         (170,  90, 210),
        Tile.HIDE:         ( 60, 160,  80),
        Tile.PLAYER_START: ( 50, 150, 230),
        Tile.ALIEN_START:  (210,  50,  50),
        Tile.EXIT:         (230, 190,  40),
    }
    LABELS = {
        Tile.VENT:         "V",
        Tile.HIDE:         "H",
        Tile.PLAYER_START: "P",
        Tile.ALIEN_START:  "A",
        Tile.EXIT:         "E",
    }

    W, H = gen.width * cell, gen.height * cell
    pygame.init()
    font = pygame.font.SysFont(None, int(cell * 0.65))
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        f"Map  seed={gen.seed}  alpha={gen.alpha:+.2f}  |  ESC to quit"
    )
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill((0, 0, 0))
        for y in range(gen.height):
            for x in range(gen.width):
                tile  = int(gen.grid[y, x])
                color = COLORS.get(tile, (255, 0, 255))
                rect  = pygame.Rect(x * cell, y * cell, cell - 1, cell - 1)
                pygame.draw.rect(screen, color, rect)

                if tile in LABELS:
                    surf = font.render(LABELS[tile], True, (255, 255, 255))
                    screen.blit(surf, (x * cell + 4, y * cell + 3))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ── Demo ───────────────────────────────────────────────────────────────────────
def run_demo(seed: int | None = None):
    if seed is None:
        seed = random.randint(0, 2**31)

    print("=" * 70)
    print(f"  BALANCED MAP  (alpha = 0.0, seed = {seed})")
    print("=" * 70)
    g0 = MapGenerator(width=42, height=20, alpha=0.0, seed=seed)
    g0.generate()
    g0.print_map()

    print()
    print("=" * 70)
    print(f"  ALIEN-FAVOURED  (alpha = +0.8, seed = {seed})  — more vents")
    print("=" * 70)
    g_a = MapGenerator(width=42, height=20, alpha=0.8, seed=seed)
    g_a.generate()
    g_a.print_map()

    print()
    print("=" * 70)
    print(f"  PLAYER-FAVOURED  (alpha = -0.8, seed = {seed})  — more hiding spots")
    print("=" * 70)
    g_p = MapGenerator(width=42, height=20, alpha=-0.8, seed=seed)
    g_p.generate()
    g_p.print_map()

    print()
    print("=" * 70)
    print("  MAP POOL  (5 maps across alpha range)")
    print("=" * 70)
    pool = MapPool(n_maps=5, width=30, height=18, alpha_range=(-1.0, 1.0))
    pool.generate_all(seed_offset=seed)
    for m in pool.summary():
        print(
            f"  alpha={m['alpha']:+.2f}  "
            f"vents={m['vent_ratio']:.3f}  "
            f"hides={m['hide_number']:.3f}  "
            f"P->exit={m['dist_player_exit']}  "
            f"A->P={m['dist_alien_player']}"
        )

    g0.save(f"maps/demo_balanced_seed_{seed}.json")
    print()
    print(f"Saved maps/demo_balanced_seed_{seed}.json")

    # Uncomment to open the Pygame window:
    # visualise_pygame(g0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alien Isolation-inspired map generator")
    parser.add_argument(
        "seed",
        nargs="?",
        type=int,
        default=None,
        help="Optional integer seed. Different seeds produce different maps.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="Map width in cells (default: 32)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=20,
        help="Map height in cells (default: 20)",
    )
    args = parser.parse_args()
    
    # Override demo to use specified dimensions
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    print()
    print("=" * 70)
    print(f"  BALANCED  (alpha = 0.0, seed = {seed})  [{args.width}x{args.height}]")
    print("=" * 70)
    g0 = MapGenerator(width=args.width, height=args.height, alpha=0.0, seed=seed)
    g0.generate()
    g0.print_map()

    print()
    print("=" * 70)
    print(f"  ALIEN-FAVOURED  (alpha = +0.8, seed = {seed})  [{args.width}x{args.height}]")
    print("=" * 70)
    g_a = MapGenerator(width=args.width, height=args.height, alpha=0.8, seed=seed)
    g_a.generate()
    g_a.print_map()

    print()
    print("=" * 70)
    print(f"  PLAYER-FAVOURED  (alpha = -0.8, seed = {seed})  [{args.width}x{args.height}]")
    print("=" * 70)
    g_p = MapGenerator(width=args.width, height=args.height, alpha=-0.8, seed=seed)
    g_p.generate()
    g_p.print_map()

    g0.save(f"maps/demo_balanced_seed_{seed}.json")
    print()
    print(f"Saved maps/demo_balanced_seed_{seed}.json")
