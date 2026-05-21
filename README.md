# AASMA — Asymmetric Agents in a Hide-and-Seek Game

An Alien Isolation-inspired asymmetric pursuit-evasion game with procedurally generated maps, rule-based and random agent baselines, and a flexible simulation framework.

---

## The Game

Two agents compete on a partially-observable grid map:

| Agent             | Goal                                | Abilities                          |
|-------------------|-------------------------------------|------------------------------------|
| **Human** (blue)  | Reach the exit without being caught | Walk, wait, hide in hiding spots   |
| **Alien** (red)   | Catch the human before it escapes   | Walk, wait, teleport through vents |

**Key mechanics:**

- **Partial observation** — the human sees a directional forward cone; the alien has an omnidirectional FOV blocked by walls and hiding spots.
- **Radar** — every few steps the human receives a threat-level ping (CRITICAL / CLOSE / NEAR / FAR) based on path distance to the alien.
- **Noise** — each step the human has a chance to emit a sound at a jittered position; the alien hears this and uses it to track the human.
- **Hiding spots** — the human can enter a hiding spot to break line-of-sight and suppress sound emission.
- **Vents** — the alien can teleport between observed vents when it would save significant distance.

---

## Quick Start

```bash
# Clone and install dependencies
git clone <repo>
cd AASMA
pip install numpy matplotlib

# Run a rule-based game (produces a GIF)
python scripts/run.py --demo rule --no-show

# Run with random agents
python scripts/run.py --demo random --no-show

# Show the world map only (no knowledge panels)
python scripts/run.py --demo rule --style world --no-show
```

Output is saved to `output/simulation.gif` by default.

---

## Directory Structure

```
AASMA/
├── agents/
│   ├── alien.py          — rule-based alien (A*, belief map, FSM, vent routing)
│   ├── human.py          — rule-based human (BFS, hiding, radar-reactive)
│   ├── random_alien.py   — random alien baseline
│   └── random_human.py   — random human baseline
├── map_generator.py      — procedural map generation + PNG visualization
├── simulation.py         — simulation engine (game loop, rendering, GIF output)
├── scripts/
│   └── run.py            — unified CLI entry point
├── training/             — RL training pipeline (WIP)
├── maps/                 — saved map JSON files
└── output/               — generated GIFs and PNGs
```

---

## Running Simulations

All simulation options go through `scripts/run.py`:

```
python scripts/run.py [options]

--demo {rule,random}     Agent pair to use (default: rule)
--style {full,world}     full = world + knowledge panels; world = world only (default: full)
--knowledge {on,off}     Show per-agent knowledge panels (default: on)
--seed N                 Map and agent seed (default: 42)
--width N                Map width in cells (default: 50)
--height N               Map height in cells (default: 35)
--alpha F                Map bias: negative = more hides, positive = more vents (default: 0.0)
--max-steps N            Episode length cap (default: 300)
--fps N                  GIF frames per second (default: 12)
--output PATH            Output GIF path (default: output/simulation.gif)
--no-show                Skip interactive preview window
--human-view N           Human observation radius (default: 6)
--alien-fov N            Alien FOV radius (default: 6)
```

**Examples:**

```bash
# Alien-favoured map, world view only
python scripts/run.py --alpha 0.8 --style world --no-show

# Player-favoured map, longer episode
python scripts/run.py --alpha -0.5 --max-steps 500 --seed 7 --no-show

# Random agents, no knowledge panels
python scripts/run.py --demo random --knowledge off --no-show
```

---

## Map Generator

Maps are procedurally generated with a single `alpha` parameter that shifts the balance:

| alpha | Effect                                          |
|-------|-------------------------------------------------|
| `< 0` | Player-favoured: more hiding spots, fewer vents |
| `= 0` | Balanced                                        |
| `> 0` | Alien-favoured: more vents, fewer hiding spots  |

**Tile types:**

| Tile          | Value | Description                         |
|---------------|-------|-------------------------------------|
| WALL          | 0     | Impassable                          |
| FLOOR         | 1     | Passable by both                    |
| VENT          | 2     | Alien teleport network              |
| HIDE          | 3     | Human hiding spot, blocks alien FOV |
| PLAYER_START  | 4     | Human spawn                         |
| ALIEN_START   | 5     | Alien spawn                         |
| EXIT          | 6     | Human goal                          |

**CLI usage:**

```bash
# Print map to terminal
python map_generator.py [seed] [--width N] [--height N]

# Save PNG visualizations to output/
python map_generator.py [seed] --visualize
```

---

## Agents

### Rule-based Alien (`agents/alien.py`)

A finite-state machine with three states:

- **SEARCH** — explores unknown frontiers using A*, falls back to patrol waypoints.
- **INVESTIGATE** — moves to the last known or last heard position.
- **HUNT** — player is in FOV; pursues at 2 cells/step.

Additional capabilities: Bayesian belief map over player location, auditory evidence tracking (heard position with jitter), strategic vent teleportation when it saves 4+ steps.

### Rule-based Human (`agents/human.py`)

Priority-ordered decision loop:

1. **Stay hidden** — if currently hiding and threat is CRITICAL or CLOSE, wait.
2. **Exit** — BFS path to exit once it is known.
3. **Hide** — seek nearest hiding spot when radar threat is CRITICAL, or CLOSE without a nearby exit.
4. **Explore** — BFS toward unknown frontier tiles.

### Random Baselines (`agents/random_{alien,human}.py`)

Move to a uniformly random passable neighbour each step. The random alien also teleports through vents when standing on one and another is known. Used for sanity checks and as a lower-performance comparison baseline.

---

## Simulation Engine (`simulation.py`)

`GenericMapSimulation` runs the game loop and renders output GIFs. It supports any agent that exposes either a `step(player_pos, heard_pos, step_num)` or `_act(obs, radar_threat, radar_dist)` interface.

**Key parameters:**

```python
GenericMapSimulation(
    grid,
    agents,                  # list of AgentSpec
    knowledge_mode="on",     # "on" records per-agent knowledge maps for rendering
    enable_mechanics=True,   # enables radar, noise, cone observation
    p_noise=0.1,             # probability of sound event per step
    radar_interval=5,        # steps between radar pings
)
```

When `enable_mechanics=False` (used with random agents), all three mechanics are disabled and agents receive exact positions instead of noisy signals.
