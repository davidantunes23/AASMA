#!/usr/bin/env python3
"""Simulate a game episode using PPO checkpoints.

This runs the alien model against the player model on the fixed alpha=0
seed=42 map and saves the same style of animation as simulate_game.py.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from agents.alien import AlienAgent, compute_fov
from agents.human import Direction as HumanDirection
from agents.human import HumanAgent
from game import Game
from map_generator import MapGenerator, Tile
from simulate_game import FrameState, find_tile_pos, visualize, visualize_world_only
from training.envs import AlienEnv, PlayerEnv

@dataclass
class WarmupModels:
    alien_model: PPO
    player_model: PPO


def make_fixed_map(seed: int = 42, width: int = 60, height: int = 40, alpha: float = 0.0) -> np.ndarray:
    generator = MapGenerator(width=width, height=height, alpha=alpha, seed=seed)
    return generator.generate()


def resolve_checkpoint_path(path_value: str) -> str:
    path = Path(path_value)
    candidates = [
        path,
        Path.cwd() / path,
        ROOT / path,
        ROOT / "models" / path.name,
        ROOT / "models" / path_value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Checkpoint not found: {path_value}")


def load_models(alien_path: str, player_path: str, fixed_map: np.ndarray, max_steps: int):
    alien_env = AlienEnv(fixed_map, max_steps=max_steps, opponent_model=None)
    player_env = PlayerEnv(fixed_map, max_steps=max_steps, opponent_model=None)
    alien_model = PPO.load(resolve_checkpoint_path(alien_path), env=alien_env, device="cpu")
    player_model = PPO.load(resolve_checkpoint_path(player_path), env=player_env, device="cpu")
    return WarmupModels(alien_model=alien_model, player_model=player_model)


def build_initial_game(grid: np.ndarray, view_length: int) -> Game:
    human_start = find_tile_pos(grid, Tile.PLAYER_START)
    alien_start = find_tile_pos(grid, Tile.ALIEN_START)

    human_agent = HumanAgent(start_pos=human_start, start_dir=HumanDirection.NORTH)
    alien_agent = AlienAgent(grid=grid.copy(), start_pos=(alien_start[1], alien_start[0]))

    return Game(
        map=grid.copy(),
        human_agent=human_agent,
        alien_agent=alien_agent,
        human_view_length=view_length,
    )


def record_frame(game: Game, step: int) -> FrameState:
    known = game.human_agent._known_map.copy()
    alien_knowledge = game.alien_agent.knowledge.get_copy()
    human_hidden = game.human_agent._is_hidden()
    human_obs = game._human_cone_observation()
    alien_pos_yx = game.alien_pos
    human_sees_alien = False
    if 0 <= alien_pos_yx[0] < human_obs.shape[0] and 0 <= alien_pos_yx[1] < human_obs.shape[1]:
        human_sees_alien = human_obs[alien_pos_yx] != Game.UNSEEN_TILE
    alien_pos_xy = (alien_pos_yx[1], alien_pos_yx[0])
    human_pos_xy = (game.human_pos[1], game.human_pos[0])
    alien_fov = compute_fov(game.alien_agent.grid, alien_pos_xy, game.alien_agent.fov_radius)
    alien_sees_human = human_pos_xy in alien_fov and not human_hidden

    return FrameState(
        step=step,
        human_pos=game.human_pos,
        human_hidden=human_hidden,
        alien_pos=game.alien_pos,
        known_map=known,
        alien_belief=alien_knowledge,
        human_sees_alien=human_sees_alien,
        alien_sees_human=alien_sees_human,
        radar_threat=getattr(game, "last_radar_threat", None),
        radar_dist=getattr(game, "last_radar_dist", None),
        noise_ripple_pos=game.last_noise_ripple,
        alien_heard_pos=game.alien_agent.last_heard_pos,
        alien_pursuing=game.alien_agent.last_heard_pos is not None and game.alien_agent.steps_since_heard <= 5,
    )


def run_warmup_episode(alien_model: PPO, player_model: PPO, grid: np.ndarray, max_steps: int, view_length: int):
    env = AlienEnv(grid, max_steps=max_steps, opponent_model=player_model)
    obs, _ = env.reset(seed=42)
    frames: list[FrameState] = []
    outcome = "ongoing"

    for step in range(max_steps + 1):
        frames.append(record_frame(env.game, step))
        if env.game.human_pos == env.game.alien_pos:
            outcome = "alien_caught_human"
            break
        exit_pos = find_tile_pos(grid, Tile.EXIT)
        if env.game.human_pos == exit_pos:
            outcome = "human_reached_exit"
            break
        if step == max_steps:
            outcome = "max_steps_reached"
            break

        action, _ = alien_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        outcome = info.get("outcome", outcome)
        if terminated or truncated:
            frames.append(record_frame(env.game, step + 1))
            break

    return frames, outcome


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate any PPO checkpoint pair on the fixed map")
    parser.add_argument("--alien-model", type=str, default="models/alien_warmup.zip")
    parser.add_argument("--player-model", type=str, default="models/player_warmup.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-width", type=int, default=60)
    parser.add_argument("--map-height", type=int, default=40)
    parser.add_argument("--map-alpha", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--view-length", type=int, default=6)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output", type=str, default="output/warmup_simulation.gif")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--style", type=str, default="full", choices=["full", "world"])
    return parser.parse_args()


def main():
    args = parse_args()
    map_seed = args.seed
    print(f"Using map seed: {map_seed}")

    grid = make_fixed_map(seed=map_seed, width=args.map_width, height=args.map_height, alpha=args.map_alpha)
    models = load_models(args.alien_model, args.player_model, grid, args.max_steps)
    frames, outcome = run_warmup_episode(models.alien_model, models.player_model, grid, args.max_steps, args.view_length)
    print(f"Simulation finished in {len(frames) - 1} steps with outcome: {outcome}")

    if args.style == "world":
        world_output = args.output.replace(".gif", "_world.gif")
        visualize_world_only(grid=grid, frames=frames, outcome=outcome, output_path=world_output, fps=args.fps, show_window=not args.no_show)
    else:
        visualize(grid=grid, frames=frames, outcome=outcome, output_path=args.output, fps=args.fps, show_window=not args.no_show)


if __name__ == "__main__":
    main()
