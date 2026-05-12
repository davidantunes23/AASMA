"""Training driver for asymmetric AET setup.

Phase 0 warm-up: train alien vs rule-based human, and player vs rule-based alien.
This script creates two independent PPO agents and saves checkpoints.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from map_generator import MapGenerator
from training.envs import AlienEnv, PlayerEnv


def make_fixed_map(seed=42):
    gen = MapGenerator(width=60, height=40, alpha=0.0, seed=seed)
    grid = gen.generate()
    return grid


def main(args):
    os.makedirs("models", exist_ok=True)
    fixed_map = make_fixed_map(seed=42)

    # Alien warm-up vs rule-based human
    alien_env = DummyVecEnv([lambda: AlienEnv(fixed_map, max_steps=500)])
    alien_model = PPO("MlpPolicy", env=alien_env, verbose=1,
                      gamma=0.99, learning_rate=5e-5,
                      ent_coef=0.01, n_steps=400, batch_size=80)
    print("Training alien (warm-up) for", args.alien_steps)
    alien_model.learn(total_timesteps=args.alien_steps)
    alien_model.save("models/alien_warmup.zip")

    # Player warm-up vs rule-based alien
    player_env = DummyVecEnv([lambda: PlayerEnv(fixed_map, max_steps=500)])
    player_model = PPO("MlpPolicy", env=player_env, verbose=1,
                       gamma=0.99, learning_rate=5e-5,
                       ent_coef=0.01, n_steps=400, batch_size=80)
    print("Training player (warm-up) for", args.player_steps)
    player_model.learn(total_timesteps=args.player_steps)
    player_model.save("models/player_warmup.zip")

    print("Phase 0 complete. Models saved in models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alien-steps", type=int, default=50000)
    parser.add_argument("--player-steps", type=int, default=50000)
    args = parser.parse_args()
    main(args)
