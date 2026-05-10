"""Round-based AET trainer for the Alien vs Human setup.

This script builds on the Phase 0 warm-up checkpoints and then alternates
training the alien and human against a frozen opponent sampled from a
historical pool (pFSP-style selection). A lightweight ADA rule adjusts
training budgets based on recent win rates.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from map_generator import MapGenerator
from training.envs import AlienEnv, PlayerEnv


ALIEN_WINS = {"alien_caught_human"}
PLAYER_WINS = {"human_reached_exit"}
PLAYER_SURVIVED = {"max_steps_reached"}


def make_fixed_map(seed: int = 42):
    generator = MapGenerator(width=60, height=40, alpha=0.0, seed=seed)
    return generator.generate()


def make_env(role: str, fixed_map: np.ndarray, opponent_model=None, max_steps: int = 500, global_step_offset: int = 0, decay_steps: int = 300_000):
    if role == "alien":
        return AlienEnv(fixed_map, max_steps=max_steps, opponent_model=opponent_model, global_step_offset=global_step_offset, decay_steps=decay_steps)
    if role == "human":
        return PlayerEnv(fixed_map, max_steps=max_steps, opponent_model=opponent_model, global_step_offset=global_step_offset, decay_steps=decay_steps)
    raise ValueError(f"Unknown role: {role}")


def make_vec_env(role: str, fixed_map: np.ndarray, opponent_model=None, max_steps: int = 500, global_step_offset: int = 0, decay_steps: int = 300_000):
    return DummyVecEnv([lambda om=opponent_model, gso=global_step_offset: make_env(role, fixed_map, opponent_model=om, max_steps=max_steps, global_step_offset=gso, decay_steps=decay_steps)])


def load_or_init_model(model_path: str, env, verbose: int = 0):
    if os.path.exists(model_path):
        return PPO.load(model_path, env=env)
    return PPO(
        "MlpPolicy",
        env=env,
        verbose=verbose,
        gamma=0.99,
        learning_rate=5e-5,
        ent_coef=0.01,
        n_steps=400,
        batch_size=64,
    )


def select_pfsp_opponent(latest_model, history_paths: list[str], rng: random.Random, fixed_map: np.ndarray, role: str, max_steps: int = 500, latest_prob: float = 0.8, global_step_offset: int = 0, decay_steps: int = 300_000):
    if not history_paths or rng.random() < latest_prob:
        return latest_model
    chosen_path = rng.choice(history_paths)
    tmp_env = make_vec_env(role, fixed_map, max_steps=max_steps, global_step_offset=global_step_offset, decay_steps=decay_steps)
    return PPO.load(chosen_path, env=tmp_env)


def ada_steps(base_steps: int, role_win_rate: float, opponent_win_rate: float, lo: float = 0.5, hi: float = 1.5) -> int:
    """More steps when losing, fewer when winning."""
    if role_win_rate < opponent_win_rate:
        gap = opponent_win_rate - role_win_rate
        scale = 1.0 + min(hi - 1.0, gap * 2.0)
    else:
        gap = role_win_rate - opponent_win_rate
        scale = max(lo, 1.0 - gap)
    return max(1, int(round(base_steps * scale)))


def evaluate(role: str, model, opponent_model, fixed_map: np.ndarray, episodes: int = 20, max_steps: int = 500, seed: int = 42, global_step_offset: int = 0, decay_steps: int = 300_000):
    wins = 0
    escaped = 0
    survived = 0
    outcomes: list[str] = []

    for episode in range(episodes):
        env = make_env(role, fixed_map, opponent_model=opponent_model, max_steps=max_steps, global_step_offset=global_step_offset, decay_steps=decay_steps)
        obs, _ = env.reset(seed=seed + episode)
        done = False
        outcome = "ongoing"

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            outcome = info.get("outcome", outcome)

        outcomes.append(outcome)
        if role == "alien":
            wins += int(outcome in ALIEN_WINS)
        else:
            wins += int(outcome in PLAYER_WINS)
            escaped += int(outcome in PLAYER_WINS)
        survived += int(outcome in PLAYER_SURVIVED)

    return wins / max(episodes, 1), outcomes, escaped / max(episodes, 1), survived / max(episodes, 1)


def train_round(role: str, model, opponent_model, fixed_map: np.ndarray, steps: int, max_steps: int, seed: int, global_step_offset: int = 0, decay_steps: int = 300_000):
    env = make_vec_env(role, fixed_map, opponent_model=opponent_model, max_steps=max_steps, global_step_offset=global_step_offset, decay_steps=decay_steps)
    model.set_env(env)
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Round-based AET trainer with pFSP-style opponent selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--base-steps", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--warmup-alien", type=str, default="models/alien_warmup.zip")
    parser.add_argument("--warmup-player", type=str, default="models/player_warmup.zip")
    parser.add_argument("--latest-prob", type=float, default=0.8)
    parser.add_argument("--decay-steps", type=int, default=300_000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fixed_map = make_fixed_map(seed=args.seed)
    rng = random.Random(args.seed)

    alien_checkpoint = args.warmup_alien if os.path.exists(args.warmup_alien) else os.path.join(args.output_dir, "alien_init.zip")
    player_checkpoint = args.warmup_player if os.path.exists(args.warmup_player) else os.path.join(args.output_dir, "player_init.zip")

    if not os.path.exists(alien_checkpoint):
        init_alien_env = make_vec_env("alien", fixed_map, max_steps=args.eval_max_steps)
        init_alien = PPO(
            "MlpPolicy",
            env=init_alien_env,
            verbose=0,
            gamma=0.99,
            learning_rate=5e-5,
            ent_coef=0.01,
            n_steps=400,
            batch_size=64,
        )
        init_alien.save(alien_checkpoint)

    if not os.path.exists(player_checkpoint):
        init_player_env = make_vec_env("human", fixed_map, max_steps=args.eval_max_steps)
        init_player = PPO(
            "MlpPolicy",
            env=init_player_env,
            verbose=0,
            gamma=0.99,
            learning_rate=5e-5,
            ent_coef=0.01,
            n_steps=400,
            batch_size=64,
        )
        init_player.save(player_checkpoint)

    alien_init_env = make_vec_env("alien", fixed_map, max_steps=args.eval_max_steps, decay_steps=args.decay_steps)
    player_init_env = make_vec_env("human", fixed_map, max_steps=args.eval_max_steps, decay_steps=args.decay_steps)
    alien_model = PPO.load(alien_checkpoint, env=alien_init_env)
    player_model = PPO.load(player_checkpoint, env=player_init_env)

    # Phase 0: warm-up (interleaved). If warm-up checkpoint files already exist, skip.
    warmup_present = os.path.exists(args.warmup_alien) and os.path.exists(args.warmup_player)
    total_alien_steps = 0
    total_player_steps = 0
    if args.warmup_steps and not warmup_present:
        print(f"Starting interleaved warm-up: {args.warmup_steps} steps per agent (using eval_max_steps={args.eval_max_steps})")
        steps_done = 0
        chunk = 10_000
        while steps_done < args.warmup_steps:
            this_chunk = min(chunk, args.warmup_steps - steps_done)
            # Alien trains vs rule-based human (opponent_model=None)
            # Use eval_max_steps to match evaluation episodes
            aenv = make_vec_env("alien", fixed_map, opponent_model=None, max_steps=args.eval_max_steps, decay_steps=args.decay_steps)
            alien_model.set_env(aenv)
            alien_model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)

            # Player trains vs rule-based alien (opponent_model=None)
            # Use eval_max_steps to match evaluation episodes
            penv = make_vec_env("human", fixed_map, opponent_model=None, max_steps=args.eval_max_steps, decay_steps=args.decay_steps)
            player_model.set_env(penv)
            player_model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)

            steps_done += this_chunk
            print(f"Warm-up progress: {steps_done}/{args.warmup_steps} per-agent steps")

        # Save warm-up checkpoints
        os.makedirs(args.output_dir, exist_ok=True)
        alien_warmup_path = args.warmup_alien if args.warmup_alien else os.path.join(args.output_dir, "alien_warmup.zip")
        player_warmup_path = args.warmup_player if args.warmup_player else os.path.join(args.output_dir, "player_warmup.zip")
        alien_model.save(alien_warmup_path)
        player_model.save(player_warmup_path)
        print(f"Warm-up complete. Saved: {alien_warmup_path}, {player_warmup_path}")
        total_alien_steps = args.warmup_steps
        total_player_steps = args.warmup_steps
        # point initial checkpoints to warmup artifacts for history
        alien_checkpoint = alien_warmup_path
        player_checkpoint = player_warmup_path
    else:
        # No warm-up performed; totals remain zero or inferred from existing warmup
        if warmup_present:
            total_alien_steps = args.warmup_steps
            total_player_steps = args.warmup_steps
        else:
            total_alien_steps = 0
            total_player_steps = 0
    alien_history = [alien_checkpoint]
    player_history = [player_checkpoint]
    imbalance_streak = 0
    next_round_boost = 1.0

    metrics_path = os.path.join(args.output_dir, "aet_metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for round_idx in range(1, args.rounds + 1):
            boost = next_round_boost
            alien_opponent = select_pfsp_opponent(
                player_model,
                player_history,
                rng,
                fixed_map,
                "human",
                max_steps=args.eval_max_steps,
                latest_prob=args.latest_prob,
                global_step_offset=total_player_steps,
                decay_steps=args.decay_steps,
            )
            player_opponent = select_pfsp_opponent(
                alien_model,
                alien_history,
                rng,
                fixed_map,
                "alien",
                max_steps=args.eval_max_steps,
                latest_prob=args.latest_prob,
                global_step_offset=total_alien_steps,
                decay_steps=args.decay_steps,
            )

            eval_alien_win_rate, _, eval_alien_catch_rate, eval_alien_survive_rate = evaluate(
                "alien",
                alien_model,
                alien_opponent,
                fixed_map,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                seed=args.seed + round_idx * 1000,
                global_step_offset=total_alien_steps,
                decay_steps=args.decay_steps,
            )
            eval_player_win_rate, _, eval_player_escape_rate, eval_player_survive_rate = evaluate(
                "human",
                player_model,
                player_opponent,
                fixed_map,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                seed=args.seed + round_idx * 2000,
                global_step_offset=total_player_steps,
                decay_steps=args.decay_steps,
            )

            alien_steps = int(ada_steps(args.base_steps, eval_alien_win_rate, eval_player_win_rate) * boost)
            player_steps = int(ada_steps(args.base_steps, eval_player_win_rate, eval_alien_win_rate) * boost)

            # Use eval_max_steps for training to match evaluation episodes
            alien_env = make_vec_env("alien", fixed_map, opponent_model=alien_opponent, max_steps=args.eval_max_steps, global_step_offset=total_alien_steps, decay_steps=args.decay_steps)
            alien_model.set_env(alien_env)
            alien_model.learn(total_timesteps=alien_steps, reset_num_timesteps=False)
            total_alien_steps += alien_steps
            alien_path = os.path.join(args.output_dir, f"alien_round_{round_idx:03d}.zip")
            alien_model.save(alien_path)
            alien_history.append(alien_path)

            # Use eval_max_steps for training to match evaluation episodes
            player_env = make_vec_env("human", fixed_map, opponent_model=player_opponent, max_steps=args.eval_max_steps, global_step_offset=total_player_steps, decay_steps=args.decay_steps)
            player_model.set_env(player_env)
            player_model.learn(total_timesteps=player_steps, reset_num_timesteps=False)
            total_player_steps += player_steps
            player_path = os.path.join(args.output_dir, f"player_round_{round_idx:03d}.zip")
            player_model.save(player_path)
            player_history.append(player_path)

            post_alien_win_rate, _, post_alien_catch_rate, post_alien_survive_rate = evaluate(
                "alien",
                alien_model,
                player_model,
                fixed_map,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                seed=args.seed + round_idx * 3000,
                global_step_offset=total_alien_steps,
                decay_steps=args.decay_steps,
            )
            post_player_win_rate, _, post_player_escape_rate, post_player_survive_rate = evaluate(
                "human",
                player_model,
                alien_model,
                fixed_map,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                seed=args.seed + round_idx * 4000,
                global_step_offset=total_player_steps,
                decay_steps=args.decay_steps,
            )

            imbalance = abs(post_alien_win_rate - post_player_win_rate)
            if imbalance > 0.15:
                imbalance_streak += 1
            else:
                imbalance_streak = 0

            if imbalance_streak >= 3:
                next_round_boost = 1.25
                imbalance_streak = 0
                er_note = "ada_boost"
            else:
                next_round_boost = 1.0
                er_note = "inactive"

            record = {
                "round": round_idx,
                "pre_eval_alien_win_rate": eval_alien_win_rate,
                "pre_eval_player_win_rate": eval_player_win_rate,
                "pre_eval_alien_catch_rate": eval_alien_catch_rate,
                "pre_eval_player_escape_rate": eval_player_escape_rate,
                "pre_eval_player_survive_rate": eval_player_survive_rate,
                "post_eval_alien_win_rate": post_alien_win_rate,
                "post_eval_player_win_rate": post_player_win_rate,
                "post_eval_alien_catch_rate": post_alien_catch_rate,
                "post_eval_alien_survive_rate": post_alien_survive_rate,
                "post_eval_player_escape_rate": post_player_escape_rate,
                "post_eval_player_survive_rate": post_player_survive_rate,
                "alien_steps": alien_steps,
                "player_steps": player_steps,
                "imbalance": imbalance,
                "er_note": er_note,
                "applied_boost": boost,
                "next_round_boost": next_round_boost,
                "alien_checkpoint": alien_path,
                "player_checkpoint": player_path,
            }
            metrics_file.write(f"{record}\n")
            metrics_file.flush()
            print(record)

    print(f"AET loop complete. Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
