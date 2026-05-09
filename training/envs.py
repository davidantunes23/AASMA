import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.human import HumanAgent, Direction
from agents.alien import AlienAgent, PASSABLE_ALIEN
from game import Game


ACTION_MEANING = {
    0: ("WAIT", None),
    1: ("WALK", Direction.NORTH),
    2: ("WALK", Direction.EAST),
    3: ("WALK", Direction.SOUTH),
    4: ("WALK", Direction.WEST),
}

# Walk actions (exclude WAIT)
_WALK_ACTIONS = [1, 2, 3, 4]


class BaseAETEnv(gym.Env):
    """Base environment wrapping the existing game for single-agent control.

    The controlled agent acts with 5 discrete actions: WAIT + 4 cardinal walks.
    The opponent is the rule-based agent from `agents`.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, fixed_map: np.ndarray, max_steps: int = 500, role: str = "alien", opponent_model=None, global_step_offset: int = 0, decay_steps: int = 300_000):
        super().__init__()
        assert role in {"alien", "human"}
        self.role = role
        self.grid = fixed_map.copy()
        self.max_steps = max_steps
        self.opponent_model = opponent_model
        self.global_step_offset = global_step_offset
        self.decay_steps = decay_steps
        self.step_count = 0
        self._rng = np.random.default_rng()

        # Discrete 5 actions
        self.action_space = spaces.Discrete(5)

        # Observation vector padded to 128 floats
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(128,), dtype=np.float32)

        self.human_agent = None
        self.alien_agent = None
        self.game = None
        self.first_hide = True
        self._prev_dist = None
        self._prev_exit_dist = None
        self._prev_alien_pos = None

    def _predict_model_action(self, model, obs: np.ndarray) -> int:
        if hasattr(model, "observation_space") and getattr(model.observation_space, "shape", None):
            target_size = int(np.prod(model.observation_space.shape))
            flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if flat_obs.size < target_size:
                padded = np.zeros((target_size,), dtype=np.float32)
                padded[: flat_obs.size] = flat_obs
                obs = padded
            elif flat_obs.size > target_size:
                obs = flat_obs[:target_size]
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)

    def _filter_action(self, action: int) -> int:
        """Map WAIT (0) to a random walk unless it's a valid hide WAIT for the human.

        - If action != 0, return unchanged.
        - If role is human and currently hidden, allow WAIT (0).
        - Otherwise return a random walk from _WALK_ACTIONS.
        The alien never receives WAIT: WAIT will always be remapped to a walk.
        """
        if int(action) != 0:
            return int(action)
        # WAIT action requested
        if self.role == "human" and getattr(self.human_agent, "hidden", False):
            return 0
        # Map WAIT to a random walk
        return int(self._rng.choice(_WALK_ACTIONS))

    def _action_to_human_tuple(self, action: int):
        from agents.human import Action as HumanAction

        act_name, dirv = ACTION_MEANING[int(action)]
        if act_name == "WAIT":
            return (HumanAction.WAIT, self.human_agent.direction)
        return (HumanAction.WALK, dirv)

    def _action_to_alien_step(self, action: int):
        act_name, dirv = ACTION_MEANING[int(action)]
        if act_name == "WAIT":
            return None
        dmap = {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
        }
        return dmap[dirv]

    def _exit_pos(self):
        from map_generator import Tile

        exit_yx = np.argwhere(self.grid == int(Tile.EXIT))
        if len(exit_yx) == 0:
            return None
        y, x = exit_yx[0]
        return (int(y), int(x))

    def _check_outcome(self) -> str:
        exit_pos = self._exit_pos()
        if self.game is None:
            return "ongoing"
        if exit_pos is not None and self.game.human_pos == exit_pos:
            return "human_reached_exit"
        ax, ay = self.alien_agent.pos
        if self.game.human_pos == (ay, ax):
            return "alien_caught_human"
        if self.step_count >= self.max_steps:
            return "max_steps_reached"
        return "ongoing"

    def _curr_dist(self) -> float:
        if self.game is None:
            return 0.0
        hx, hy = self.game.human_pos[1], self.game.human_pos[0]
        ax, ay = self.alien_agent.pos
        return float(abs(hx - ax) + abs(hy - ay))

    def _curr_exit_dist(self) -> float | None:
        exit_pos = self._exit_pos()
        if self.game is None or exit_pos is None:
            return None
        ey, ex = exit_pos
        hy, hx = self.game.human_pos
        return float(abs(hx - ex) + abs(hy - ey))

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else int(np.random.randint(0, 2**31 - 1))
        np.random.seed(rng_seed)
        self._rng = np.random.default_rng(rng_seed)
        from map_generator import Tile

        h_pos = None
        a_pos = None
        H, W = self.grid.shape
        for y in range(H):
            for x in range(W):
                if self.grid[y, x] == int(Tile.PLAYER_START):
                    h_pos = (y, x)
                if self.grid[y, x] == int(Tile.ALIEN_START):
                    a_pos = (x, y)
        if h_pos is None:
            h_pos = (1, 1)
        if a_pos is None:
            a_pos = (1, 1)

        self.human_agent = HumanAgent(start_pos=h_pos, start_dir=Direction.NORTH)
        self.alien_agent = AlienAgent(grid=self.grid, start_pos=a_pos)

        self.game = Game(self.grid, self.human_agent, self.alien_agent)
        self.step_count = 0
        self.first_hide = True
        self._prev_dist = None
        self._prev_exit_dist = None
        self._prev_alien_pos = self.alien_agent.pos

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        # Monkey-patch controlled agent for this single step
        # make a filtered copy of the requested action (applies to controlled agent)
        filtered_action = self._filter_action(action)

        if self.role == "human":
            orig_act = self.human_agent._act

            def _act_override(obs, radar_threat=None, radar_dist=None):
                # Keep the human's memory up to date, then force the RL action.
                orig_act(obs, radar_threat, radar_dist)
                return self._action_to_human_tuple(filtered_action)

            self.human_agent._act = _act_override

            if self.opponent_model is not None:
                model_obs = self._get_alien_obs()
                model_action = self._predict_model_action(self.opponent_model, model_obs)
                model_action = self._filter_action(model_action)
                self.alien_agent._rl_action_override = self._action_to_alien_step(model_action)

        else:  # alien controlled
            orig_act = self.human_agent._act
            self.alien_agent._rl_action_override = self._action_to_alien_step(filtered_action)

            if self.opponent_model is not None:

                def _act_override(obs, radar_threat=None, radar_dist=None):
                    # Update the human's internal memory first, then force the PPO policy.
                    orig_act(obs, radar_threat, radar_dist)
                    model_obs = self._get_player_obs()
                    model_action = self._predict_model_action(self.opponent_model, model_obs)
                    model_action = self._filter_action(model_action)
                    return self._action_to_human_tuple(model_action)

                self.human_agent._act = _act_override

        # Step the game (uses the patched agent behavior)
        prev_alien_pos = self.alien_agent.pos
        self.game._step()

        # restore monkey-patches
        if self.role == "human":
            self.human_agent._act = orig_act
        else:
            if self.opponent_model is not None:
                self.human_agent._act = orig_act
        self.alien_agent._rl_action_override = None

        self.step_count += 1
        outcome = self._check_outcome()
        g_coef = max(0.0, 1.0 - ((self.global_step_offset + self.step_count) / self.decay_steps))

        if self.role == "alien":
            curr_dist = self._curr_dist()
            prev_dist = self._prev_dist if self._prev_dist is not None else curr_dist
            reward = self._compute_alien_reward(prev_dist, curr_dist, outcome, g_coef, prev_alien_pos)
            self._prev_dist = curr_dist
            self._prev_alien_pos = self.alien_agent.pos
        else:
            curr_exit_dist = self._curr_exit_dist()
            prev_exit_dist = self._prev_exit_dist if self._prev_exit_dist is not None else curr_exit_dist
            reward, self.first_hide = self._compute_player_reward(
                prev_exit_dist,
                curr_exit_dist,
                outcome,
                g_coef,
                self.first_hide,
            )
            self._prev_exit_dist = curr_exit_dist

        terminated = outcome in {"alien_caught_human", "human_reached_exit"}
        truncated = outcome == "max_steps_reached"
        obs = self._get_obs()
        info = {"outcome": outcome, "step_count": self.step_count, "role": self.role}
        return obs, reward, terminated, truncated, info

    def _compute_alien_reward(self, prev_dist, curr_dist, outcome, g_coef, prev_alien_pos=None):
        from training.obs_rewards import compute_alien_reward

        return compute_alien_reward(
            self.game,
            prev_dist,
            curr_dist,
            outcome,
            self.step_count,
            self.max_steps,
            g_coef,
            prev_alien_pos=prev_alien_pos,
        )

    def _compute_player_reward(self, prev_exit_dist, curr_exit_dist, outcome, g_coef, first_hide_this_episode):
        from training.obs_rewards import compute_player_reward

        return compute_player_reward(
            self.game,
            self.human_agent,
            prev_exit_dist,
            curr_exit_dist,
            outcome,
            self.step_count,
            self.max_steps,
            g_coef,
            first_hide_this_episode,
        )

    def _get_obs(self):
        from training.obs_rewards import get_alien_obs, get_player_obs
        if self.role == "alien":
            vec = get_alien_obs(self.game, self.alien_agent)
        else:
            vec = get_player_obs(self.game, self.human_agent)
        out = np.zeros((128,), dtype=np.float32)
        out[: len(vec)] = vec
        return out

    def _get_player_obs(self):
        from training.obs_rewards import get_player_obs
        return get_player_obs(self.game, self.human_agent)

    def _get_alien_obs(self):
        from training.obs_rewards import get_alien_obs
        return get_alien_obs(self.game, self.alien_agent)

    def render(self, mode="human"):
        pass


class AlienEnv(BaseAETEnv):
    def __init__(self, fixed_map: np.ndarray, max_steps: int = 500, opponent_model=None, global_step_offset: int = 0, decay_steps: int = 300_000):
        super().__init__(fixed_map, max_steps, role="alien", opponent_model=opponent_model, global_step_offset=global_step_offset, decay_steps=decay_steps)


class PlayerEnv(BaseAETEnv):
    def __init__(self, fixed_map: np.ndarray, max_steps: int = 500, opponent_model=None, global_step_offset: int = 0, decay_steps: int = 300_000):
        super().__init__(fixed_map, max_steps, role="human", opponent_model=opponent_model, global_step_offset=global_step_offset, decay_steps=decay_steps)
