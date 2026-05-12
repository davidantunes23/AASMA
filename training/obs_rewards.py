import numpy as np

from agents.human import HumanAgent


def find_exit(grid):
    from map_generator import Tile
    ey, ex = np.where(grid == int(Tile.EXIT))
    if len(ey) > 0:
        return (int(ex[0]), int(ey[0]))
    return (0, 0)


def _resize_bilinear_2d(array: np.ndarray, out_size: int = 8) -> np.ndarray:
    """Resize a 2D array to out_size x out_size using bilinear interpolation."""
    source = np.asarray(array, dtype=np.float32)
    height, width = source.shape
    if height == out_size and width == out_size:
        return source.copy()

    ys = np.linspace(0.0, height - 1.0, out_size)
    xs = np.linspace(0.0, width - 1.0, out_size)
    out = np.empty((out_size, out_size), dtype=np.float32)

    for i, y in enumerate(ys):
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, height - 1)
        wy = y - y0
        row0 = source[y0]
        row1 = source[y1]
        for j, x in enumerate(xs):
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, width - 1)
            wx = x - x0
            top = row0[x0] * (1.0 - wx) + row0[x1] * wx
            bottom = row1[x0] * (1.0 - wx) + row1[x1] * wx
            out[i, j] = top * (1.0 - wy) + bottom * wy

    return out


def get_alien_obs(game, alien_agent) -> np.ndarray:
    H, W = game.map.shape
    ax, ay = alien_agent.pos          # (x, y)

    pos = [ax / W, ay / H]

    state_oh = [0, 0, 0]
    idx = max(0, alien_agent.state.value - 1)
    if 0 <= idx < 3:
        state_oh[idx] = 1

    if alien_agent.last_heard_pos is not None:
        hx, hy = alien_agent.last_heard_pos
        heard = [hx / W, hy / H, min(alien_agent.steps_since_heard / 10, 1.0)]
    else:
        heard = [-1.0, -1.0, 1.0]

    belief_small = alien_agent.belief.belief
    belief_flat = _resize_bilinear_2d(belief_small, out_size=8).flatten()

    known_ratio = float((alien_agent.knowledge.knowledge != -1).mean())

    exit_pos = find_exit(game.map)
    exit_delta = [(exit_pos[0] - ax) / W, (exit_pos[1] - ay) / H]

    return np.array(
        pos + state_oh + heard + list(belief_flat)
        + [known_ratio] + exit_delta,
        dtype=np.float32
    )


def get_player_obs(game, human_agent) -> np.ndarray:
    H, W = game.map.shape
    py, px = human_agent.position

    pos = [px / W, py / H]

    hidden = [1.0 if human_agent.hidden else 0.0]

    bands = {"CRITICAL": 0, "CLOSE": 1, "NEAR": 2, "FAR": 3}
    radar_oh = [0.0, 0.0, 0.0, 0.0]
    if game.last_radar_threat is not None:
        radar_oh[bands[game.last_radar_threat]] = 1.0
    steps_since_radar_norm = min(game.steps_since_radar / game.radar_interval, 1.0)

    if human_agent._known_exit is not None:
        ey, ex = human_agent._known_exit
        exit_delta = [(ex - px) / W, (ey - py) / H]
        exit_known = [1.0]
    else:
        exit_delta = [0.0, 0.0]
        exit_known = [0.0]

    n_hiding = len(human_agent._get_known_hiding_spots()) / 10.0
    known_ratio = float(
        (human_agent._known_map != HumanAgent.UNKNOWN).mean()
    ) if human_agent._known_map is not None else 0.0

    made_noise = [1.0 if game.last_noise_ripple is not None else 0.0]

    # Downsampled known map: 1.0 = explored, 0.0 = unknown.
    # Gives the policy a 2D spatial signal for where to explore next,
    # analogous to the alien's belief_flat.
    if human_agent._known_map is not None:
        known_binary = (human_agent._known_map != HumanAgent.UNKNOWN).astype(np.float32)
        known_map_small = _resize_bilinear_2d(known_binary, out_size=8).flatten()
    else:
        known_map_small = np.zeros(64, dtype=np.float32)

    return np.array(
        pos + hidden + radar_oh + [steps_since_radar_norm]
        + exit_delta + exit_known + [n_hiding, known_ratio]
        + made_noise + list(known_map_small),
        dtype=np.float32
    )


def compute_alien_reward(
    game,
    prev_dist,
    curr_dist,
    outcome,
    step,
    max_steps,
    g_coef,
    prev_alien_pos=None,
    is_new_cell=False,
    reward_log=None,
):
    """Asymmetric alien reward: focused on catching human (primary win condition).

    Returns: (reward, reward_log_dict)
    reward_log tracks component breakdown: terminal, step_cost, pursuit, exploration
    """
    if reward_log is None:
        reward_log = {
            "terminal": 0.0,
            "step_cost": 0.0,
            "pursuit": 0.0,
            "exploration": 0.0,
            "idle_penalty": 0.0,
        }

    r = 0.0

    # Per-step time cost (tiny, just prevents infinite wandering)
    step_cost = -0.01
    r += step_cost
    reward_log["step_cost"] += step_cost

    if prev_alien_pos is not None and game.alien_agent.pos == prev_alien_pos:
        idle_penalty = -0.02
        r += idle_penalty
        reward_log["idle_penalty"] += idle_penalty

    # Exploration bonus only in first half of episode (early discovery helps, not repeated farming)
    if is_new_cell and step < max_steps * 0.5:
        exploration = 0.02
        r += exploration
        reward_log["exploration"] += exploration

    # Pursuit reward: only when recently heard human (semantically meaningful pursuit)
    # Strong shaping here because it directly correlates with catching
    if game.last_noise_ripple is not None and game.alien_agent.steps_since_heard <= 10:
        dist_delta = prev_dist - curr_dist
        pursuit = 0.3 * dist_delta  # Reward approaching heard human
        r += pursuit
        reward_log["pursuit"] += pursuit

    # Terminal rewards absolutely dominate (no farming possible)
    if outcome == "alien_caught_human":
        terminal = 20.0
        r += terminal
        reward_log["terminal"] += terminal
    elif outcome == "human_reached_exit":
        terminal = -20.0
        r += terminal
        reward_log["terminal"] += terminal
    elif outcome == "max_steps_reached":
        # Slight penalty for timeout (no catch, no prevention)
        terminal = -0.5
        r += terminal
        reward_log["terminal"] += terminal

    return r, reward_log


def compute_player_reward(
    game,
    human_agent,
    prev_exit_dist,
    curr_exit_dist,
    outcome,
    step,
    max_steps,
    g_coef,
    first_hide_this_episode,
    prev_human_pos=None,
    is_new_cell=False,
    reward_log=None,
    prev_known_ratio=0.0,
    curr_known_ratio=0.0,
):
    """Asymmetric human reward: focused on escaping to exit (primary win condition).

    Returns: (reward, first_hide_this_episode, reward_log_dict)
    """
    if reward_log is None:
        reward_log = {
            "terminal": 0.0,
            "step_cost": 0.0,
            "exit_progress": 0.0,
            "exit_proximity": 0.0,
            "hide_bonus": 0.0,
            "danger_penalty": 0.0,
            "discovery": 0.0,
            "exploration": 0.0,
            "no_progress_penalty": 0.0,
            "found_exit_bonus": 0.0,
            "idle_penalty": 0.0,
        }

    r = 0.0

    # Per-step time cost (tiny, just prevents infinite wandering)
    step_cost = -0.01
    r += step_cost
    reward_log["step_cost"] += step_cost

    if prev_human_pos is not None and human_agent.position == prev_human_pos:
        from map_generator import Tile

        py, px = human_agent.position
        is_hiding_spot = game.map[py, px] == int(Tile.HIDE)
        if not is_hiding_spot:
            idle_penalty = -0.02
            r += idle_penalty
            reward_log["idle_penalty"] += idle_penalty

    # Danger penalty: when critically threatened while unhidden (encourages hiding or moving)
    if game.last_radar_threat == "CRITICAL" and not human_agent.hidden:
        danger = -0.2
        r += danger
        reward_log["danger_penalty"] += danger

    # Before exit is known: reward discovery of new map areas
    if human_agent._known_exit is None and curr_known_ratio > prev_known_ratio:
        discovery = 0.15 * (curr_known_ratio - prev_known_ratio)
        r += discovery
        reward_log["discovery"] += discovery

    # Exploration bonus ONLY before exit is known, and only in first half
    if (
        human_agent._known_exit is None
        and is_new_cell
        and step < max_steps * 0.5
    ):
        exploration = 0.03  # slightly smaller, pure exploration is secondary
        r += exploration
        reward_log["exploration"] += exploration

    # One-time exit discovery bonus: first time exit becomes known
    if (
        human_agent._known_exit is not None
        and prev_exit_dist is None
        and curr_exit_dist is not None
    ):
        found_exit_bonus = 3.0
        r += found_exit_bonus
        reward_log["found_exit_bonus"] += found_exit_bonus

    # Exit progress & proximity shaping, only when exit is known
    if (
        human_agent._known_exit is not None
        and prev_exit_dist is not None
        and curr_exit_dist is not None
    ):
        exit_delta = prev_exit_dist - curr_exit_dist

        progress_term = 0.25 * exit_delta

        r += progress_term
        reward_log["exit_progress"] += progress_term

        # Small penalty when making effectively no net progress (discourage oscillations)
        if abs(exit_delta) < 0.5:
            no_progress_penalty = -0.02
            r += no_progress_penalty
            reward_log["no_progress_penalty"] += no_progress_penalty

    # ONE-TIME hide bonus: only first time hiding per episode (not farmable)
    if human_agent.hidden and first_hide_this_episode:
        hide_bonus = 0.5
        r += hide_bonus
        reward_log["hide_bonus"] += hide_bonus
        first_hide_this_episode = False

    # Terminal rewards absolutely dominate (no farming possible)
    if outcome == "human_reached_exit":
        terminal = 30.0
        r += terminal
        reward_log["terminal"] += terminal
    elif outcome == "alien_caught_human":
        terminal = -20.0
        r += terminal
        reward_log["terminal"] += terminal
    elif outcome == "max_steps_reached":
        # Surviving without escaping is slightly bad
        terminal = -1.0
        r += terminal
        reward_log["terminal"] += terminal

    return r, first_hide_this_episode, reward_log


def get_g_coef(total_steps_done, decay_steps=300_000):
    return max(0.0, 1.0 - total_steps_done / decay_steps)