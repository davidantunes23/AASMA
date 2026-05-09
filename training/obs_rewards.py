import numpy as np
from agents.alien import AlienState
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

    return np.array(pos + state_oh + heard + list(belief_flat)
                    + [known_ratio] + exit_delta, dtype=np.float32)


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
    known_ratio = float((human_agent._known_map != HumanAgent.UNKNOWN).mean()) if human_agent._known_map is not None else 0.0

    made_noise = [1.0 if game.last_noise_ripple is not None else 0.0]

    return np.array(pos + hidden + radar_oh + [steps_since_radar_norm]
                    + exit_delta + exit_known + [n_hiding, known_ratio]
                    + made_noise, dtype=np.float32)


def compute_alien_reward(game, prev_dist, curr_dist, outcome, step, max_steps, g_coef, prev_alien_pos=None):
    r = 0.0
    r += -0.001

    if prev_alien_pos is not None and game.alien_agent.pos == prev_alien_pos:
        r -= 0.05

    try:
        if game.alien_agent.state.name == "HUNT":
            r += 0.5
        elif game.alien_agent.state.name == "INVESTIGATE":
            r += 0.1
    except Exception:
        pass

    if game.last_noise_ripple is not None:
        r += 0.2

    dist_delta = prev_dist - curr_dist
    r += g_coef * 0.3 * dist_delta

    if outcome == "alien_caught_human":
        r += 10.0
    elif outcome == "human_reached_exit":
        r += -2.0
    elif outcome == "max_steps_reached":
        r += -0.5

    return r


def compute_player_reward(game, human_agent, prev_exit_dist, curr_exit_dist,
                           outcome, step, max_steps, g_coef, first_hide_this_episode):
    r = 0.0
    r += 0.01

    if game.last_radar_threat == "CRITICAL" and not human_agent.hidden:
        r += -0.2

    if game.last_noise_ripple is not None:
        r += -0.05

    if human_agent._known_exit is not None and prev_exit_dist is not None and curr_exit_dist is not None:
        exit_delta = prev_exit_dist - curr_exit_dist
        r += g_coef * 0.05 * exit_delta

    if human_agent.hidden and first_hide_this_episode:
        r += g_coef * 0.2
        first_hide_this_episode = False

    if outcome == "human_reached_exit":
        r += 5.0
    elif outcome == "alien_caught_human":
        r += -5.0
    elif outcome == "max_steps_reached":
        r += 0.0

    return r, first_hide_this_episode


def get_g_coef(total_steps_done, decay_steps=300_000):
    return max(0.0, 1.0 - total_steps_done / decay_steps)
