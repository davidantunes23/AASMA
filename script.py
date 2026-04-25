
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import numpy as np
import json, os, sys
import random

# ── make sure map_generator is importable ──────────────────────────────────────
sys.path.insert(0, ".")
from map_generator import (
    MapGenerator, MapPool, Tile,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
TILE_COLORS = {
    Tile.WALL:         "#1a1a2e",   # deep navy
    Tile.FLOOR:        "#2e2e4a",   # dark slate
    Tile.VENT:         "#9b59b6",   # purple
    Tile.HIDE:         "#27ae60",   # green
    Tile.PLAYER_START: "#2980b9",   # blue
    Tile.ALIEN_START:  "#c0392b",   # red
    Tile.EXIT:         "#f39c12",   # amber
}

TILE_LABELS = {
    Tile.WALL:         "Wall",
    Tile.FLOOR:        "Floor",
    Tile.VENT:         "Vent (alien shortcut)",
    Tile.HIDE:         "Hiding Spot",
    Tile.PLAYER_START: "Player Start",
    Tile.ALIEN_START:  "Alien Start",
    Tile.EXIT:         "Exit",
}

TILE_SYMBOLS = {
    Tile.VENT:         "V",
    Tile.HIDE:         "H",
    Tile.PLAYER_START: "P",
    Tile.ALIEN_START:  "A",
    Tile.EXIT:         "E",
}

# ── Core visualiser ────────────────────────────────────────────────────────────
def visualise_map(
    gen: MapGenerator,
    title: str | None = None,
    save_path: str | None = None,
    cell_size: float = 0.55,
    show_grid: bool = True,
    show_symbols: bool = True,
    show_distances: bool = True,
    ax=None,
):
    """
    Render a MapGenerator grid as a colour-coded matplotlib figure.
    
    Parameters
    ----------
    gen          : MapGenerator with .grid populated
    title        : figure title (auto-generated if None)
    save_path    : if given, saves PNG to this path
    cell_size    : inches per cell (controls overall figure size)
    show_grid    : draw thin grid lines between cells
    show_symbols : overlay letters (P, A, E, V, H) on special tiles
    show_distances: show BFS distances in the subtitle
    ax           : if provided, draw into this existing Axes
    """
    grid = gen.grid
    H, W = grid.shape

    # Build RGB image
    rgb = np.zeros((H, W, 3))
    for tile_id, hex_color in TILE_COLORS.items():
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        mask = grid == tile_id
        rgb[mask] = [r/255, g/255, b/255]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(W * cell_size, H * cell_size),
            facecolor="#0d0d1a",
        )

    ax.imshow(rgb, interpolation="nearest", aspect="equal")

    # Grid lines
    if show_grid:
        for x in range(W + 1):
            ax.axvline(x - 0.5, color="#0d0d1a", lw=0.4, alpha=0.7)
        for y in range(H + 1):
            ax.axhline(y - 0.5, color="#0d0d1a", lw=0.4, alpha=0.7)

    # Symbols
    if show_symbols:
        sym_fontsize = max(5, min(10, int(cell_size * 14)))
        for tile_id, symbol in TILE_SYMBOLS.items():
            ys, xs = np.where(grid == tile_id)
            for x, y in zip(xs, ys):
                ax.text(
                    x, y, symbol,
                    ha="center", va="center",
                    fontsize=sym_fontsize, fontweight="bold",
                    color="white", alpha=0.92,
                )

    # Title
    m = gen.metadata
    alpha_sign = "+" if gen.alpha >= 0 else ""
    auto_title = (
        f"Map  |  seed={gen.seed}  "
        f"alpha={alpha_sign}{gen.alpha:.2f}  "
        f"rooms={m.get('n_rooms', '?')}  "
        f"{W}×{H}"
    )
    if show_distances:
        sub = (
            f"P→exit: {m.get('dist_player_exit')} steps   "
            f"A→exit: {m.get('dist_alien_exit')} steps   "
            f"A→P: {m.get('dist_alien_player')} steps   "
            f"vents: {m.get('vent_ratio',0):.1%}   "
            f"hides: {m.get('hide_number',0)}"
        )
        ax.set_title(
            (title or auto_title) + "\n" + sub,
            color="white", fontsize=9, pad=8, linespacing=1.6,
        )
    else:
        ax.set_title(title or auto_title, color="white", fontsize=10, pad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.5)
    ax.set_facecolor("#0d0d1a")

    # Legend
    patches = [
        mpatches.Patch(color=TILE_COLORS[t], label=TILE_LABELS[t])
        for t in [Tile.WALL, Tile.FLOOR, Tile.VENT, Tile.HIDE, Tile.PLAYER_START, Tile.ALIEN_START, Tile.EXIT]
    ]
    ax.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        framealpha=0.15,
        facecolor="#1a1a2e",
        edgecolor="#444",
        fontsize=7.5,
        labelcolor="white",
        handlelength=1.2,
        handleheight=0.9,
        borderpad=0.7,
        columnspacing=1.0,
    )

    if standalone:
        fig.patch.set_facecolor("#0d0d1a")
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#0d0d1a")
            print(f"Saved -> {save_path}")
        return fig, ax

    return None, ax


# ── Comparison grid (multiple alphas side by side) ─────────────────────────────
def visualise_alpha_comparison(
    alphas: list[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    seed: int = 42,
    width: int = 32,
    height: int = 18,
    save_path: str = "output/alpha_comparison.png",
    cell_size: float = 0.38,
):
    """
    Generate one map per alpha value and render them in a single figure row.
    """
    n = len(alphas)
    fig, axes = plt.subplots(
        1, n,
        figsize=(width * cell_size * n * 0.52, height * cell_size * 1.55),
        facecolor="#0d0d1a",
    )
    if n == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        gen = MapGenerator(width=width, height=height, alpha=alpha, seed=seed)
        gen.generate()
        sign = "+" if alpha >= 0 else ""
        visualise_map(
            gen,
            title=f"alpha = {sign}{alpha:.1f}",
            show_distances=False,
            cell_size=cell_size,
            ax=ax,
        )
        ax.set_title(
            f"alpha = {sign}{alpha:.1f}\n"
            f"vents {gen.metadata['vent_ratio']:.1%}  "
            f"hides {gen.metadata['hide_number']}",
            color="white", fontsize=8, pad=5,
        )

    # shared legend below
    patches = [
        mpatches.Patch(color=TILE_COLORS[t], label=TILE_LABELS[t])
        for t in [Tile.WALL, Tile.FLOOR, Tile.VENT, Tile.HIDE, Tile.PLAYER_START, Tile.ALIEN_START, Tile.EXIT]
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=7,
        framealpha=0.15,
        facecolor="#1a1a2e",
        edgecolor="#555",
        fontsize=7.5,
        labelcolor="white",
        handlelength=1.2,
        borderpad=0.7,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Alien Isolation Map Generator  —  Alpha Advantage Comparison",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"Saved -> {save_path}")
    return fig


def visualise_current_maps_comparison(
    maps: list[MapGenerator],
    labels: list[str] | None = None,
    save_path: str = "output/current_maps_comparison.png",
    cell_size: float = 0.38,
):
    """Render an already-generated set of maps side by side."""
    if not maps:
        raise ValueError("maps must not be empty")

    n = len(maps)
    width = maps[0].width
    height = maps[0].height
    fig, axes = plt.subplots(
        1, n,
        figsize=(width * cell_size * n * 0.52, height * cell_size * 1.55),
        facecolor="#0d0d1a",
    )
    if n == 1:
        axes = [axes]

    labels = labels or [f"map {i + 1}" for i in range(n)]
    for ax, gen, label in zip(axes, maps, labels):
        visualise_map(
            gen,
            title=label,
            show_distances=False,
            cell_size=cell_size,
            ax=ax,
        )
        ax.set_title(
            f"{label}\n"
            f"vents {gen.metadata['vent_ratio']:.1%}  "
            f"hides {gen.metadata['hide_number']}",
            color="white", fontsize=8, pad=5,
        )

    patches = [
        mpatches.Patch(color=TILE_COLORS[t], label=TILE_LABELS[t])
        for t in [Tile.WALL, Tile.FLOOR, Tile.VENT, Tile.HIDE, Tile.PLAYER_START, Tile.ALIEN_START, Tile.EXIT]
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=7,
        framealpha=0.15,
        facecolor="#1a1a2e",
        edgecolor="#555",
        fontsize=7.5,
        labelcolor="white",
        handlelength=1.2,
        borderpad=0.7,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Current Map Comparison",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"Saved -> {save_path}")
    return fig


# ── Run demos ──────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)


def render_saved_map(json_path: str, png_path: str):
    """Load a map saved by map_generator.py and render it to an image."""
    if not os.path.exists(json_path):
        print(f"Skipped -> {json_path} not found")
        return None

    saved_map = MapGenerator.load(json_path)
    visualise_map(saved_map, save_path=png_path)
    return saved_map

def run_demo(seed: int | None = None):
    if seed is None:
        seed = random.randint(0, 2**31)

    # 1. Single balanced map
    gen_balanced = MapGenerator(width=42, height=22, alpha=0.0, seed=seed)
    gen_balanced.generate()
    visualise_map(gen_balanced, save_path="output/map_balanced.png")

    # 2. Alien-favoured
    gen_alien = MapGenerator(width=42, height=22, alpha=0.8, seed=seed)
    gen_alien.generate()
    visualise_map(gen_alien, save_path="output/map_alien_favoured.png")

    # 3. Player-favoured
    gen_player = MapGenerator(width=42, height=22, alpha=-0.8, seed=seed)
    gen_player.generate()
    visualise_map(gen_player, save_path="output/map_player_favoured.png")

    # 4. Side-by-side comparison of the current maps
    visualise_current_maps_comparison(
        [gen_balanced, gen_alien, gen_player],
        labels=["balanced", "alien-favoured", "player-favoured"],
        save_path="output/current_maps_comparison.png",
    )

    # 5. Render the map that map_generator.py saved to disk
    render_saved_map("maps/demo_balanced.json", "output/map_balanced_from_json.png")

    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize maps from map_generator.py")
    parser.add_argument(
        "seed",
        nargs="?",
        type=int,
        default=None,
        help="Optional integer seed. Different seeds produce different maps.",
    )
    args = parser.parse_args()
    run_demo(args.seed)