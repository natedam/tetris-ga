"""
Shared helpers for the analysis scripts (stats.py, plots.py, animate.py):
loading the saved results, a few statistics helpers, and "style" measurements
of how a player actually plays.

Nothing here runs the GA - it only reads what experiments/run.py saved in results/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tetris.agent import LinearAgent  # noqa: E402
from tetris.baselines import HAND_TUNED_WEIGHTS, LITERATURE_WEIGHTS  # noqa: E402
from tetris.engine import TetrisGame  # noqa: E402
from tetris.features import compute_features  # noqa: E402

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
ANIMATIONS = RESULTS / "animations"

N_SEEDS = 10
TEST_SEEDS = [1_000_000_000 + i for i in range(20)]   # same as ga.evolution.final_test_seeds(20)

# Which run group belongs to which experiment, in display order: (group folder, short label)
EXPERIMENTS = {
    "E2": [("population_size=20", "pop 20"), ("default", "pop 50"), ("population_size=100", "pop 100")],
    "E3": [("mutation_rate=0.01", "mut 0.01"), ("default", "mut 0.05"), ("mutation_rate=0.15", "mut 0.15")],
    "E4": [("default", "tournament + arithmetic"), ("crossover=uniform", "tournament + uniform"),
           ("selection=roulette", "roulette + arithmetic"),
           ("crossover=uniform+selection=roulette", "roulette + uniform")],
}
POPULATION = {"population_size=20": 20, "population_size=100": 100}   # all other groups: 50
GAMES_PER_EVAL = 3

FEATURES = ["height", "holes", "bumpiness", "lines", "max height", "wells"]


# ------------------------------------------------------------------ loading
def load_histories(group):
    """All 10 runs of one group as one DataFrame (one row per generation, plus a 'seed' column)."""
    frames = []
    for seed in range(N_SEEDS):
        h = pd.read_csv(RUNS / group / f"seed_{seed}.csv")
        h["seed"] = seed
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def load_champion(group, seed):
    with open(RUNS / group / f"seed_{seed}_champion.json") as f:
        return json.load(f)


def champion_table():
    """One row per GA run: group, seed, validation score, generation found, weights,
    and the champion's score on the 20 test games (from results/E2-E4_champions.csv)."""
    games = pd.read_csv(RESULTS / "E2-E4_champions.csv")
    test = games.groupby("player")["lines"].agg(test_mean="mean", test_median="median").reset_index()
    rows = []
    for group in sorted(p.name for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("quick_")):
        for seed in range(N_SEEDS):
            c = load_champion(group, seed)
            rows.append({"group": group, "seed": seed, "player": c["name"],
                         "validation": c["validation"], "generation": c["generation"],
                         "weights": c["weights"]})
    return pd.DataFrame(rows).merge(test, on="player", how="left")


def per_game(csv_name):
    """A test-game CSV (E1 / E5 / E2-E4) as a DataFrame."""
    return pd.read_csv(RESULTS / csv_name)


# ----------------------------------------------------------- weight analysis
def effective_weights(weights):
    """Rewrite a weight vector without the redundant 'holes' weight.

    holes = height - cells, and the cells after a move = cells before + 4 - 10*lines,
    so for every possible move: holes = height + 10*lines + constant.
    Therefore w_height*height + w_holes*holes + w_lines*lines
           = (w_height + w_holes)*height + (w_lines + 10*w_holes)*lines + constant,
    and the player ranks the moves exactly like one with these 5 'effective' weights
    (10 = board width; tests/test_analysis.py checks this). Only exact ties between
    two moves may be broken differently, because of floating-point rounding.

    Everything is divided by the size of the height penalty, so the numbers mean
    "how many cells of stack height is this worth":
      line_value - net reward for clearing one line (includes the 10 cells of height it removes)
      bumpiness, max_height, wells - cost of one unit of each feature
    """
    w = np.asarray(weights, dtype=float)
    w = w / np.linalg.norm(w)
    height = w[0] + w[1]                       # effective height weight (negative)
    lines = w[3] + 10 * w[1]                   # effective lines weight
    scale = -height
    return {
        "line_value": (10 * scale + lines) / scale,
        "bumpiness": -w[2] / scale,
        "max_height": -w[4] / scale,
        "wells": -w[5] / scale,
    }


# --------------------------------------------------------------- statistics
def a12(x, y):
    """Vargha-Delaney A12 effect size: probability that a random value from x is
    larger than a random value from y (ties count half). 0.5 = no difference;
    commonly 0.56 small, 0.64 medium, 0.71 large (and the mirror values below 0.5)."""
    x, y = np.asarray(x), np.asarray(y)
    greater = (x[:, None] > y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return (greater + 0.5 * ties) / (len(x) * len(y))


def holm(pvalues):
    """Holm-Bonferroni correction for several tests at once (returns adjusted p-values)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def mean_ci(values, axis=0, confidence=0.95):
    """Mean and half-width of a t-based confidence interval (over runs)."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    sem = values.std(axis=axis, ddof=1) / np.sqrt(n)
    return mean, sem * stats.t.ppf(0.5 + confidence / 2, n - 1)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------- playing style
def style_profile(weights, seeds=TEST_SEEDS[:10], height=20, max_pieces=1000):
    """Measure HOW a player plays (not just how many lines it clears).

    Plays the given games (first `max_pieces` pieces of each) and averages, after
    every move: stack height, tallest column, holes, bumpiness; plus which share
    of the line clears were doubles/triples/tetrises."""
    agent = LinearAgent(weights)
    per_move = {"avg_height": [], "max_height": [], "holes": [], "bumpiness": []}
    clears = np.zeros(5)
    lines = pieces = 0
    for seed in seeds:
        game = TetrisGame(seed=seed, height=height, max_pieces=max_pieces)
        while not game.game_over:
            game.play(agent.choose_move(game))
            f = compute_features(game.heights, game.cells, 0, game.height)
            per_move["avg_height"].append(f[0] / game.width)
            per_move["holes"].append(f[1])
            per_move["bumpiness"].append(f[2])
            per_move["max_height"].append(f[4])
        clears += game.clear_counts
        lines += game.lines
        pieces += game.pieces_placed
    n_clears = clears[1:].sum() or 1
    profile = {k: float(np.mean(v)) for k, v in per_move.items()}
    profile["multi_line_share"] = float(clears[2:].sum() / n_clears)
    profile["lines_per_piece"] = lines / pieces
    return profile


# --------------------------------------------------------- the style players
def style_players():
    """The players compared in the style analysis and the animations, picked by
    fixed rules (never by test score):
      - GA top-1 / GA top-2: the best and 2nd-best VALIDATION scores of all 80 runs
      - GA greedy: of the 10 default-config champions, the one whose weights value a
        cleared line the most (highest effective line_value)
    plus the hand-tuned player. Returns a list of (short name, weights, run name)."""
    table = champion_table()
    table["line_value"] = [effective_weights(w)["line_value"] for w in table.weights]
    top = table.sort_values("validation", ascending=False).iloc[:2]
    default = table[table.group == "default"]
    greedy = default.loc[default.line_value.idxmax()]
    return [
        ("Hand-tuned", HAND_TUNED_WEIGHTS, "hand-tuned weights"),
        ("GA greedy", greedy.weights, greedy.player),
        ("GA top-1", top.iloc[0].weights, top.iloc[0].player),
        ("GA top-2", top.iloc[1].weights, top.iloc[1].player),
    ]


BASELINE_WEIGHTS = {"Hand-tuned": HAND_TUNED_WEIGHTS, "Literature": LITERATURE_WEIGHTS}
