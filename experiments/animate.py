"""
Phase 5 animations (GIFs in results/animations/), made from the saved results.

    python experiments/animate.py

1. styles_10x20.gif    - the 4 "style players" (see analysis.style_players) play the
                         same 400 pieces on the standard 10x20 board. Watch the
                         stack height and the holes counter: the top GA players keep
                         the board lower and with fewer holes.
2. race_10x10.gif      - the same 4 players on the 10x10 training board until every
                         one of them loses (a survival race).
3. evolution_10x10.gif - one GA run, the best player found so far after generation
                         0, 2, 5 and 49, on the 10x10 training board until all lose.

All three use the first test game (seed 1,000,000,000), which was never used in
training or validation and was not picked by hand. One game is only an
illustration - the statistics over many games are in results/tables and the figures.
"""

import pandas as pd

import analysis as A                     # also puts the repository root on sys.path
from tetris.agent import LinearAgent
from visualize import make_comparison_gif

SEED = A.TEST_SEEDS[0]

CAPTIONS = {
    "Hand-tuned": "my common-sense weights",
    "GA greedy": "values line clears most",
    "GA top-1": "best validation (80 runs)",
    "GA top-2": "2nd-best validation",
}

EVOLUTION_RUN = ("mutation_rate=0.15", 2)    # the run that produced "GA top-2"
EVOLUTION_GENERATIONS = [0, 2, 5, 49]


def style_gif_players():
    return [(name, LinearAgent(weights), CAPTIONS[name]) for name, weights, _ in A.style_players()]


def champion_so_far(group, seed, generation):
    """Weights and validation score of the run's champion after `generation`.
    Each row of the run CSV holds the best-validated player of that generation
    (columns w0..w5); the champion is the best of these so far."""
    history = pd.read_csv(A.RUNS / group / f"seed_{seed}.csv")
    best_score, best_weights = -1.0, None
    for row in history[history.generation <= generation].itertuples():
        if row.best_validation > best_score:
            best_score = row.best_validation
            best_weights = [getattr(row, f"w{i}") for i in range(6)]
    return best_weights, best_score


def main():
    A.ANIMATIONS.mkdir(parents=True, exist_ok=True)

    print("1/3 styles on the 10x20 board")
    make_comparison_gif(style_gif_players(), SEED, str(A.ANIMATIONS / "styles_10x20.gif"),
                        width=10, height=20, max_pieces=400, every=2, frame_ms=60, cell=16,
                        header="Same 400 pieces for everyone - standard 10x20 board")

    print("2/3 survival race on the 10x10 board")
    make_comparison_gif(style_gif_players(), SEED, str(A.ANIMATIONS / "race_10x10.gif"),
                        width=10, height=10, max_pieces=None, every=6, frame_ms=50, cell=18,
                        header="Survival race on the 10x10 training board - same pieces")

    print("3/3 one GA run, generation by generation")
    group, seed = EVOLUTION_RUN
    players = []
    for gen in EVOLUTION_GENERATIONS:
        weights, score = champion_so_far(group, seed, gen)
        name = f"Generation {gen}" + (" (end)" if gen == EVOLUTION_GENERATIONS[-1] else "")
        players.append((name, LinearAgent(weights), f"validation score {score:.0f}"))
    make_comparison_gif(players, SEED, str(A.ANIMATIONS / "evolution_10x10.gif"),
                        width=10, height=10, max_pieces=None, every=6, frame_ms=50, cell=18,
                        header=f"Evolution at work - best player so far in one GA run ({group}, seed {seed})")


if __name__ == "__main__":
    main()
