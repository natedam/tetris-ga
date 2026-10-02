"""
Baseline players that the evolved players are compared against (RQ1).

1. RandomAgent       - picks a random legal placement. The "floor".
2. HAND_TUNED        - weights chosen by hand, using common sense only
                       (holes are very bad, height is bad, clearing lines is good).
3. LITERATURE        - published weights by Yiyuan Lee (2013), "Tetris AI - The
                       (Near) Perfect Bot". They use 4 of our 6 features, so
                       max_height and wells get weight 0. A strong reference point.
"""

import random

from tetris.agent import LinearAgent

# Order: aggregate_height, holes, bumpiness, lines_cleared, max_height, wells
HAND_TUNED_WEIGHTS = [-0.5, -1.0, -0.2, 1.0, -0.1, -0.1]

LITERATURE_WEIGHTS = [-0.510066, -0.35663, -0.184483, 0.760666, 0.0, 0.0]


class RandomAgent:
    """Chooses a uniformly random legal placement."""

    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def choose_move(self, game):
        return self._rng.choice(game.possible_moves())


def hand_tuned_agent():
    return LinearAgent(HAND_TUNED_WEIGHTS)


def literature_agent():
    return LinearAgent(LITERATURE_WEIGHTS)
