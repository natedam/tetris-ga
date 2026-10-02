"""Tests for the Phase 5 analysis helpers and the animation counters."""

import numpy as np

from experiments.analysis import a12, effective_weights, holm
from tests.helpers import board_from_rows
from tetris.agent import LinearAgent
from tetris.baselines import HAND_TUNED_WEIGHTS
from tetris.engine import TetrisGame, column_heights
from tetris.features import compute_features
from visualize import board_stats


def test_board_stats_matches_features():
    board = board_from_rows(["..........",
                             "#.........",
                             "#..#......",
                             "#.##....##",
                             "##########"])
    heights = column_heights(board)
    features = compute_features(heights, int(np.count_nonzero(board)), 0)
    assert board_stats(board) == (features[4], features[1]) == (4, 0)
    board[1, 0] = 0                              # punch a hole under column 0
    assert board_stats(board) == (4, 1)


def test_holes_weight_is_redundant():
    """The claim behind effective_weights(): moving the 'holes' weight into the
    'height' and 'lines' weights changes every move's score by the SAME constant,
    so the player ranks the moves exactly as before."""
    rng = np.random.default_rng(0)
    for w in [HAND_TUNED_WEIGHTS] + [list(rng.normal(size=6)) for _ in range(3)]:
        original = LinearAgent(w)
        same_player = LinearAgent([w[0] + w[1], 0.0, w[2], w[3] + 10 * w[1], w[4], w[5]])
        game = TetrisGame(seed=123, max_pieces=200)
        while not game.game_over:
            differences = [original.evaluate(game, m) - same_player.evaluate(game, m)
                           for m in game.possible_moves() if game.simulate(m) is not None]
            assert max(differences) - min(differences) < 1e-9
            game.play(original.choose_move(game))


def test_effective_weights_of_hand_tuned():
    e = effective_weights(HAND_TUNED_WEIGHTS)
    # height -1.5, lines 1 + 10*(-1) = -9 -> a line is worth (15 - 9) / 1.5 = 4 cells of height
    assert abs(e["line_value"] - 4.0) < 1e-9
    assert abs(e["bumpiness"] - 0.2 / 1.5) < 1e-9


def test_a12_and_holm():
    assert a12([1, 2, 3], [1, 2, 3]) == 0.5
    assert a12([4, 5], [1, 2]) == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
