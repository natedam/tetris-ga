"""Unit tests for the board features and the agent."""

import numpy as np

from tests.helpers import board_from_rows
from tetris.agent import LinearAgent, play_game
from tetris.baselines import HAND_TUNED_WEIGHTS, RandomAgent
from tetris.engine import TetrisGame, column_heights
from tetris.features import FEATURE_NAMES, compute_features


def features_of(rows, lines_cleared=0):
    board = board_from_rows(rows)
    heights = column_heights(board)
    return dict(zip(FEATURE_NAMES,
                    compute_features(heights, int(np.count_nonzero(board)), lines_cleared)))


def count_holes_slowly(board):
    """Direct definition: an empty cell with at least one filled cell above it."""
    holes = 0
    rows, cols = board.shape
    for c in range(cols):
        for r in range(rows):
            if board[r, c] == 0 and board[r + 1:, c].any():
                holes += 1
    return holes


def test_empty_board_has_all_features_zero():
    f = features_of(["." * 10])
    assert all(v == 0 for v in f.values())


def test_heights_bumpiness_and_max_height():
    f = features_of([
        "#.........",
        "#.#.......",
        "#.##......",
    ])
    # heights: 3,0,2,1,0,0,0,0,0,0
    assert f["aggregate_height"] == 6
    assert f["max_height"] == 3
    assert f["bumpiness"] == 3 + 2 + 1 + 1


def test_holes_formula_matches_direct_count():
    rows = [
        "##..#.....",
        "#...#..#..",
        ".#.##..#.#",
        "#.#..###.#",
    ]
    board = board_from_rows(rows)
    assert features_of(rows)["holes"] == count_holes_slowly(board)


def test_holes_formula_on_random_boards():
    rng = np.random.default_rng(0)
    for _ in range(50):
        board = (rng.random((20, 10)) < 0.4).astype(np.uint8)
        heights = column_heights(board)
        f = compute_features(heights, int(board.sum()), 0)
        assert f[1] == count_holes_slowly(board)


def test_wells():
    # heights: 3,0,3,0,0,0,0,0,0,0 -> column 1 is a well of depth 3.
    # Column 9 is next to the wall (tall) and column 8 (height 0): not a well.
    f = features_of([
        "#.#.......",
        "#.#.......",
        "#.#.......",
    ])
    assert f["wells"] == 3


def test_well_next_to_wall():
    # heights: 0,2,0,...: column 0 is between the wall and a height-2 column -> depth 2.
    f = features_of([
        ".#........",
        ".#........",
    ])
    assert f["wells"] == 2


def test_lines_cleared_is_passed_through():
    assert features_of(["." * 10], lines_cleared=3)["lines_cleared"] == 3


# ------------------------------------------------------------------- agent
def test_agent_prefers_the_move_that_clears_a_line():
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows(["....######"]))
    game.set_current_piece("I")
    move = LinearAgent(HAND_TUNED_WEIGHTS).choose_move(game)
    assert game.simulate(move)[2] == 1          # the chosen move clears a line


def test_agent_only_plays_legal_moves_and_is_deterministic():
    agent = LinearAgent(HAND_TUNED_WEIGHTS)
    a = play_game(agent, seed=5, max_pieces=200)
    b = play_game(agent, seed=5, max_pieces=200)
    assert (a.lines, a.pieces) == (b.lines, b.pieces)


def test_hand_tuned_beats_random():
    random_lines = sum(play_game(RandomAgent(s), seed=s).lines for s in range(3))
    hand_lines = sum(play_game(LinearAgent(HAND_TUNED_WEIGHTS), seed=s, max_pieces=300).lines
                     for s in range(3))
    assert hand_lines > 10 * max(random_lines, 1)


def test_recording_frames():
    result = play_game(LinearAgent(HAND_TUNED_WEIGHTS), seed=1, max_pieces=20, record=True)
    assert len(result.frames) == result.pieces + 1          # + the empty starting board
    assert result.frame_lines[-1] == result.lines
