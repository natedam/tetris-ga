"""Unit tests for the Tetris engine. Run with:  python -m pytest"""

import random

import numpy as np

from tests.helpers import board_from_rows
from tetris.engine import PIECE_NAMES, PIECES, TetrisGame


def piece(name):
    return PIECES[PIECE_NAMES.index(name)]


# ------------------------------------------------------------------ pieces
def test_number_of_distinct_rotations():
    expected = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
    for name, n in expected.items():
        assert len(piece(name)) == n, name


def test_every_rotation_has_four_cells():
    for rotations in PIECES:
        for rot in rotations:
            assert len(rot.cells) == 4
            assert sum(rot.row_counts) == 4


def test_number_of_placements_on_empty_board():
    # O: 9 columns. I: 7 (flat) + 10 (upright). T: 8 + 9 + 8 + 9.
    for name, n in {"O": 9, "I": 17, "T": 34, "S": 17}.items():
        game = TetrisGame(seed=0)
        game.set_current_piece(name)
        assert len(game.possible_moves()) == n, name


# ------------------------------------------------------------ piece sequence
def test_same_seed_gives_same_pieces():
    def sequence(seed):
        game = TetrisGame(seed=seed)
        out = []
        for _ in range(30):
            out.append(game.current_piece)
            game.current_piece = game._draw_piece()
        return out

    assert sequence(7) == sequence(7)
    assert sequence(7) != sequence(8)


# --------------------------------------------------------------- hard drop
def test_piece_lands_on_floor():
    game = TetrisGame(seed=0)
    game.set_current_piece("O")
    game.play((0, 0))
    assert game.heights[:3] == [2, 2, 0]
    assert game.cells == 4


def test_piece_lands_on_highest_column_under_it():
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows([
        "#.........",
        "#.........",
        "#.........",
    ]))
    game.set_current_piece("I")
    game.play((0, 0))                       # flat I over columns 0-3
    # Column 0 is 3 high, so the flat I rests on row 3 -> all four columns height 4.
    assert game.heights[:5] == [4, 4, 4, 4, 0]
    assert game.cells == 3 + 4


def test_no_sliding_under_overhang():
    # Column 1 has a hole under an overhang. Dropping an upright I in column 1
    # must stop ON TOP of the overhang, not slide into the hole.
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows([
        "##........",
        "#.........",
    ]))
    game.set_current_piece("I")
    upright = 1 if piece("I")[1].width == 1 else 0
    game.play((upright, 1))
    assert game.heights[1] == 2 + 4


# ------------------------------------------------------------- line clears
def test_single_line_clear():
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows(["....######"]))
    game.set_current_piece("I")
    lines = game.play((0, 0))               # flat I fills columns 0-3
    assert lines == 1
    assert game.cells == 0
    assert game.heights == [0] * 10


def test_tetris_clears_four_lines_and_scores_1200():
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows(["#########."] * 4))
    game.set_current_piece("I")
    upright = [i for i, r in enumerate(piece("I")) if r.width == 1][0]
    lines = game.play((upright, 9))
    assert lines == 4
    assert game.score == 1200
    assert game.clear_counts[4] == 1
    assert game.cells == 0


def test_rows_above_a_cleared_line_fall_down():
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows([
        ".....#....",       # this block should fall one row after the clear
        "....######",
    ]))
    game.set_current_piece("I")
    game.play((0, 0))                       # flat I completes the bottom row
    assert game.board[0, 5] != 0            # the block is now on the floor
    assert game.cells == 1
    assert game.heights[5] == 1


def test_clear_can_expose_a_hole():
    # Column 0 has an empty cell in row 0 covered by a block in row 1.
    # The T fills row 1, the row is cleared, and column 0 becomes empty:
    # its height must drop from 2 to 0 (not just by 1).
    game = TetrisGame(seed=0)
    game.set_board(board_from_rows([
        "#...######",
        ".#########",
    ]))
    game.set_current_piece("T")             # rotation 0 = flat side down, bump on top
    predicted = game.simulate((0, 1))
    lines = game.play((0, 1))
    assert lines == 1
    assert game.heights == [0, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert game.cells == 10
    assert predicted == (game.heights, game.cells, 1)


def test_simulate_matches_play_on_random_games():
    """simulate() (fast path) must agree with play() (full board update) on every move."""
    rng = random.Random(0)
    for seed in range(20):
        game = TetrisGame(seed=seed, max_pieces=200)
        while not game.game_over:
            moves = game.possible_moves()
            for move in moves:              # every candidate move
                heights, cells, lines = game.simulate(move)
                copy = TetrisGame(seed=0)
                copy.set_board(game.board.copy())
                copy.set_current_piece(PIECE_NAMES[game.current_piece])
                real_lines = copy.play(move)
                assert (heights, cells, lines) == (copy.heights, copy.cells, real_lines)
            game.play(rng.choice(moves))


# --------------------------------------------------------------- game over
def _nearly_full_board():
    """Every row has exactly one gap (so no row is full), at column r % 10.
    Result: columns 0-8 have height 20, column 9 has height 19."""
    board = np.ones((20, 10), dtype=np.uint8)
    for r in range(20):
        board[r, r % 10] = 0
    return board


def test_game_over_when_no_piece_fits():
    game = TetrisGame(seed=0)
    game.set_board(_nearly_full_board())
    assert game.game_over                   # only one free cell at the top: nothing fits


def test_not_game_over_when_top_row_is_free():
    game = TetrisGame(seed=0)
    board = _nearly_full_board()
    board[19, :] = 0                        # free the whole top row
    game.set_board(board)
    game.set_current_piece("I")
    assert not game.game_over               # a flat I fits in the top row


def test_piece_cap_ends_game():
    game = TetrisGame(seed=0, max_pieces=5)
    for _ in range(5):
        assert not game.game_over
        game.play(game.possible_moves()[0])
    assert game.game_over
    assert game.pieces_placed == 5


def test_random_play_eventually_ends():
    game = TetrisGame(seed=3)
    rng = random.Random(3)
    while not game.game_over:
        game.play(rng.choice(game.possible_moves()))
    assert game.pieces_placed > 0
    assert game.pieces_placed < 1000


def test_small_board_works():
    game = TetrisGame(seed=1, width=6, height=8)
    rng = random.Random(1)
    while not game.game_over:
        game.play(rng.choice(game.possible_moves()))
    assert game.board.shape == (8, 6)
