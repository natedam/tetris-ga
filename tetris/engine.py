"""
Headless Tetris engine.

Conventions used everywhere in this project
-------------------------------------------
* The board is a NumPy array of shape (height, width).
* Row 0 is the BOTTOM row (so "row r" means "r cells above the floor").
  This makes heights easy to reason about. Rendering code flips it.
* A cell holds 0 if empty, or (piece_id + 1) if filled, so the renderer
  can colour each block by the piece it came from.
* A "move" (placement) is a pair (rotation_index, x): which rotation of the
  current piece to use, and the column of the piece's left edge. The piece is
  then hard-dropped straight down (no sliding or spinning under overhangs).
  This is the standard setup used in Tetris-AI research.
* Pieces come from a seeded random generator, so the same seed always gives
  the same piece sequence. That lets us compare players fairly.

Speed trick
-----------
The GA plays millions of pieces, so the engine avoids copying the board for
every candidate move. For a hard drop we only need the column heights to know
where a piece lands, and the number of filled cells to count holes
(see features.py). The board itself is only copied when a candidate move
clears lines, because then the heights can change in non-obvious ways.
"""

import random

import numpy as np

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Tetris line-clear points (original Nintendo scoring, level 0).
# Only used for extra statistics, the GA fitness is "lines cleared".
LINE_SCORES = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}

PIECE_NAMES = "IOTSZJL"

# Each piece in its spawn orientation, drawn with row 0 at the BOTTOM.
# Every entry is a (row, col) cell.
_BASE_SHAPES = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],   # ####
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],   # ##
                                             # ##
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],   # .#.
                                             # ###
    "S": [(0, 0), (0, 1), (1, 1), (1, 2)],   # .##
                                             # ##.
    "Z": [(0, 1), (0, 2), (1, 0), (1, 1)],   # ##.
                                             # .##
    "J": [(0, 0), (0, 1), (0, 2), (1, 0)],   # #..
                                             # ###
    "L": [(0, 0), (0, 1), (0, 2), (1, 2)],   # ..#
                                             # ###
}


class Rotation:
    """One orientation of a piece, plus numbers pre-computed for fast drops."""

    def __init__(self, cells):
        self.cells = cells                                   # list of (row, col)
        self.width = max(c for _, c in cells) + 1
        self.height = max(r for r, _ in cells) + 1
        # For each column i of the piece: its lowest and highest cell row.
        self.bottom = [min(r for r, c in cells if c == i) for i in range(self.width)]
        self.top = [max(r for r, c in cells if c == i) for i in range(self.width)]
        # How many cells the piece has in each of its rows.
        self.row_counts = [sum(1 for r, _ in cells if r == k) for k in range(self.height)]


def _normalize(cells):
    """Shift cells so the smallest row and column are both 0; sort for comparison."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _all_rotations(cells):
    """Rotate a shape by 90 degrees up to 3 times and keep only distinct shapes.
    O has 1 distinct rotation, I/S/Z have 2, and T/J/L have 4."""
    rotations = []
    shape = _normalize(cells)
    for _ in range(4):
        if shape not in rotations:
            rotations.append(shape)
        shape = _normalize([(c, -r) for r, c in shape])      # rotate 90 degrees
    return [Rotation(s) for s in rotations]


# PIECES[piece_id] is the list of distinct rotations of that piece.
PIECES = [_all_rotations(_BASE_SHAPES[name]) for name in PIECE_NAMES]


def column_heights(board):
    """Height of every column = 1 + index of its highest filled cell (0 if empty)."""
    filled = board != 0
    height = board.shape[0]
    # argmax on the upside-down board finds the highest filled row of each column.
    top_from_above = filled[::-1].argmax(axis=0)
    heights = np.where(filled.any(axis=0), height - top_from_above, 0)
    return [int(h) for h in heights]


class TetrisGame:
    """A single game of Tetris.

    Parameters
    ----------
    seed : int
        Seed of the piece sequence. Same seed -> same pieces.
    width, height : int
        Board size (standard Tetris is 10 x 20).
    max_pieces : int or None
        Stop the game after this many pieces (None = play until game over).
    """

    def __init__(self, seed=0, width=BOARD_WIDTH, height=BOARD_HEIGHT, max_pieces=None):
        self.width = width
        self.height = height
        self.max_pieces = max_pieces
        self._rng = random.Random(seed)

        self.board = np.zeros((height, width), dtype=np.uint8)
        self.heights = [0] * width          # height of every column
        self.row_fill = [0] * height        # number of filled cells in every row
        self.cells = 0                      # total number of filled cells

        # Statistics
        self.lines = 0                      # total lines cleared
        self.score = 0                      # classic Tetris points
        self.pieces_placed = 0
        self.clear_counts = [0, 0, 0, 0, 0]  # how many moves cleared 0/1/2/3/4 lines
        self.game_over = False

        self._moves = None                  # cache for possible_moves()
        self.current_piece = self._draw_piece()
        self._check_game_over()

    # ------------------------------------------------------------------ pieces
    def _draw_piece(self):
        return self._rng.randrange(len(PIECES))

    # ------------------------------------------------------------------- moves
    def possible_moves(self):
        """All (rotation_index, x) placements of the current piece that fit on the board.

        The list is computed once per piece and then re-used (cached)."""
        if self._moves is not None:
            return self._moves
        moves = []
        for rot_index, rot in enumerate(PIECES[self.current_piece]):
            for x in range(self.width - rot.width + 1):
                if self._landing_row(rot, x) + rot.height <= self.height:
                    moves.append((rot_index, x))
        self._moves = moves
        return moves

    def _landing_row(self, rot, x):
        """Row where the bottom of the piece stops after a hard drop at column x.

        The piece stops as soon as one of its columns touches the stack, i.e.
        at the highest value of (column height - lowest cell of piece in that column).
        """
        heights = self.heights
        base = 0
        for i in range(rot.width):
            row = heights[x + i] - rot.bottom[i]
            if row > base:
                base = row
        return base

    def simulate(self, move):
        """Result of a move WITHOUT changing the game.

        Returns (heights, cells, lines_cleared) of the board after the move,
        or None if the piece would stick out above the top of the board.
        This is all the information features.py needs.
        """
        rot_index, x = move
        rot = PIECES[self.current_piece][rot_index]
        base = self._landing_row(rot, x)
        if base + rot.height > self.height:
            return None

        # Which of the rows touched by the piece become full?
        lines = 0
        for k in range(rot.height):
            if self.row_fill[base + k] + rot.row_counts[k] == self.width:
                lines += 1

        if lines == 0:
            # Fast path: only the columns under the piece change height.
            heights = self.heights.copy()
            for i in range(rot.width):
                heights[x + i] = base + rot.top[i] + 1
            return heights, self.cells + 4, 0

        # Slow path: lines are cleared, so build the new board and measure it.
        board = self._place_on_copy(rot, x, base)
        board, lines = _clear_full_rows(board)
        return column_heights(board), int(np.count_nonzero(board)), lines

    def play(self, move):
        """Apply a move: drop the piece, clear lines, draw the next piece.

        Returns the number of lines cleared by this move.
        """
        if self.game_over:
            raise RuntimeError("The game is already over")
        rot_index, x = move
        rot = PIECES[self.current_piece][rot_index]
        base = self._landing_row(rot, x)
        if base + rot.height > self.height:
            raise ValueError(f"Move {move} does not fit on the board")

        board = self._place_on_copy(rot, x, base)
        board, lines = _clear_full_rows(board)
        self._set_board_state(board)

        self.lines += lines
        self.score += LINE_SCORES[lines]
        self.clear_counts[lines] += 1
        self.pieces_placed += 1

        self._moves = None
        self.current_piece = self._draw_piece()
        self._check_game_over()
        return lines

    # ------------------------------------------------- set up positions (tests)
    def set_board(self, board):
        """Replace the board (row 0 = bottom). Mainly used by the unit tests."""
        self._set_board_state(np.array(board, dtype=np.uint8))
        self._moves = None
        self._check_game_over()

    def set_current_piece(self, name):
        """Force the current piece, e.g. set_current_piece("I"). Used by tests."""
        self.current_piece = PIECE_NAMES.index(name)
        self._moves = None
        self._check_game_over()

    # ----------------------------------------------------------------- helpers
    def _set_board_state(self, board):
        """Store a board and recompute the numbers derived from it."""
        self.board = board
        self.heights = column_heights(board)
        self.row_fill = [int(n) for n in np.count_nonzero(board, axis=1)]
        self.cells = sum(self.row_fill)

    def _place_on_copy(self, rot, x, base):
        board = self.board.copy()
        for r, c in rot.cells:
            board[base + r, x + c] = self.current_piece + 1
        return board

    def _check_game_over(self):
        """The game ends when the piece cap is reached, or when the current
        piece cannot be placed anywhere on the board."""
        reached_cap = self.max_pieces is not None and self.pieces_placed >= self.max_pieces
        self.game_over = reached_cap or not self.possible_moves()


def _clear_full_rows(board):
    """Remove full rows, let everything above fall down. Returns (board, lines)."""
    full = (board != 0).all(axis=1)
    lines = int(full.sum())
    if lines == 0:
        return board, 0
    remaining = board[~full]
    empty_rows = np.zeros((lines, board.shape[1]), dtype=board.dtype)
    # Row 0 is the bottom, so the new empty rows go on TOP (the end of the array).
    return np.vstack([remaining, empty_rows]), lines
