import numpy as np


def board_from_rows(rows, height=20):
    """Build a board from strings drawn TOP to BOTTOM, e.g.

        board_from_rows(["#.........",
                         "##########"])

    '#' = filled, '.' = empty. Missing rows above are empty.
    Returns an array with row 0 = bottom (the engine's convention).
    """
    width = len(rows[0])
    board = np.zeros((height, width), dtype=np.uint8)
    for i, text in enumerate(reversed(rows)):       # reversed: last string = bottom row
        for c, ch in enumerate(text):
            if ch == "#":
                board[i, c] = 1
    return board
