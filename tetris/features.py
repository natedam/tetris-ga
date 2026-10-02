"""
Board features used by the AI to judge a position.

All six features are computed from three numbers the engine gives us for the
board AFTER a candidate move:
    heights        - height of every column
    cells          - number of filled cells on the board
    lines_cleared  - lines cleared by the move

Feature               Meaning                                      Expected sign
--------------------  -------------------------------------------  -------------
aggregate_height      sum of all column heights                    negative
holes                 empty cells with a filled cell above them    negative
bumpiness             sum of |height difference| of neighbours     negative
lines_cleared         lines cleared by this move                   positive
max_height            height of the tallest column                 negative
wells                 total depth of "wells" (see below)           negative

The GA is NOT told these signs - it has to discover them.
"""

FEATURE_NAMES = [
    "aggregate_height",
    "holes",
    "bumpiness",
    "lines_cleared",
    "max_height",
    "wells",
]
NUM_FEATURES = len(FEATURE_NAMES)


def compute_features(heights, cells, lines_cleared, board_height=20):
    """Return the 6 features as a tuple, in the order of FEATURE_NAMES."""
    aggregate_height = sum(heights)

    # Every cell below the top of a column is either filled or a hole.
    # So: holes = (all cells below the column tops) - (filled cells).
    holes = aggregate_height - cells

    max_height = max(heights)

    # One pass over the columns computes bumpiness and wells together.
    # A well is a column lower than BOTH of its neighbours (a wall counts as
    # a very tall neighbour). Its depth is how far it is below the lower
    # neighbour. We add up the depths of all wells.
    bumpiness = 0
    wells = 0
    left = board_height                      # the left wall
    n = len(heights)
    for i in range(n):
        h = heights[i]
        right = heights[i + 1] if i + 1 < n else board_height
        if i + 1 < n:
            bumpiness += h - right if h > right else right - h
        lower_neighbour = left if left < right else right
        if lower_neighbour > h:
            wells += lower_neighbour - h
        left = h

    return (aggregate_height, holes, bumpiness, lines_cleared, max_height, wells)
