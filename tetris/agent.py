"""
The AI player: a linear evaluation function.

For every possible placement of the current piece, the agent computes the six
board features of the resulting position and gives it a score:

    score(move) = w1*f1 + w2*f2 + ... + w6*f6

It then plays the move with the highest score. The weight vector w is the
player's "DNA" - it is exactly what the genetic algorithm evolves.
"""

from dataclasses import dataclass, field

from tetris.engine import BOARD_HEIGHT, BOARD_WIDTH, TetrisGame
from tetris.features import compute_features


class LinearAgent:
    def __init__(self, weights):
        self.weights = [float(w) for w in weights]

    def evaluate(self, game, move):
        """Score of a move, or None if the move is not possible."""
        outcome = game.simulate(move)
        if outcome is None:
            return None
        heights, cells, lines = outcome
        features = compute_features(heights, cells, lines, game.height)
        # Add the terms one by one, left to right. (Python 3.12 changed the built-in
        # sum() to round differently, which changes how exact ties are broken.)
        score = 0.0
        for w, f in zip(self.weights, features):
            score += w * f
        return score

    def choose_move(self, game):
        best_move, best_score = None, float("-inf")
        for move in game.possible_moves():
            score = self.evaluate(game, move)
            # Strict ">" means ties go to the first move found (deterministic).
            if score is not None and score > best_score:
                best_move, best_score = move, score
        return best_move


@dataclass
class GameResult:
    lines: int
    pieces: int
    score: int
    clear_counts: list                              # moves that cleared 0,1,2,3,4 lines
    frames: list = field(default_factory=list)      # board after each move (if recorded)
    frame_lines: list = field(default_factory=list)  # total lines after each move (if recorded)


def play_game(agent, seed, max_pieces=None, width=BOARD_WIDTH, height=BOARD_HEIGHT,
              record=False):
    """Let `agent` play one game with the piece sequence given by `seed`.

    If record=True, a copy of the board after every move is kept in
    result.frames (used to make animations).
    """
    game = TetrisGame(seed=seed, width=width, height=height, max_pieces=max_pieces)
    frames = [game.board.copy()] if record else []
    frame_lines = [0] if record else []
    while not game.game_over:
        game.play(agent.choose_move(game))
        if record:
            frames.append(game.board.copy())
            frame_lines.append(game.lines)
    return GameResult(
        lines=game.lines,
        pieces=game.pieces_placed,
        score=game.score,
        clear_counts=list(game.clear_counts),
        frames=frames,
        frame_lines=frame_lines,
    )
