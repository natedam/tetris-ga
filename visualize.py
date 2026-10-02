"""
Animations of players, as GIF files.

Several players play the SAME piece sequence side by side, so you can see how
their styles differ. Each panel shows the board, the lines cleared so far, and
(optionally) the player's weight vector as a small bar chart - so a viewer can
connect "what the weights say" with "how the player plays".

Examples (run from the repository root):
    python visualize.py random hand --out results/demo.gif
    python visualize.py hand literature results/best_weights.json --pieces 400
    python visualize.py hand --board 10x10 --every 2

A player spec is one of:
    random | hand | literature | path/to/file.json
The JSON file must contain {"weights": [...]} and may contain {"name": "..."}.
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

from tetris.agent import LinearAgent, play_game
from tetris.baselines import HAND_TUNED_WEIGHTS, LITERATURE_WEIGHTS, RandomAgent

# ------------------------------------------------------------------ colours
BACKGROUND = (17, 20, 24)
PANEL = (27, 31, 36)
GRID = (38, 43, 49)
TEXT = (230, 230, 230)
MUTED = (150, 156, 163)
POSITIVE = (88, 196, 90)
NEGATIVE = (224, 81, 76)
# One colour per piece, in the order of engine.PIECE_NAMES = "IOTSZJL".
PIECE_COLOURS = [
    (63, 199, 224),   # I cyan
    (242, 211, 60),   # O yellow
    (164, 91, 214),   # T purple
    (88, 196, 90),    # S green
    (224, 81, 76),    # Z red
    (61, 111, 224),   # J blue
    (240, 140, 46),   # L orange
]
SHORT_FEATURE_NAMES = ["height", "holes", "bumpy", "lines", "max h", "wells"]


def _font(size, bold=False):
    """A TrueType font if one is installed, otherwise Pillow's built-in font."""
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)        # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _lighter(colour, amount=60):
    return tuple(min(255, c + amount) for c in colour)


def _darker(colour, amount=60):
    return tuple(max(0, c - amount) for c in colour)


# ----------------------------------------------------------------- drawing
def draw_board(draw, board, left, top, cell):
    """Draw one board with its top-left corner at (left, top)."""
    rows, cols = board.shape
    draw.rectangle([left - 2, top - 2, left + cols * cell + 1, top + rows * cell + 1],
                   outline=MUTED, width=2)
    for r in range(rows):
        y = top + (rows - 1 - r) * cell          # row 0 is the bottom -> flip vertically
        for c in range(cols):
            x = left + c * cell
            value = board[r, c]
            if value == 0:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=PANEL, outline=GRID)
            else:
                colour = PIECE_COLOURS[(value - 1) % len(PIECE_COLOURS)]
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=colour,
                               outline=_darker(colour))
                # small highlight -> blocks look a bit 3D
                draw.line([x + 1, y + 1, x + cell - 3, y + 1], fill=_lighter(colour))
                draw.line([x + 1, y + 1, x + 1, y + cell - 3], fill=_lighter(colour))


def draw_weights(draw, weights, left, top, width, fonts):
    """Horizontal bar per weight: green = positive, red = negative.
    Weights are scaled to length 1 first, so panels are comparable."""
    norm = sum(w * w for w in weights) ** 0.5 or 1.0
    weights = [w / norm for w in weights]
    label_w = 46
    row_h = 14
    bar_left = left + label_w
    bar_w = width - label_w - 4
    centre = bar_left + bar_w // 2
    draw.line([centre, top, centre, top + row_h * len(weights)], fill=MUTED)
    for i, w in enumerate(weights):
        y = top + i * row_h
        draw.text((left, y), SHORT_FEATURE_NAMES[i], font=fonts["small"], fill=MUTED)
        length = int(abs(w) * (bar_w // 2))
        if w >= 0:
            draw.rectangle([centre, y + 3, centre + length, y + row_h - 3], fill=POSITIVE)
        else:
            draw.rectangle([centre - length, y + 3, centre, y + row_h - 3], fill=NEGATIVE)


def render_frame(panels, t, board_shape, cell, show_weights, fonts):
    """Render moment t (= after t pieces) of every player into one image."""
    rows, cols = board_shape
    board_w, board_h = cols * cell, rows * cell
    panel_w = max(board_w + 24, 150)
    title_h, stats_h = 30, 26
    weights_h = 6 * 14 + 10 if show_weights else 0
    height = title_h + board_h + stats_h + weights_h + 16
    image = Image.new("RGB", (panel_w * len(panels), height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for p, panel in enumerate(panels):
        x0 = p * panel_w
        result = panel["result"]
        last = len(result.frames) - 1
        i = min(t, last)                      # a finished game keeps showing its last board
        left = x0 + (panel_w - board_w) // 2

        draw.text((x0 + panel_w // 2, 8), panel["name"], font=fonts["title"],
                  fill=TEXT, anchor="mt")
        draw_board(draw, result.frames[i], left, title_h, cell)
        stats = f"lines {result.frame_lines[i]}   pieces {i}"
        draw.text((x0 + panel_w // 2, title_h + board_h + 8), stats,
                  font=fonts["normal"], fill=TEXT, anchor="mt")
        if t >= last and panel["died"]:
            centre_y = title_h + board_h // 2
            draw.rectangle([left, centre_y - 14, left + board_w, centre_y + 14], fill=BACKGROUND)
            draw.text((left + board_w // 2, centre_y), "GAME OVER", font=fonts["title"],
                      fill=NEGATIVE, anchor="mm")
        if show_weights:
            weights_top = title_h + board_h + stats_h + 6
            if panel["weights"] is not None:
                draw_weights(draw, panel["weights"], x0 + 10, weights_top, panel_w - 20, fonts)
            else:
                draw.text((x0 + panel_w // 2, weights_top + 30), "no weights:\nrandom moves",
                          font=fonts["small"], fill=MUTED, anchor="ma", align="center")
    return image


def make_comparison_gif(players, seed, path, width=10, height=20, max_pieces=300,
                        every=1, frame_ms=60, cell=16, show_weights=True):
    """Play every player on the same piece sequence and save a side-by-side GIF.

    players : list of (name, agent) pairs. LinearAgents also get a weight chart.
    every   : keep only every n-th move (makes long games shorter/smaller).
    """
    panels = []
    for name, agent in players:
        result = play_game(agent, seed=seed, max_pieces=max_pieces,
                           width=width, height=height, record=True)
        panels.append({
            "name": name,
            "result": result,
            "died": max_pieces is None or result.pieces < max_pieces,
            "weights": getattr(agent, "weights", None),
        })
        print(f"  {name:>12}: {result.lines} lines, {result.pieces} pieces")

    longest = max(len(p["result"].frames) for p in panels) - 1
    times = list(range(0, longest + 1, every))
    if times[-1] != longest:
        times.append(longest)

    fonts = {"title": _font(15, bold=True), "normal": _font(13), "small": _font(11)}
    frames = [render_frame(panels, t, (height, width), cell, show_weights, fonts) for t in times]
    durations = [frame_ms] * len(frames)
    durations[-1] = 2500                        # hold the final frame
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print(f"Saved {path} ({len(frames)} frames)")
    return panels


# --------------------------------------------------------------------- CLI
def player_from_spec(spec, seed=0):
    """Turn 'random' / 'hand' / 'literature' / 'file.json' into (name, agent)."""
    if spec == "random":
        return "Random", RandomAgent(seed)
    if spec == "hand":
        return "Hand-tuned", LinearAgent(HAND_TUNED_WEIGHTS)
    if spec == "literature":
        return "Literature", LinearAgent(LITERATURE_WEIGHTS)
    with open(spec) as f:
        data = json.load(f)
    name = data.get("name", os.path.splitext(os.path.basename(spec))[0])
    return name, LinearAgent(data["weights"])


def main():
    parser = argparse.ArgumentParser(description="Side-by-side Tetris animation (GIF).")
    parser.add_argument("players", nargs="+",
                        help="random | hand | literature | path/to/weights.json")
    parser.add_argument("--seed", type=int, default=1_000_000_000, help="piece-sequence seed")
    parser.add_argument("--pieces", type=int, default=300, help="max pieces per game")
    parser.add_argument("--board", default="10x20", help="WIDTHxHEIGHT, e.g. 10x20")
    parser.add_argument("--every", type=int, default=1, help="keep every n-th move")
    parser.add_argument("--ms", type=int, default=60, help="milliseconds per frame")
    parser.add_argument("--no-weights", action="store_true", help="hide the weight charts")
    parser.add_argument("--out", default="results/animation.gif")
    args = parser.parse_args()

    width, height = (int(v) for v in args.board.lower().split("x"))
    players = [player_from_spec(s, seed=args.seed) for s in args.players]
    make_comparison_gif(players, args.seed, args.out, width, height, args.pieces,
                        args.every, args.ms, show_weights=not args.no_weights)


if __name__ == "__main__":
    main()
