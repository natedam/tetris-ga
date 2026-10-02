"""
Animations of players, as GIF files.

Several players play the SAME piece sequence side by side, so you can see how
their styles differ. Each panel shows the board, live counters (lines, pieces,
stack height, holes), a small chart of the holes over time, and (optionally)
the player's weight vector as a bar chart - so a viewer can connect "what the
weights say" with "how the player plays".

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

import numpy as np
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
CHART_LINE = (99, 160, 230)
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


# ------------------------------------------------------------ board counters
def board_stats(board):
    """(tallest column, number of holes) of a board (row 0 = bottom).
    A hole is an empty cell with a filled cell somewhere above it."""
    filled = board != 0
    rows = board.shape[0]
    # first filled cell from the top of each column -> column height
    heights = np.where(filled.any(axis=0), rows - np.argmax(filled[::-1], axis=0), 0)
    return int(heights.max()), int(heights.sum() - filled.sum())


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


def draw_history(draw, values, t, longest, top_value, left, top, width, height, fonts):
    """Small line chart of one counter (e.g. holes) from move 0 up to move t.
    All panels use the same scales (longest game, largest value) so they can be compared."""
    draw.text((left, top), "holes over time", font=fonts["small"], fill=MUTED)
    top += 15
    draw.line([left, top + height, left + width, top + height], fill=GRID)      # time axis
    step = max(1, (t + 1) // width)              # at most ~1 point per pixel
    points = [(left + width * k / max(longest, 1),
               top + height - height * values[k] / top_value)
              for k in range(0, min(t, len(values) - 1) + 1, step)]
    if len(points) > 1:
        draw.line(points, fill=CHART_LINE, width=2)


def render_frame(panels, t, board_shape, cell, show_weights, fonts, header=None, longest=1):
    """Render moment t (= after t pieces) of every player into one image."""
    rows, cols = board_shape
    board_w, board_h = cols * cell, rows * cell
    panel_w = max(board_w + 24, 170)
    header_h = 34 if header else 0
    caption_h = 16 if any(p["caption"] for p in panels) else 0
    title_h = 30 + caption_h
    stats_h, chart_h = 44, 60
    weights_h = 6 * 14 + 14 if show_weights else 0
    height = header_h + title_h + board_h + stats_h + chart_h + weights_h + 12
    image = Image.new("RGB", (panel_w * len(panels), height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    if header:
        draw.text((image.width // 2, 10), header, font=fonts["title"], fill=TEXT, anchor="mt")
    top_holes = max(max(p["holes"]) for p in panels) or 1
    all_died = all(p["died"] for p in panels)
    winner = max(range(len(panels)), key=lambda k: panels[k]["result"].lines)

    for p, panel in enumerate(panels):
        x0 = p * panel_w
        result = panel["result"]
        last = len(result.frames) - 1
        i = min(t, last)                      # a finished game keeps showing its last board
        left = x0 + (panel_w - board_w) // 2
        y = header_h + 8

        draw.text((x0 + panel_w // 2, y), panel["name"], font=fonts["title"], fill=TEXT, anchor="mt")
        if panel["caption"]:
            draw.text((x0 + panel_w // 2, y + 21), panel["caption"], font=fonts["small"],
                      fill=MUTED, anchor="mt")
        board_top = header_h + title_h
        draw_board(draw, result.frames[i], left, board_top, cell)

        # live counters under the board
        y = board_top + board_h + 8
        draw.text((x0 + panel_w // 2, y), f"lines {result.frame_lines[i]}   pieces {i}",
                  font=fonts["normal"], fill=TEXT, anchor="mt")
        draw.text((x0 + panel_w // 2, y + 18), f"height {panel['height'][i]}   holes {panel['holes'][i]}",
                  font=fonts["normal"], fill=MUTED, anchor="mt")
        draw_history(draw, panel["holes"], i, longest, top_holes, x0 + 12, y + stats_h,
                     panel_w - 24, chart_h - 24, fonts)

        if t >= last and panel["died"]:       # this player's game has ended
            centre_y = board_top + board_h // 2
            is_winner = all_died and p == winner and t >= longest
            text, colour = ("WINNER", POSITIVE) if is_winner else ("GAME OVER", NEGATIVE)
            draw.rectangle([left, centre_y - 14, left + board_w, centre_y + 14], fill=BACKGROUND)
            draw.text((left + board_w // 2, centre_y), text, font=fonts["title"], fill=colour, anchor="mm")
        if show_weights:
            weights_top = board_top + board_h + stats_h + chart_h + 8
            if panel["weights"] is not None:
                draw_weights(draw, panel["weights"], x0 + 10, weights_top, panel_w - 20, fonts)
            else:
                draw.text((x0 + panel_w // 2, weights_top + 30), "no weights:\nrandom moves",
                          font=fonts["small"], fill=MUTED, anchor="ma", align="center")
    return image


def make_comparison_gif(players, seed, path, width=10, height=20, max_pieces=300,
                        every=1, frame_ms=60, cell=16, show_weights=True, header=None):
    """Play every player on the same piece sequence and save a side-by-side GIF.

    players : list of (name, agent) or (name, agent, caption). LinearAgents also
              get a weight chart; the caption is a short grey line under the name.
    every   : keep only every n-th move (makes long games shorter/smaller).
    header  : optional line of text across the top of the animation.
    """
    panels = []
    for name, agent, *caption in players:
        result = play_game(agent, seed=seed, max_pieces=max_pieces,
                           width=width, height=height, record=True)
        counters = [board_stats(board) for board in result.frames]
        panels.append({
            "name": name,
            "caption": caption[0] if caption else "",
            "result": result,
            "died": max_pieces is None or result.pieces < max_pieces,
            "weights": getattr(agent, "weights", None),
            "height": [c[0] for c in counters],
            "holes": [c[1] for c in counters],
        })
        print(f"  {name:>12}: {result.lines} lines, {result.pieces} pieces")

    longest = max(len(p["result"].frames) for p in panels) - 1
    times, t = [], 0
    while t < longest:
        times.append(t)
        still_playing = sum(len(p["result"].frames) - 1 > t for p in panels)
        t += every if still_playing > 1 else 3 * every      # fast-forward when one player is left
    times.append(longest)

    fonts = {"title": _font(15, bold=True), "normal": _font(13), "small": _font(11)}
    frames = [render_frame(panels, t, (height, width), cell, show_weights, fonts, header, longest)
              for t in times]
    # One shared colour palette for all frames (taken from the last frame, which
    # contains every colour) -> no flicker and a much smaller file.
    palette = frames[-1].quantize(colors=128, dither=Image.Dither.NONE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
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
