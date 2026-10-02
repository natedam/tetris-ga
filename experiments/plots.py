"""
All figures of the report. Reads the saved results (and the tables made by
stats.py) and writes PNG files to results/figures/.

Run from the repository root, after stats.py:
    python experiments/stats.py
    python experiments/plots.py

Style rules (kept the same in every figure so they read as one set):
  * colour = who the player is, and the same player always has the same colour:
    hand-tuned = orange, GA = blue, GA top-2 = violet, GA greedy = aqua,
    literature / random = greys
  * thin lines, light solid gridlines, a legend whenever there are 2+ series
  * the palette was checked with a colour-blindness validator
"""

import json

import matplotlib

matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.stats import kruskal, spearmanr  # noqa: E402

import analysis as A                      # noqa: E402

# ----------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"          # secondary text
MUTED = "#898781"          # axis labels, ticks
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
BLUE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]            # ordered variants (light -> dark)
E4_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # 4 unordered variants (slots 1-4)

PLAYER_COLORS = {"Random": MUTED, "Literature": INK_2, "Hand-tuned": ORANGE, "GA": BLUE,
                 "GA greedy": AQUA, "GA top-1": BLUE, "GA top-2": VIOLET}
PLAYER_MARKERS = {"Hand-tuned": "s", "GA greedy": "D", "GA top-1": "o", "GA top-2": "^"}

comma = FuncFormatter(lambda v, _: f"{v:,.0f}")


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASELINE, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "lines.linewidth": 1.8,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def heading(fig, title, subtitle):
    """Left-aligned title + one-line subtitle at the top of the figure."""
    fig.text(0.012, 0.975, title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=9.5, color=INK_2, va="top")


def save(fig, name):
    A.FIGURES.mkdir(parents=True, exist_ok=True)
    path = A.FIGURES / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"saved {path.relative_to(A.ROOT)}")


def strip_box(ax, groups, colors, log=False, rng=None):
    """One column per group: a thin box (median + quartiles) and every game as a dot."""
    rng = rng or np.random.default_rng(0)
    for i, (values, color) in enumerate(zip(groups, colors)):
        values = np.asarray(values, dtype=float)
        ax.boxplot(values, positions=[i], widths=0.5, showfliers=False, whis=(0, 100),
                   medianprops={"color": INK, "linewidth": 2},
                   boxprops={"color": INK_2, "linewidth": 0.9},
                   whiskerprops={"color": BASELINE, "linewidth": 0.9}, capprops={"linewidth": 0})
        x = i + rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(x, values, s=22 if len(values) <= 30 else 12, color=color, alpha=0.85,
                   edgecolors=SURFACE, linewidths=0.7, zorder=3)


def label_medians(ax, groups, log=False, suffixes=None):
    for i, values in enumerate(groups):
        med = np.median(values)
        text = f"{med:,.0f}" + (suffixes[i] if suffixes else "")
        y = np.max(values)
        ax.annotate(text, (i, y), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=9, color=INK, fontweight="semibold")


def games(df, player):
    return df[df.player == player].sort_values("game_seed").lines.values


# --------------------------------------------------------------- RQ1 / RQ4
def fig_training_board():
    e1 = A.per_game("E1_baselines.csv")
    default = e1[e1.player.str.startswith("GA (default")].lines.values
    cats = [("Random", games(e1, "Random"), "Random"), ("Literature", games(e1, "Literature"), "Literature"),
            ("Hand-tuned", games(e1, "Hand-tuned"), "Hand-tuned"),
            ("GA best of\ndefault runs", games(e1, "GA best (default)"), "GA"),
            ("GA, all 10\ndefault runs", default, "GA")]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.98)
    strip_box(ax, [c[1] for c in cats], [PLAYER_COLORS[c[2]] for c in cats])
    label_medians(ax, [c[1] for c in cats])
    ax.set_xticks(range(len(cats)), [c[0] for c in cats], color=INK_2)
    ax.set_ylabel("lines cleared per game")
    ax.yaxis.set_major_formatter(comma)
    ax.set_ylim(-20, 1150)
    heading(fig, "RQ1 - On the 10x10 training board, evolved players match hand-tuned",
            "Lines per test game (20 games; the pooled column has 10 runs x 20 games). Box = median and quartiles, "
            "number = median.")
    save(fig, "fig1_training_board.png")


def fig_standard_board():
    e5 = A.per_game("E5_generalization.csv")
    top = {p.split(":")[0]: p for p in e5.player.unique() if p.startswith("GA top")}
    default = e5[e5.player.str.startswith("GA (default")].lines.values
    cats = [("Literature", games(e5, "Literature"), "Literature"),
            ("Hand-tuned", games(e5, "Hand-tuned"), "Hand-tuned"),
            ("GA, all 10\ndefault runs", default, "GA"),
            ("GA best of\ndefault runs", games(e5, "GA best (default)"), "GA"),
            ("GA top-3\n(validation)", games(e5, top["GA top-3"]), "GA"),
            ("GA top-1\n(validation)", games(e5, top["GA top-1"]), "GA top-1"),
            ("GA top-2\n(validation)", games(e5, top["GA top-2"]), "GA top-2")]
    hand = np.median(games(e5, "Hand-tuned"))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.98)
    strip_box(ax, [c[1] for c in cats], [PLAYER_COLORS[c[2]] for c in cats])
    ax.set_yscale("log")
    label_medians(ax, [c[1] for c in cats], suffixes=[f"  ({np.median(c[1]) / hand:.1f}x)" for c in cats])
    capped = e5[(e5.player == top["GA top-2"]) & e5.reached_cap]
    if len(capped):
        ax.annotate("stopped at the\n250,000-piece cap", (6, capped.lines.iloc[0]), xytext=(-70, -8),
                    textcoords="offset points", fontsize=8, color=INK_2, ha="right", va="center",
                    arrowprops={"arrowstyle": "-", "color": BASELINE, "linewidth": 0.8})
    ax.set_xticks(range(len(cats)), [c[0] for c in cats], color=INK_2)
    ax.set_ylabel("lines cleared per game (log scale)")
    ax.yaxis.set_major_formatter(comma)
    ax.set_ylim(20, 400_000)
    heading(fig, "RQ1 + RQ4 - On the 10x20 board, the top evolved players win by 4.5x and 22x",
            "Lines per test game (20 games). Number = median (x hand-tuned median). Random clears 0 lines (not shown).\n"
            "Top-1/2/3 = the 3 best validation scores of all 80 GA runs, chosen without looking at test games.")
    save(fig, "fig2_standard_board.png")


def fig_generalization():
    e5 = A.per_game("E5_generalization.csv")
    small = pd.concat([A.per_game("E2-E4_champions.csv"), A.per_game("E1_baselines.csv")])
    style = {source: name for name, _, source in A.style_players()}
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.11, right=0.97)
    hand_xy = None
    for player in e5.player.unique():
        key = player.split(": ", 1)[1] if player.startswith("GA top") else player
        if key in ("Random", "GA best (default)"):
            continue
        x, y = games(small, key).mean(), games(e5, player).mean()
        name = style.get(key, "Hand-tuned" if key == "Hand-tuned" else key)
        if key == "Hand-tuned":
            hand_xy = (x, y)
        if name in PLAYER_MARKERS or key == "Literature":
            color = PLAYER_COLORS["Literature" if key == "Literature" else name]
            ax.scatter(x, y, s=90, color=color, marker=PLAYER_MARKERS.get(name, "o"), edgecolors=SURFACE,
                       linewidths=1.5, zorder=4)
            ax.annotate(name if key != "Literature" else "Literature", (x, y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        else:
            ax.scatter(x, y, s=45, color="#86b6ef", edgecolors=SURFACE, linewidths=1.2, zorder=3)
    xs = np.array([80, 600])
    ax.plot(xs, xs * hand_xy[1] / hand_xy[0], color=BASELINE, linewidth=1.2, zorder=1)
    ax.scatter([], [], s=45, color="#86b6ef", label="other GA champions (default runs, top-3)")
    ax.legend(loc="upper left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(80, 600); ax.set_ylim(300, 60_000)
    ax.set_xticks([100, 150, 200, 300, 400, 500])
    ax.grid(True, axis="both")
    # label the reference line, rotated to follow it on screen
    ratio = hand_xy[1] / hand_xy[0]
    (x0, y0), (x1, y1) = ax.transData.transform([(300, 300 * ratio), (550, 550 * ratio)])
    ax.annotate("same gain as hand-tuned (x10)", (550, 550 * ratio), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=MUTED, ha="right", va="bottom", rotation_mode="anchor",
                rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(comma); axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("mean lines on the 10x10 training board (test games)")
    ax.set_ylabel("mean lines on the 10x20 board")
    rho = spearmanr([games(small, p.split(': ', 1)[1] if p.startswith('GA top') else p).mean()
                     for p in e5.player.unique() if p not in ('Random', 'GA best (default)')],
                    [games(e5, p).mean() for p in e5.player.unique() if p not in ('Random', 'GA best (default)')])
    heading(fig, "RQ4 - Skills learned on the small board carry over, and the gaps grow",
            f"Each dot is one player (log-log). Above the grey line = gained more than hand-tuned. "
            f"Spearman rho = {rho.statistic:.2f}.")
    save(fig, "fig3_generalization.png")


# ---------------------------------------------------------------- RQ2 / RQ3
def curve(ax, group, column, color, label, x="generation", smooth=1):
    h = A.load_histories(group)
    table = h.pivot(index="generation", columns="seed", values=column)
    if smooth > 1:
        table = table.rolling(smooth, min_periods=1).mean()
    mean, ci = A.mean_ci(table.values, axis=1)
    xs = table.index.values
    if x == "games":
        xs = (xs + 1) * A.POPULATION.get(group, 50) * A.GAMES_PER_EVAL / 1000
    ax.fill_between(xs, mean - ci, mean + ci, color=color, alpha=0.12, linewidth=0)
    ax.plot(xs, mean, color=color, label=label, solid_capstyle="round")


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=10, color=INK, fontweight="semibold")


def fig_population():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.07, right=0.98, wspace=0.08)
    for (group, label), color in zip(A.EXPERIMENTS["E2"], BLUE_RAMP):
        curve(axes[0], group, "champion_validation", color, label)
        curve(axes[1], group, "champion_validation", color, label, x="games")
    panel_title(axes[0], "a) by generation"); panel_title(axes[1], "b) by training games played (equal compute)")
    axes[0].set_xlabel("generation"); axes[1].set_xlabel("training games played (thousands)")
    axes[0].set_ylabel("best validation score so far")
    axes[0].legend(loc="lower right", ncol=3)
    heading(fig, "RQ2 - Bigger populations find better champions, but mostly because they play more games",
            "Champion's validation score (10 fixed games), mean of 10 runs with 95% confidence band.")
    save(fig, "fig4_population.png")


def fig_mutation():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    for (group, label), color in zip(A.EXPERIMENTS["E3"], BLUE_RAMP):
        curve(axes[0], group, "diversity", color, label)
        curve(axes[1], group, "champion_validation", color, label)
    panel_title(axes[0], "a) population diversity"); panel_title(axes[1], "b) best validation score so far")
    for ax in axes:
        ax.set_xlabel("generation")
    axes[0].set_ylabel("mean distance to population centre")
    axes[1].set_ylabel("lines per game (10 validation games)")
    axes[0].legend(loc="upper right", title="mutation rate (per weight)", title_fontsize=9)
    heading(fig, "RQ2 - Mutation rate controls how quickly the population collapses",
            "Mean of 10 runs with 95% confidence band. Higher mutation keeps the population varied;\n"
            "the champions' test scores do not differ significantly (Kruskal-Wallis p = 0.26).")
    save(fig, "fig5_mutation.png")


def fig_operators():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(top=0.74, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    for (group, label), color in zip(A.EXPERIMENTS["E4"], E4_COLORS):
        curve(axes[0], group, "diversity", color, label)
        curve(axes[1], group, "mean_fitness", color, label, smooth=5)
    panel_title(axes[0], "a) population diversity")
    panel_title(axes[1], "b) population mean fitness (5-generation average)")
    for ax in axes:
        ax.set_xlabel("generation")
    axes[0].set_ylabel("mean distance to population centre")
    axes[1].set_ylabel("lines per game (training games)")
    axes[0].legend(loc="upper right", title="selection + crossover", title_fontsize=9)
    heading(fig, "RQ3 - Uniform crossover keeps twice the diversity; all four settings reach the same level",
            "Mean of 10 runs with 95% confidence band. Arithmetic crossover averages two parents, so the population "
            "shrinks towards its centre.")
    save(fig, "fig6_operators.png")


def fig_champion_scores():
    champions = A.champion_table()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True, gridspec_kw={"width_ratios": [3, 3, 4]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.06, right=0.99, wspace=0.06)
    for ax, (exp, variants) in zip(axes, A.EXPERIMENTS.items()):
        values = [champions[champions.group == g].test_mean.values for g, _ in variants]
        colors = E4_COLORS if exp == "E4" else BLUE_RAMP
        strip_box(ax, values, colors)
        labels = [lab.replace(" + ", " +\n") + ("\n(default)" if g == "default" else "") for g, lab in variants]
        ax.set_xticks(range(len(variants)), labels, color=INK_2, fontsize=8.5)
        p = kruskal(*values).pvalue
        panel_title(ax, f"{exp}  (Kruskal-Wallis p = {A.fmt_p(p)})")
    axes[0].set_ylabel("champion's mean lines on 20 test games (10x10)")
    heading(fig, "RQ2 + RQ3 - Final champions: only population size makes a significant difference",
            "Each dot = the champion of one GA run (10 runs per setting). E2 = population size, E3 = mutation rate, "
            "E4 = selection + crossover.")
    save(fig, "fig7_champion_scores.png")


# ---------------------------------------------------------- weights & styles
def fig_weights():
    champions = pd.read_csv(A.TABLES / "champions.csv")
    weights = np.array([json.loads(w) for w in champions.weights])
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [3, 2]})
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.06, right=0.98, wspace=0.22)
    ax = axes[0]
    rng = np.random.default_rng(1)
    for j in range(6):
        ax.scatter(j + rng.uniform(-0.22, 0.22, len(weights)), weights[:, j], s=14, color="#b9b8b1",
                   edgecolors="none", zorder=2)
    for name, w, _ in A.style_players():
        w = np.asarray(w) / np.linalg.norm(w)
        ax.scatter(np.arange(6), w, s=70, color=PLAYER_COLORS[name], marker=PLAYER_MARKERS[name],
                   edgecolors=SURFACE, linewidths=1.3, zorder=4, label=name)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(range(6), A.FEATURES, color=INK_2)
    ax.set_ylabel("weight (vector scaled to length 1)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
    panel_title(ax, "a) raw weights of all 80 champions (grey) and the 4 style players")

    ax = axes[1]
    rho = spearmanr(champions.bumpiness, champions.test_mean)
    ax.scatter(champions.bumpiness, champions.test_mean, s=24, color=BLUE, alpha=0.8, edgecolors=SURFACE,
               linewidths=0.7)
    ax.set_xlabel("bumpiness penalty (cells of height per unit)")
    ax.set_ylabel("champion's mean test lines (10x10)")
    ax.grid(True, axis="both")
    panel_title(ax, f"b) smoother boards win (rho = {rho.statistic:.2f}, p < 0.001)")
    heading(fig, "Evolved weights: many different-looking vectors, one clear signal",
            "'Holes' is redundant with 'height' and 'lines' (holes = height - cells), so raw weights can look very "
            "different and still mean the same player.")
    save(fig, "fig8_weights.png")


def fig_styles():
    styles = pd.read_csv(A.TABLES / "style_profiles.csv")
    metrics = [("eff. line_value", "value of clearing a line\n(in cells of stack height)", "{:.1f}"),
               ("avg_height", "average stack height\n(cells)", "{:.1f}"),
               ("holes", "holes on the board\n(average)", "{:.1f}"),
               ("multi_line_share", "clears that remove\n2+ lines at once", "{:.0%}"),
               ("10x20 median lines", "median lines per game\non 10x20", "{:,.0f}")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 3.9), sharey=True)
    fig.subplots_adjust(top=0.70, bottom=0.07, left=0.085, right=0.985, wspace=0.35)
    names = list(styles.player)[::-1]
    for ax, (col, title, fmt) in zip(axes, metrics):
        values = styles.set_index("player").loc[names, col].values
        ax.barh(range(len(names)), values, height=0.55, color=[PLAYER_COLORS[n] for n in names])
        for i, v in enumerate(values):
            ax.annotate(fmt.format(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
        ax.set_xlim(0, values.max() * 1.45)
        ax.set_xticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=9.5, color=INK_2, loc="left")
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(range(len(names)), names, color=INK)
    heading(fig, "Playing styles: the best evolved players barely reward line clears - they keep the board low and clean",
            "10x20 board, first 1,000 pieces of 10 test games, measured after every move. "
            "Line value comes from the weights (see weights_and_styles.md).")
    save(fig, "fig9_styles.png")


def main():
    setup()
    fig_training_board()
    fig_standard_board()
    fig_generalization()
    fig_population()
    fig_mutation()
    fig_operators()
    fig_champion_scores()
    fig_weights()
    fig_styles()


if __name__ == "__main__":
    main()
