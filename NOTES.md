# My notes on the code

Short summary of every file (so I can explain it). Under each "My notes" I
write 2–3 lines in my own words after reading the file.

## tetris/engine.py — the game
Board = NumPy array, row 0 is the bottom. A move is (rotation, column); the
piece is hard-dropped. `possible_moves()` lists legal moves, `simulate(move)`
says what would happen without changing the game (used by the AI to look at
every option), `play(move)` really does it. Pieces come from a seeded random
generator, so the same seed = the same pieces for every player.
Speed trick: for most moves only the column heights change, so the engine
does not copy the whole board.

My notes:

## tetris/features.py — how the AI "sees" a board
Six numbers: aggregate height, holes, bumpiness, lines cleared, max height,
wells. Holes are computed with a trick: holes = (sum of heights) − (filled cells).

My notes:

## tetris/agent.py — the AI player
score(move) = w1·f1 + … + w6·f6, play the move with the highest score. The 6
weights are the player's "DNA". `play_game()` plays one full game.

My notes:

## tetris/baselines.py — players to compare against
Random player, hand-tuned weights (my common-sense guess), and published
weights from Yiyuan Lee (2013).

My notes:

## ga/operators.py — selection, crossover, mutation
Individual = 6 weights scaled to length 1 (only the direction matters).
Tournament / roulette selection, arithmetic / uniform crossover, Gaussian mutation.

My notes:

## ga/evolution.py — the genetic algorithm
Each generation: everyone plays the same 3 games → fitness = mean lines. The
top 3 also play 10 fixed "validation" games; the best-validated player of the
whole run is the "champion". Then elitism + selection → crossover → mutation.
Training, validation and test games use different seeds (no overlap).

My notes:

## experiments/run.py + experiments/configs/*.yaml — the experiments
Each YAML file describes one experiment (E1–E5). `type: ga` runs the GA 10
times per variant; `type: evaluate` plays test games with fixed players.
Results go to results/. Finished runs are skipped, so it can be restarted.

My notes:

## experiments/analysis.py — shared analysis helpers
Loads the saved results (run histories, champions, test games). Also:
`effective_weights()` (removes the redundant holes weight, see below),
A12 effect size, Holm correction, confidence intervals, `style_profile()`
(how high / how many holes a player keeps the board) and `style_players()`
(Hand-tuned, GA greedy, GA top-1, GA top-2 — chosen by fixed rules, never by test score).

My notes:

## experiments/stats.py — statistical tests → results/tables/
Wilcoxon / Mann-Whitney / Kruskal-Wallis tests with Holm correction and A12
effect sizes for every research question; one Markdown table file per RQ, plus
design_checks.md (numbers quoted in the README's Methods section).

My notes:

## experiments/plots.py — figures → results/figures/
Nine figures (fig1–fig9), one per finding: boxes + dots for test games,
mean ± 95% CI curves for GA runs, weights and playing styles.

My notes:

## experiments/animate.py — animations → results/animations/
Three GIFs on the first test game: the 4 style players on 10×20, a survival
race on 10×10, and one GA run "growing up" (best player after generation 0, 2, 5, 49).

My notes:

## visualize.py — animations
Players play the same pieces side by side → GIF. Under each board: lines,
pieces, stack height, holes, a small "holes over time" chart and the
player's weights as a bar chart.

My notes:

## tests/ — unit tests
`python -m pytest` checks the engine (drops, line clears, game over), the
features (compared with a slow direct count), the GA (reproducible, improves),
and the analysis (holes weight really is redundant, A12, Holm, board counters).

My notes:

## Things I found out along the way
- A 500-piece cap on the 10×20 board was useless as a fitness: even my
  hand-tuned player cleared 175–198 (mean 194) of the ~200 possible lines, so the GA had
  nothing to improve. → Train on 10×10 (players do lose there), test on 10×20 (RQ4).
- "holes" carries no extra information when "aggregate height" and "lines
  cleared" are also features: holes = height − cells, and cells after a move =
  cells before + 4 − 10·lines. So only 5 of the 6 weights really matter.
