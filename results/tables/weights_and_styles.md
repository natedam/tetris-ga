# Evolved weights and playing styles

## Raw weights (normalised to length 1) and effective weights

Effective weights remove the redundant 'holes' weight and are measured in cells of stack height (see analysis.effective_weights): line_value = net reward for clearing one line.

| player | run | w height | w holes | w bumpiness | w lines | w max height | w wells | eff. line_value | eff. bumpiness | eff. max_height | eff. wells |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hand-tuned | hand-tuned weights | -0.33 | -0.66 | -0.13 | 0.66 | -0.07 | -0.07 | 4.00 | 0.13 | 0.07 | 0.07 |
| GA greedy | GA (default, seed 0) | -0.98 | 0.02 | -0.09 | -0.03 | -0.14 | -0.14 | 10.18 | 0.09 | 0.14 | 0.15 |
| GA top-1 | GA (crossover=uniform+selection=roulette, seed 1) | -0.08 | -0.85 | -0.13 | -0.50 | 0.03 | -0.09 | 0.37 | 0.14 | -0.03 | 0.10 |
| GA top-2 | GA (mutation_rate=0.15, seed 2) | 0.01 | -0.87 | -0.09 | 0.42 | -0.15 | -0.18 | 0.41 | 0.10 | 0.18 | 0.20 |

## How they play (10x20 board, first 1,000 pieces of 10 test games, averaged after every move)

| player | avg_height | max_height | holes | bumpiness | multi_line_share | lines_per_piece | 10x20 median lines |
|---|---|---|---|---|---|---|---|
| Hand-tuned | 4.46 | 6.25 | 2.97 | 8.31 | 0.10 | 0.39 | 1,244.50 |
| GA greedy | 4.12 | 5.94 | 2.68 | 8.28 | 0.10 | 0.40 | 1,912.00 |
| GA top-1 | 3.32 | 5.86 | 1.15 | 8.98 | 0.17 | 0.40 | 5,586.00 |
| GA top-2 | 3.24 | 5.08 | 1.57 | 7.98 | 0.11 | 0.40 | 27,267.50 |

## Which effective weights go with a good champion? (all 80 champions, test score on 10x10)

| effective weight | Spearman rho with test score | p (Holm) |
|---|---|---|
| line_value | -0.09 | 0.590 |
| bumpiness | 0.41 | <0.001 |
| max_height | -0.12 | 0.590 |
| wells | -0.25 | 0.067 |
