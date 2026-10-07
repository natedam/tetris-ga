# Design checks (Methods)

Hand-tuned player on the 10x20 board with a 500-piece cap (20 test games): 175-198 lines per game, mean 194.2. 500 pieces = 2,000 cells, so at most 200 lines are possible -> the score saturates.

Validation score (10 fixed games) vs final test score (20 test games) of each run's champion, 10x10:

| runs | Spearman rho (validation vs test) | p | mean validation | mean test | validation > test |
|---|---|---|---|---|---|
| all 80 runs | 0.33 | 0.003 | 248.2 | 175.6 | 96% |
| default setting (10 runs) | 0.76 | 0.011 | 243.8 | 172.3 | 90% |

Champions found already in generation 0 (a random initial individual): 6 of 80 runs (crossover=uniform seed 5, crossover=uniform+selection=roulette seed 5, default seed 5, mutation_rate=0.01 seed 5, population_size=100 seed 5, selection=roulette seed 5).
