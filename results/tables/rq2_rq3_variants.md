# RQ2 + RQ3 - GA settings (E2 population size, E3 mutation rate, E4 operators)

## E2

| variant | test mean | SD over runs | median of runs | validation | pop. mean (last 10 gens) | gens to 90% | diversity (gens 25-49) | min / run |
|---|---|---|---|---|---|---|---|---|
| pop 20 | 149.53 | 26.37 | 145.90 | 235.60 | 149.75 | 21.60 | 0.09 | 2.21 |
| pop 50 (default) | 172.28 | 44.01 | 162.05 | 243.76 | 144.23 | 16.40 | 0.10 | 3.52 |
| pop 100 | 187.56 | 24.96 | 194.70 | 254.30 | 148.80 | 14.00 | 0.10 | 8.25 |

Kruskal-Wallis across variants: test score p = 0.022, validation p = 0.123, population mean p = 0.450, gens to 90% p = 0.315, diversity p = 0.030

| comparison | test-score A12 | Mann-Whitney p | paired Wilcoxon p | validation p | diversity A12 | diversity p |
|---|---|---|---|---|---|---|
| pop 20 vs pop 50 | 0.30 | 0.140 (Holm 0.225) | 0.131 | 0.345 | 0.38 | 0.385 |
| pop 100 vs pop 50 | 0.71 | 0.112 (Holm 0.225) | 0.301 | 0.290 | 0.74 | 0.076 |
| pop 100 vs pop 20 | 0.83 | 0.013 (Holm 0.038) | 0.010 | 0.049 | 0.83 | 0.014 |

Equal compute - average champion validation score after the same number of training games:

| training games | pop 20 | pop 50 | pop 100 |
|---|---|---|---|
| 3,000 | 235.6 | 217.2 | 218.6 |
| 7,500 | - | 243.8 | 239.2 |

## E3

| variant | test mean | SD over runs | median of runs | validation | pop. mean (last 10 gens) | gens to 90% | diversity (gens 25-49) | min / run |
|---|---|---|---|---|---|---|---|---|
| mut 0.01 | 171.24 | 21.67 | 172.35 | 222.32 | 151.04 | 18.30 | 0.04 | 3.54 |
| mut 0.05 (default) | 172.28 | 44.01 | 162.05 | 243.76 | 144.23 | 16.40 | 0.10 | 3.52 |
| mut 0.15 | 207.56 | 85.10 | 176.88 | 265.36 | 141.79 | 16.30 | 0.19 | 3.57 |

Kruskal-Wallis across variants: test score p = 0.264, validation p = 0.008, population mean p = 0.364, gens to 90% p = 0.970, diversity p = <0.001

| comparison | test-score A12 | Mann-Whitney p | paired Wilcoxon p | validation p | diversity A12 | diversity p |
|---|---|---|---|---|---|---|
| mut 0.01 vs mut 0.05 | 0.57 | 0.597 (Holm 0.689) | 0.910 | 0.059 | 0.00 | <0.001 |
| mut 0.15 vs mut 0.05 | 0.71 | 0.112 (Holm 0.337) | 0.375 | 0.089 | 1.00 | <0.001 |
| mut 0.15 vs mut 0.01 | 0.63 | 0.345 (Holm 0.689) | 0.160 | 0.006 | 1.00 | <0.001 |

## E4

| variant | test mean | SD over runs | median of runs | validation | pop. mean (last 10 gens) | gens to 90% | diversity (gens 25-49) | min / run |
|---|---|---|---|---|---|---|---|---|
| tournament + arithmetic (default) | 172.28 | 44.01 | 162.05 | 243.76 | 144.23 | 16.40 | 0.10 | 3.52 |
| tournament + uniform | 165.24 | 30.62 | 162.95 | 253.74 | 146.55 | 18.10 | 0.17 | 3.75 |
| roulette + arithmetic | 186.46 | 57.43 | 173.75 | 254.64 | 145.99 | 23.90 | 0.11 | 4.55 |
| roulette + uniform | 164.93 | 44.66 | 148.70 | 256.16 | 142.78 | 15.70 | 0.20 | 4.61 |

Kruskal-Wallis across variants: test score p = 0.679, validation p = 0.684, population mean p = 0.791, gens to 90% p = 0.346, diversity p = <0.001

| comparison | test-score A12 | Mann-Whitney p | paired Wilcoxon p | validation p | diversity A12 | diversity p |
|---|---|---|---|---|---|---|
| tournament + uniform vs tournament + arithmetic | 0.47 | 0.820 (Holm 1.000) | 0.910 | 0.325 | 1.00 | <0.001 |
| roulette + arithmetic vs tournament + arithmetic | 0.57 | 0.597 (Holm 1.000) | 0.445 | 0.325 | 0.74 | 0.076 |
| roulette + uniform vs tournament + arithmetic | 0.40 | 0.450 (Holm 1.000) | 0.301 | 0.384 | 1.00 | <0.001 |

Test score = each run's champion on the 20 test games (10x10), averaged per run; 10 runs per variant. Runs with the same seed share their initial population and training games, so the Wilcoxon test pairs them. Mann-Whitney p-values on the test score are Holm-corrected within each experiment; validation p and diversity p are uncorrected Mann-Whitney tests.
