# RQ1 - lines cleared on the 10x10 training board (E1, 20 test games)

| player | mean | median | 25% | 75% | max | wins vs hand | p vs hand (Holm) | A12 vs hand |
|---|---|---|---|---|---|---|---|---|
| Random | 0.1 | 0.0 | 0.0 | 0.0 | 1 | 0/20 | <0.001 | 0.00 |
| Literature | 96.1 | 82.5 | 58.2 | 109.2 | 227 | 6/20 | 0.041 | 0.29 |
| Hand-tuned | 177.3 | 143.5 | 86.0 | 228.2 | 540 | - | - | - |
| GA best of default runs | 273.3 | 122.5 | 69.2 | 363.5 | 1,041 | 9/20 | 1.000 | 0.51 |
| GA, all 10 default runs | 172.3 | 122.5 | 66.8 | 218.0 | 1,041 | pooled | 1.000 | 0.46 |

Paired Wilcoxon test (same 20 games) for single players, Mann-Whitney U for the pooled 10 runs (200 games); p-values Holm-corrected over the 4 comparisons. A12 = probability that the player beats hand-tuned in a random game pair.
