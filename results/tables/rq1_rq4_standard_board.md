# RQ1 + RQ4 - lines cleared on the standard 10x20 board (E5, 20 test games)

| player | mean | median | 25% | 75% | max | wins vs hand | p vs hand (Holm) | A12 vs hand | median / hand |
|---|---|---|---|---|---|---|---|---|---|
| Random | 0 | 0 | 0 | 0 | 0 | 0/20 | <0.001 | 0.00 | 0.0x |
| Literature | 563 | 468 | 196 | 924 | 1,235 | 2/20 | <0.001 | 0.20 | 0.4x |
| Hand-tuned | 1,805 | 1,244 | 793 | 2,051 | 8,332 | - | - | - | 1.0x |
| GA best of default runs | 2,413 | 1,260 | 1,050 | 2,762 | 7,798 | 11/20 | 1.000 | 0.56 | 1.0x |
| GA, all 10 default runs | 1,792 | 1,132 | 552 | 2,219 | 14,808 | pooled | 1.000 | 0.48 | 0.9x |
| GA top-1: crossover=uniform+selection=roulette, seed 1 | 7,871 | 5,586 | 2,950 | 11,650 | 21,445 | 18/20 | <0.001 | 0.88 | 4.5x |
| GA top-2: mutation_rate=0.15, seed 2 (1 game(s) hit the cap) | 35,072 | 27,268 | 10,124 | 53,640 | 99,995 | 19/20 | <0.001 | 0.96 | 21.9x |
| GA top-3: selection=roulette, seed 3 | 2,350 | 1,451 | 760 | 1,983 | 9,544 | 11/20 | 1.000 | 0.54 | 1.2x |

Same players as on the training board plus the 3 champions with the best VALIDATION score out of all 80 GA runs (chosen without looking at test games). Cap: 250,000 pieces. Tests as above (Holm over 6 comparisons).
