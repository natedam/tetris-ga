# RQ4 - the same players on the training board (10x10) and the standard board (10x20)

| player | 10x10 mean | 10x10 median | 10x20 mean | 10x20 median | 10x20 / 10x10 (means) |
|---|---|---|---|---|---|
| top-2: GA (mutation_rate=0.15, seed 2) | 442.1 | 241.0 | 35,071.7 | 27,267.5 | 79.3 |
| top-1: GA (crossover=uniform+selection=roulette, seed 1) | 261.1 | 212.0 | 7,870.8 | 5,586.0 | 30.1 |
| GA (default, seed 0) | 188.9 | 143.5 | 2,322.2 | 1,912.0 | 12.3 |
| GA (default, seed 9) | 159.2 | 120.5 | 3,027.2 | 1,674.0 | 19.0 |
| GA (default, seed 2) | 164.8 | 128.0 | 1,859.8 | 1,650.0 | 11.3 |
| top-3: GA (selection=roulette, seed 3) | 139.2 | 120.0 | 2,350.4 | 1,451.0 | 16.9 |
| GA (default, seed 1) | 147.8 | 114.0 | 2,091.3 | 1,408.5 | 14.2 |
| GA (default, seed 8) | 273.3 | 122.5 | 2,412.6 | 1,260.5 | 8.8 |
| Hand-tuned | 177.3 | 143.5 | 1,804.9 | 1,244.5 | 10.2 |
| GA (default, seed 6) | 152.6 | 123.0 | 1,566.1 | 1,111.5 | 10.3 |
| GA (default, seed 3) | 110.1 | 65.0 | 1,222.3 | 988.0 | 11.1 |
| GA (default, seed 5) | 207.6 | 140.0 | 1,164.8 | 870.5 | 5.6 |
| GA (default, seed 4) | 171.2 | 144.0 | 1,343.0 | 826.5 | 7.8 |
| GA (default, seed 7) | 147.3 | 140.5 | 905.9 | 690.5 | 6.2 |
| Literature | 96.1 | 82.5 | 563.1 | 468.5 | 5.9 |

Rank correlation between the two boards over these 15 players: Spearman rho = 0.58 (p = 0.024) for means, 0.35 (p = 0.201) for medians.
