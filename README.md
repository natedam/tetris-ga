# Evolving a Tetris Player with a Genetic Algorithm

**Netanel Demalash**

Course: Topics in Applications of Computer Science (Prof. Moshe Sipper)

> **Work in progress.** The full research report (Introduction, Methods,
> Experiment Design, Results, Conclusions) will be written here once the
> experiments are finished.

A Tetris player chooses every move with a linear evaluation function over six
board features. A genetic algorithm evolves the six weights. The project asks
whether evolved players beat random and hand-tuned players, how GA settings
affect the result, and whether players evolved on a small board still play
well on the standard board.

## Code structure

```
tetris/       engine.py (game), features.py, agent.py (linear player), baselines.py
ga/           operators.py (selection/crossover/mutation), evolution.py (the GA)
experiments/  configs/*.yaml (one file per experiment), run.py (runner)
results/      CSV results, champion weights, figures, animations
visualize.py  side-by-side GIF animations of players
tests/        unit tests (python -m pytest)
```

## How to run

Requires Python 3.11+.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)   |   source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt

python -m pytest                                              # unit tests
python experiments/run.py all --quick                         # 1-minute smoke test of everything
python experiments/run.py experiments/configs/e2_population.yaml   # one experiment
python experiments/run.py all                                 # all experiments (hours)
python visualize.py random hand literature --out results/demo.gif
```
