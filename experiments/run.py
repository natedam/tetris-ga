"""
Experiment runner.

Usage (from the repository root):
    python experiments/run.py experiments/configs/e2_population.yaml
    python experiments/run.py experiments/configs/e2_population.yaml --workers 8
    python experiments/run.py experiments/configs/e2_population.yaml --quick   # smoke test
    python experiments/run.py all                                              # E2,E3,E4,E1,E5

Two kinds of experiment files (see experiments/configs/):

  type: ga        Run the GA several times (repeats) for every variant.
                  Each run is saved in results/runs/<run name>/:
                      seed_<k>.csv            one row per generation
                      seed_<k>_champion.json  the run's best-validated weights
                  The run name lists only the settings that differ from
                  default.yaml, e.g. "population_size=20" or "default". So
                  identical settings in different experiments (E2's pop50 and
                  E3's mut0.05 are both "default") are run only ONCE.
                  Finished runs are skipped, so an interrupted experiment can
                  simply be started again and it continues where it stopped.

  type: evaluate  Play held-out test games with fixed players and save one row
                  per game in results/<experiment name>.csv.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

# Make "import tetris" / "import ga" work when this file is run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ga.evolution import config_from_dict, final_test_seeds, run_ga  # noqa: E402
from tetris.agent import LinearAgent, play_game  # noqa: E402
from tetris.baselines import HAND_TUNED_WEIGHTS, LITERATURE_WEIGHTS, RandomAgent  # noqa: E402

CONFIG_DIR = ROOT / "experiments" / "configs"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
ALL_EXPERIMENTS = ["e2_population", "e3_mutation", "e4_operators", "e1_baselines",
                   "e5_generalization"]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def default_settings():
    return load_yaml(CONFIG_DIR / "default.yaml")


def run_name(settings):
    """Readable, unique name: the settings that differ from default.yaml."""
    defaults = default_settings()
    diffs = [f"{k}={v}" for k, v in sorted(settings.items()) if defaults.get(k) != v]
    return "+".join(diffs) if diffs else "default"


# ---------------------------------------------------------------- GA experiments
def run_ga_experiment(exp, workers, quick=False, repeats=None):
    repeats = repeats or exp["repeats"]
    for variant, overrides in exp["variants"].items():
        settings = {**default_settings(), **(overrides or {})}
        name = run_name(settings)
        if quick:
            settings.update(generations=5, population_size=min(settings["population_size"], 12),
                            validation_games=3)
            name = "quick_" + name
        config = config_from_dict(settings)
        out_dir = RUNS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(settings, f, indent=2)

        for seed in range(repeats):
            csv_path = out_dir / f"seed_{seed}.csv"
            if csv_path.exists():
                print(f"[{variant}] {name} seed {seed}: already done, skipping")
                continue
            print(f"[{variant}] {name} seed {seed}: running")
            history, champion = run_ga(config, seed=seed, workers=workers)
            # Write the champion first and the CSV last: the CSV marks the run as finished.
            with open(out_dir / f"seed_{seed}_champion.json", "w") as f:
                json.dump({"name": f"GA ({name}, seed {seed})", **champion}, f, indent=2)
            pd.DataFrame(history).to_csv(csv_path, index=False)


# ------------------------------------------------------------ evaluate experiments
def load_champions(group):
    """All champion JSONs of a run group, e.g. group = "default"."""
    files = sorted((RUNS_DIR / group).glob("seed_*_champion.json"))
    if not files:
        raise FileNotFoundError(f"No champions in {RUNS_DIR / group} - run the GA experiments first")
    champions = []
    for path in files:
        with open(path) as f:
            champions.append(json.load(f))
    return champions


def players_from_spec(spec, group_prefix=""):
    """Turn a player spec from the YAML file into a list of (name, weights or None).
    group_prefix is "quick_" in --quick mode, so quick GA runs are used."""
    if spec == "random":
        return [("Random", None)]
    if spec == "hand":
        return [("Hand-tuned", HAND_TUNED_WEIGHTS)]
    if spec == "literature":
        return [("Literature", LITERATURE_WEIGHTS)]
    if spec.startswith("ga:"):                 # the single best champion of a group
        group = group_prefix + spec[3:]
        best = max(load_champions(group), key=lambda c: c["validation"])
        return [(f"GA best ({group})", best["weights"])]
    if spec.startswith("ga-all:"):             # every champion of a group
        group = group_prefix + spec[7:]
        return [(c["name"], c["weights"]) for c in load_champions(group)]
    raise ValueError(f"Unknown player spec: {spec}")


def _play_one(job):
    name, weights, seed, width, height, max_pieces = job
    agent = RandomAgent(seed) if weights is None else LinearAgent(weights)
    t = time.time()
    r = play_game(agent, seed=seed, max_pieces=max_pieces, width=width, height=height)
    return {
        "player": name, "game_seed": seed, "lines": r.lines, "pieces": r.pieces,
        "score": r.score, "singles": r.clear_counts[1], "doubles": r.clear_counts[2],
        "triples": r.clear_counts[3], "tetrises": r.clear_counts[4],
        "reached_cap": max_pieces is not None and r.pieces >= max_pieces,
        "seconds": round(time.time() - t, 2),
        "weights": None if weights is None else json.dumps([round(w, 6) for w in weights]),
    }


def run_evaluate_experiment(exp, workers, quick=False):
    n_games = 3 if quick else exp["test_games"]
    max_pieces = 2000 if quick else exp["max_pieces"]
    jobs = []
    for spec in exp["players"]:
        for name, weights in players_from_spec(spec, "quick_" if quick else ""):
            for seed in final_test_seeds(n_games):
                jobs.append((name, weights, seed, exp["board_width"], exp["board_height"],
                             max_pieces))
    print(f"{exp['name']}: {len(jobs)} games on {exp['board_width']}x{exp['board_height']}")
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_play_one, jobs, chunksize=1)
    else:
        rows = [_play_one(job) for job in jobs]
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{'quick_' if quick else ''}{exp['name']}.csv"
    df.to_csv(out, index=False)
    print(df.groupby("player")["lines"].agg(["mean", "median", "std", "max"]).round(1))
    print(f"Saved {out}")


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Run Tetris-GA experiments.")
    parser.add_argument("config", nargs="?", help="path to an experiment YAML file, or 'all'")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="parallel processes (default: all CPU cores)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke-test version (5 generations, 1 repeat)")
    parser.add_argument("--repeats", type=int, help="override the number of repeats")
    args = parser.parse_args()

    if args.config is None:
        parser.print_help()
        print("\nAvailable experiments:")
        for path in sorted(CONFIG_DIR.glob("e*.yaml")):
            print(f"  {path.relative_to(ROOT)}")
        return

    paths = ([CONFIG_DIR / f"{e}.yaml" for e in ALL_EXPERIMENTS] if args.config == "all"
             else [Path(args.config)])
    for path in paths:
        exp = load_yaml(path)
        print(f"\n=== {exp['name']} ({path.name}) ===")
        start = time.time()
        if exp["type"] == "ga":
            run_ga_experiment(exp, args.workers, args.quick,
                              repeats=1 if args.quick else args.repeats)
        elif exp["type"] == "evaluate":
            run_evaluate_experiment(exp, args.workers, args.quick)
        else:
            raise ValueError(f"Unknown experiment type: {exp['type']}")
        print(f"=== {exp['name']} finished in {(time.time() - start) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
