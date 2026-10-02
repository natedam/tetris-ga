"""
The genetic algorithm.

One generation:
  1. Evaluate: every individual plays the SAME few games (same piece seeds),
     fitness = mean number of lines cleared. New seeds every generation, so
     players cannot over-fit to one particular piece sequence.
  2. Validate: the top few individuals also play a FIXED validation set of
     games. The individual with the best validation score over the whole run
     is the run's "champion" (this is what we test and show in the report).
     Why? Training fitness uses only a few games and is noisy, so the top
     individual of a generation is often just lucky. The validation set is
     larger and fixed, so it gives a fairer comparison across generations.
  3. Elitism: the best `elitism` individuals are copied unchanged.
  4. The rest of the new population is created by
     selection -> crossover -> mutation.

Three separate sets of piece seeds are used, so they never overlap:
    training    run_seed*100000 + generation*100 + game   (< 800,000,000)
    validation  800,000,000 + i                          (same for every run)
    test        1,000,000,000 + i                        (used only in experiments)

Everything is seeded, so the same config + seed gives the same run.
"""

import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool

import numpy as np

from ga.operators import CROSSOVERS, SELECTIONS, gaussian_mutation, random_individual
from tetris.agent import LinearAgent, play_game
from tetris.features import NUM_FEATURES

VALIDATION_SEED_START = 800_000_000
TEST_SEED_START = 1_000_000_000


@dataclass
class GAConfig:
    population_size: int = 50
    generations: int = 50
    games_per_eval: int = 3          # training games per individual per generation
    max_pieces: int = 1000           # piece cap per game (None = no cap)
    board_width: int = 10
    board_height: int = 10           # training board (see README: why 10x10)
    selection: str = "tournament"    # "tournament" or "roulette"
    tournament_size: int = 3
    crossover: str = "arithmetic"    # "arithmetic" or "uniform"
    mutation_rate: float = 0.05      # probability that each weight is mutated
    mutation_sigma: float = 0.2      # size of a mutation (std of the Gaussian noise)
    elitism: int = 2                 # best individuals copied unchanged
    validation_games: int = 10       # size of the fixed validation set
    validation_top_k: int = 3        # how many top individuals are validated per generation


# ------------------------------------------------------------------- seeds
def training_seeds(run_seed, generation, n_games):
    """Piece seeds of one generation. The whole population plays the same games."""
    return [run_seed * 100_000 + generation * 100 + g for g in range(n_games)]


def validation_seeds(n_games):
    return [VALIDATION_SEED_START + i for i in range(n_games)]


def final_test_seeds(n_games):
    """Seeds of the held-out test games (never used during evolution)."""
    return [TEST_SEED_START + i for i in range(n_games)]


# ---------------------------------------------------------------- evaluation
def evaluate_weights(weights, seeds, max_pieces, width, height):
    """Mean lines cleared by a weight vector over the given games.
    Module-level function so multiprocessing can send it to worker processes."""
    agent = LinearAgent(weights)
    lines = [play_game(agent, seed=s, max_pieces=max_pieces, width=width, height=height).lines
             for s in seeds]
    return float(np.mean(lines))


def _evaluate_star(args):
    return evaluate_weights(*args)


def evaluate_many(individuals, seeds, config, pool=None):
    """Evaluate a list of weight vectors on the same seeds (in parallel if a pool is given)."""
    jobs = [(ind, seeds, config.max_pieces, config.board_width, config.board_height)
            for ind in individuals]
    if pool is None:
        return [_evaluate_star(job) for job in jobs]
    return pool.map(_evaluate_star, jobs)


# -------------------------------------------------------------------- the GA
def next_generation(population, fitness, order, config, rng):
    """Elitism + (selection -> crossover -> mutation) until the population is full."""
    select = SELECTIONS[config.selection]
    crossover = CROSSOVERS[config.crossover]
    new_population = [population[i].copy() for i in order[:config.elitism]]
    while len(new_population) < config.population_size:
        parent1 = select(population, fitness, rng, config.tournament_size)
        parent2 = select(population, fitness, rng, config.tournament_size)
        child = crossover(parent1, parent2, rng)
        child = gaussian_mutation(child, rng, config.mutation_rate, config.mutation_sigma)
        new_population.append(child)
    return new_population


def run_ga(config, seed, workers=1, verbose=True):
    """Run the GA once.

    Returns (history, champion):
      history  - list of dicts, one per generation (becomes one CSV row each)
      champion - dict with the best-validated weights of the whole run
    """
    rng = np.random.default_rng(seed)
    val_seeds = validation_seeds(config.validation_games)
    population = [random_individual(rng, NUM_FEATURES) for _ in range(config.population_size)]
    champion = {"weights": None, "validation": -1.0, "generation": -1}
    history = []
    start = time.time()
    pool = Pool(workers) if workers > 1 else None
    try:
        for gen in range(config.generations):
            # 1. training fitness (noisy, new games every generation)
            seeds = training_seeds(seed, gen, config.games_per_eval)
            fitness = evaluate_many(population, seeds, config, pool)
            order = list(np.argsort(fitness)[::-1])          # indices, best first

            # 2. validation of the top-k (fixed games -> comparable across generations)
            top = [population[i] for i in order[:config.validation_top_k]]
            val_scores = evaluate_many(top, val_seeds, config, pool)
            best_k = int(np.argmax(val_scores))
            if val_scores[best_k] > champion["validation"]:
                champion = {"weights": top[best_k].tolist(),
                            "validation": val_scores[best_k], "generation": gen}

            row = {
                "generation": gen,
                "best_fitness": fitness[order[0]],
                "mean_fitness": float(np.mean(fitness)),
                "median_fitness": float(np.median(fitness)),
                "std_fitness": float(np.std(fitness)),
                "best_validation": val_scores[best_k],     # best of top-k this generation
                "champion_validation": champion["validation"],
                "diversity": population_diversity(population),
                "elapsed_sec": round(time.time() - start, 2),
            }
            row.update({f"w{i}": float(w) for i, w in enumerate(top[best_k])})
            history.append(row)
            if verbose:
                print(f"  gen {gen:3d}  best {row['best_fitness']:7.1f}  "
                      f"mean {row['mean_fitness']:7.1f}  val {row['best_validation']:7.1f}  "
                      f"champion {champion['validation']:7.1f}  ({row['elapsed_sec']:.0f}s)",
                      flush=True)

            # 3 + 4. next generation (not needed after the last one)
            if gen < config.generations - 1:
                population = next_generation(population, fitness, order, config, rng)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return history, champion


def population_diversity(population):
    """Mean distance of individuals from the population's average weight vector.
    High = varied population (exploring), near 0 = everyone is the same (converged)."""
    matrix = np.array(population)
    return float(np.linalg.norm(matrix - matrix.mean(axis=0), axis=1).mean())


# ---------------------------------------------------------------- config I/O
def config_from_dict(d):
    """Build a GAConfig from a dict (e.g. loaded from YAML). Unknown keys are an error,
    so a typo in a YAML file is caught instead of being silently ignored."""
    unknown = set(d) - set(GAConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown GA settings: {sorted(unknown)}")
    return GAConfig(**d)


def config_to_dict(config):
    return asdict(config)
