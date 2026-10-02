"""Unit tests for the genetic operators and the GA loop."""

import numpy as np

from ga.evolution import GAConfig, run_ga, training_seeds, validation_seeds
from ga.operators import (arithmetic_crossover, gaussian_mutation, normalize,
                          random_individual, roulette_selection, tournament_selection,
                          uniform_crossover)


def unit_length(v):
    return abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_random_individual_has_unit_length():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert unit_length(random_individual(rng))


def test_normalize():
    assert unit_length(normalize(np.array([3.0, 4.0, 0, 0, 0, 0])))


def test_crossovers_give_unit_vectors_between_parents():
    rng = np.random.default_rng(1)
    p1, p2 = random_individual(rng), random_individual(rng)
    for _ in range(20):
        a = arithmetic_crossover(p1, p2, rng)
        u = uniform_crossover(p1, p2, rng)
        assert unit_length(a) and unit_length(u)


def test_uniform_crossover_copies_genes_from_parents():
    rng = np.random.default_rng(2)
    p1, p2 = np.ones(6), -np.ones(6)
    child = uniform_crossover(p1, p2, rng) * np.sqrt(6)      # undo the normalisation
    assert np.allclose(np.abs(child), 1.0)


def test_mutation_rate_zero_changes_nothing():
    rng = np.random.default_rng(3)
    ind = random_individual(rng)
    assert np.allclose(gaussian_mutation(ind, rng, rate=0.0), ind)


def test_mutation_rate_one_changes_every_weight():
    rng = np.random.default_rng(4)
    ind = random_individual(rng)
    mutated = gaussian_mutation(ind, rng, rate=1.0, sigma=0.5)
    assert unit_length(mutated)
    assert not np.allclose(mutated, ind)


def test_tournament_with_whole_population_picks_the_best():
    rng = np.random.default_rng(5)
    population = [np.array([float(i)]) for i in range(5)]
    fitness = [3, 9, 1, 4, 2]
    winner = tournament_selection(population, fitness, rng, tournament_size=5)
    assert winner[0] == 1.0


def test_roulette_never_picks_zero_fitness():
    rng = np.random.default_rng(6)
    population = [np.array([float(i)]) for i in range(3)]
    fitness = [0, 5, 0]
    for _ in range(50):
        assert roulette_selection(population, fitness, rng)[0] == 1.0


def test_seed_sets_do_not_overlap():
    train = {s for run in range(10) for gen in range(100) for s in training_seeds(run, gen, 5)}
    assert not train & set(validation_seeds(10))


def test_ga_is_reproducible_and_improves():
    config = GAConfig(population_size=10, generations=4, games_per_eval=2, max_pieces=150,
                      validation_games=2)
    history1, champion1 = run_ga(config, seed=0, verbose=False)
    history2, champion2 = run_ga(config, seed=0, verbose=False)
    assert history1[-1]["best_fitness"] == history2[-1]["best_fitness"]
    assert champion1["weights"] == champion2["weights"]
    assert len(history1) == 4
    assert history1[-1]["mean_fitness"] > history1[0]["mean_fitness"]
