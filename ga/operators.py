"""
Genetic operators. An individual is a NumPy vector of 6 real weights
(one per board feature), always scaled to length 1.

Why normalise? Multiplying all weights by the same positive number does not
change which move gets the highest score, so only the DIRECTION of the vector
matters. Keeping every vector at length 1 removes this redundancy and keeps the
GA's search space small (the surface of a 6-D sphere).

Every function takes an explicit random generator `rng`
(numpy.random.Generator) so that runs are reproducible.
"""

import numpy as np


def normalize(weights):
    norm = np.linalg.norm(weights)
    if norm == 0:                       # practically impossible, but be safe
        return weights
    return weights / norm


def random_individual(rng, n_genes=6):
    """A random direction: each weight ~ Normal(0, 1), then normalised.
    (This gives a uniformly random point on the sphere, so no sign is favoured.)"""
    return normalize(rng.normal(0.0, 1.0, size=n_genes))


# ------------------------------------------------------------------ selection
def tournament_selection(population, fitness, rng, tournament_size=3):
    """Pick `tournament_size` random individuals, return the fittest of them."""
    contestants = rng.choice(len(population), size=tournament_size, replace=False)
    winner = max(contestants, key=lambda i: fitness[i])
    return population[winner]


def roulette_selection(population, fitness, rng, tournament_size=None):
    """Fitness-proportional selection: probability of being picked = f_i / sum(f).
    (tournament_size is ignored; it is accepted only so both selections share a signature.)"""
    fitness = np.asarray(fitness, dtype=float)
    total = fitness.sum()
    if total <= 0:                      # e.g. generation 0 where nobody clears a line
        return population[rng.integers(len(population))]
    return population[rng.choice(len(population), p=fitness / total)]


# ------------------------------------------------------------------ crossover
def arithmetic_crossover(parent1, parent2, rng):
    """Child is a random blend of the parents: a*p1 + (1-a)*p2, with a ~ U(0, 1)."""
    a = rng.random()
    return normalize(a * parent1 + (1 - a) * parent2)


def uniform_crossover(parent1, parent2, rng):
    """Each weight is copied from parent1 or parent2 with probability 1/2."""
    take_from_first = rng.random(len(parent1)) < 0.5
    return normalize(np.where(take_from_first, parent1, parent2))


# ------------------------------------------------------------------- mutation
def gaussian_mutation(individual, rng, rate=0.05, sigma=0.2):
    """Each weight, with probability `rate`, gets Gaussian noise N(0, sigma) added."""
    child = individual.copy()
    mask = rng.random(len(child)) < rate
    child[mask] += rng.normal(0.0, sigma, size=mask.sum())
    return normalize(child)


SELECTIONS = {"tournament": tournament_selection, "roulette": roulette_selection}
CROSSOVERS = {"arithmetic": arithmetic_crossover, "uniform": uniform_crossover}
