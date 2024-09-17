import random

import numpy as np
from typing import Callable, List, Dict, Any, Optional

from optimizers.base import BaseOptimizer


class GeneticOptimizer(BaseOptimizer):
    """
    ====== Hyperparams ======

    population_size: int
    mutation_rate: float

    """

    def __init__(self, population_size: int, mutation_rate: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self._population = list()
        self._num_runs = 0

    def init_population(
        self, weights_init: Callable, init_params: Dict[str, Any]
    ) -> List[np.array]:
        for i in range(self.population_size):
            self._population[i] = weights_init(**init_params)

        return self._population

    def get_scores(self, loss: Callable, x: np.array, y: np.array) -> List[float]:
        return [loss(gen, x, y) for gen in self._population]

    @classmethod
    def crossover(cls, parent1: np.array, parent2: np.array) -> np.array:
        assert len(parent1) == len(parent2), "parents have different size"

        start, end = sorted(random.sample(range(len(parent1)), 2))
        child_p1 = parent1[start:end]
        child_p2 = [gene for gene in parent2 if gene not in child_p1]

        return np.concatenate((child_p1, child_p2))

    def breed(self, population: np.array) -> np.array:
        random.shuffle(population)

        for i in range(len(population) // 2):
            children = self.crossover(population[i], population[i + 1])
            population.append(children)

        return population

    def mutate(self, population: np.array) -> np.array:
        random.shuffle(population)
        num_creatures = len(population)
        num_features = len(population[0])

        mutated = random.sample(range(num_creatures), int(self.mutation_rate * num_creatures))

        for i in mutated:
            creature = population[i]
            for _ in range(int(self.mutation_rate * num_features)):
                idx1, idx2 = random.sample(range(num_features), 2)
                creature[idx1], creature[idx2] = creature[idx2], creature[idx1]

        return population

    def select(self, population: np.array, scores: List[float]) -> np.array:
        best = np.array(scores).argsort()[:self.population_size].tolist()
        return np.array(population)[best]

    def run(
        self,
        loss: Callable,
        x: np.array,
        y: np.array,
        weights_init: Optional[Callable] = None,
        init_params: Optional[Dict[str, Any]] = None,
        **kwargs
    )  -> np.array:
        if len(self._population) == 0:
            if weights_init is None or init_params is None:
                raise ValueError("weight_init(**init_params) is needed to create the population.")

            self._population = self.init_population(
                weights_init=weights_init, init_params=init_params
            )

        population = self.mutate(self._population)
        population = self.breed(population)
        scores = self.get_scores(loss, x, y)
        population = self.select(population, scores)

        self._num_runs += 1

        return population
