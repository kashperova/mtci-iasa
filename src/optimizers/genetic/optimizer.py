from copy import deepcopy
from typing import Optional, Callable, Tuple

from torch import nn
from torch import Tensor

from optimizers.base import BaseOptimizer
from optimizers.genetic.operators import GeneticOperators
from optimizers.genetic.population import Population


class GeneticOptimizer(BaseOptimizer):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        tournament_size: int,
        patience: Optional[int] = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = deepcopy(model)
        self.loss_fn = loss_fn
        self.patience = patience
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.population = Population(
            size=self.population_size, model=self.model, loss_fn=self.loss_fn
        )
        self.operators = GeneticOperators(loss_fn=self.loss_fn)
        self.runs = 0

    def run(self, x: Tensor, y: Tensor) -> Tuple[nn.Module, float]:
        if self.runs == 0:
            self.population.evaluate(x, y)

        selected = self.operators.select(self.population, self.tournament_size)
        next_generation = []
        for i in range(0, self.population_size, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < self.population_size else selected[i]
            child1, child2 = self.operators.crossover(
                parent1, parent2, self.crossover_rate
            )
            child1 = self.operators.mutate(child1, self.mutation_rate)
            child2 = self.operators.mutate(child2, self.mutation_rate)
            next_generation.extend([child1, child2])

        self.population.update(next_generation)
        self.population.evaluate(x, y)

        self.runs += 1

        return self.population.best
