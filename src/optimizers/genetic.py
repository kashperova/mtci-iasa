from copy import deepcopy

from typing import List, Optional

import torch
from torch import nn
from torch import Tensor

from optimizers.base import BaseOptimizer


class GeneticOptimizer(BaseOptimizer):
    """
    ====== Hyperparams ======

    population_size: int
    mutation_rate: float
    """

    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        model: nn.Module,
        loss: nn.functional,
        mutation_decay: Optional[float] = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.loss = loss
        self.mutation_decay = mutation_decay
        self._scores = torch.zeros(self.population_size)
        self._cum_sum = torch.zeros(self.population_size)
        self._population = self.init_population(model)
        self._gen_num = 0

    @property
    def best_model(self) -> nn.Module:
        return self.select()[0]

    def init_population(self, model: nn.Module) -> List[nn.Module]:
        return [deepcopy(model).eval() for _ in range(self.population_size)]

    def set_fitness(self, x: Tensor, y: Tensor):
        scores = torch.tensor([1 / (self.loss(gen(x), y).item() + 1e-6) for gen in self._population])
        self._scores = scores / (scores.sum() + 1e-6)
        self._cum_sum = scores.cumsum(dim=0)

    def roulette_wheel_select(self, parent: int) -> nn.Module:
        while True:
            prob = torch.rand(1).item()
            idx = torch.searchsorted(self._cum_sum, prob)
            if idx != parent and idx < len(self._population):
                return self._population[idx]

    @classmethod
    def crossover(cls, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = deepcopy(parent1)

        for param1, param2, param_child in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
            swap_mask = (torch.rand_like(param1) > 0.5)
            param_child.data[swap_mask] = param2.data[swap_mask]
            param_child.data[~swap_mask] = param1.data[~swap_mask]

        return child

    def breed(self):
        for i in range(self.population_size):
            parent1 = self._population[i]
            parent2 = self.roulette_wheel_select(i)

            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)

            # todo: ask whether it's correct to add new generation
            #  to previous population (roulette selection with new children)
            self._population.append(child1)
            self._population.append(child2)

    def mutate(self):
        decay_factor = torch.exp(torch.tensor(-self.mutation_decay * self._gen_num))
        for model in self._population:
            for param in model.parameters():
                noise = torch.empty_like(param).uniform_(-self.mutation_rate, self.mutation_rate)
                xi = noise * decay_factor
                param.data += xi

    def select(self):
        best = torch.argsort(self._scores.clone(), descending=True)[:self.population_size].tolist()
        self._population = [self._population[i] for i in best]
        return self._population

    def run(self, x: Tensor, y: Tensor, **kwargs) -> nn.Module:
        self._gen_num += 1

        self.set_fitness(x=x, y=y)
        self.select()

        self.breed()
        self.mutate()
        self.set_fitness(x=x, y=y)

        return self.best_model
