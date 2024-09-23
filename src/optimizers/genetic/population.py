from typing import Callable, List, Tuple

from torch import nn
from torch import Tensor

from optimizers.genetic.individual import Individual


class Population:
    def __init__(self, size: int, model: nn.Module, loss_fn: Callable):
        self.size = size
        self.individuals = [Individual(model, loss_fn) for _ in range(self.size)]

    @property
    def scores(self) -> List[float]:
        return [ind.fitness for ind in self.individuals]

    @property
    def best(self) -> Tuple[nn.Module, float]:
        best_fitness = max(self.scores)
        best_model = max(self.individuals, key=lambda ind: ind.fitness).to_model()
        return best_model, -best_fitness

    def evaluate(self, x: Tensor, y: Tensor):
        for individual in self.individuals:
            individual.evaluate(x, y)

    def update(self, individuals: List[Individual]):
        self.individuals = individuals[:self.size]
