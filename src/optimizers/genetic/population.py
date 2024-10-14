from typing import List, Tuple
from scipy.spatial.distance import pdist
from scipy.stats import tstd
from torch import Tensor

from models.base import BaseModel
from models.losses.base import BaseLoss
from optimizers.genetic.individual import Individual


class Population:
    def __init__(self, size: int, model: BaseModel, loss_fn: BaseLoss):
        self.size = size
        self.individuals = [Individual(model, loss_fn) for _ in range(self.size)]

    @property
    def scores(self) -> List[float]:
        return [ind.fitness for ind in self.individuals]

    @property
    def fitness_deviation(self) -> float:
        return tstd(self.scores)

    @property
    def genome_deviation(self) -> float:
        distances = pdist([ind.genome for ind in self.individuals])
        std_dev = tstd(distances)
        return std_dev

    @property
    def best(self) -> Tuple[BaseModel, float]:
        best_fitness = max(self.scores)
        best_model = max(self.individuals, key=lambda ind: ind.fitness).to_model()
        return best_model, -best_fitness

    def evaluate(self, x: Tensor, y: Tensor):
        for individual in self.individuals:
            individual.evaluate(x, y)

    def update(self, individuals: List[Individual]):
        self.individuals = individuals[:self.size]
