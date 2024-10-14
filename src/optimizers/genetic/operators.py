from typing import Optional, List, Tuple

import torch

from models.losses.base import BaseLoss
from optimizers.genetic.individual import Individual
from optimizers.genetic.population import Population


class GeneticOperators:
    def __init__(self, loss_fn: BaseLoss):
        self.loss_fn = loss_fn

    @classmethod
    def select(cls, population: Population, tournament_size: Optional[int] = 3) -> List[Individual]:
        selected = []
        fitness_tensor = torch.tensor(population.scores)

        for _ in range(population.size):
            indices = torch.randperm(population.size)[:tournament_size]
            scores = fitness_tensor[indices]
            winner = indices[torch.argmax(scores)]
            selected.append(population.individuals[winner])

        return selected

    @classmethod
    def mutate(cls, individual: Individual, rate: float):
        mutation_mask = torch.rand(individual.genome.size()) < rate
        mutation_values = torch.randn(individual.genome.size()) * 0.1
        individual.genome[mutation_mask] += mutation_values[mutation_mask]
        return individual

    def crossover(self, parent1: Individual, parent2: Individual, rate: float) -> Tuple[Individual, Individual]:
        if torch.rand(1).item() < rate:
            point = torch.randint(1, len(parent1.genome)-1, (1,)).item()
            child1_genome = torch.cat((parent1.genome[:point], parent2.genome[point:]))
            child2_genome = torch.cat((parent2.genome[:point], parent1.genome[point:]))
        else:
            child1_genome = parent1.genome.clone()
            child2_genome = parent2.genome.clone()

        child1 = Individual(parent1.model, self.loss_fn)
        child2 = Individual(parent2.model, self.loss_fn)
        child1.genome = child1_genome
        child2.genome = child2_genome

        return child1, child2
