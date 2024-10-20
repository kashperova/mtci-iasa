from dataclasses import dataclass


@dataclass
class GeneticConfig:
    epochs: int = 500
    train_batch_size: int = 32
    eval_batch_size: int = 64
    population_size: int = 100
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.01


@dataclass
class GDConfig:
    epochs: int = 50
    train_batch_size: int = 16
    eval_batch_size: int = 64
    momentum: float = 0.85
    lr: float = 0.001
