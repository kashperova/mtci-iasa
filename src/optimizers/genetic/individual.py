from copy import deepcopy
from typing import Callable

import torch
from torch import nn
from torch import Tensor


class Individual(object):
    def __init__(self, model: nn.Module, loss_fn: Callable):
        self.model = deepcopy(model)
        self.loss_fn = loss_fn
        self.genome = self.to_vector()
        self.fitness = None
        self.init_weights()

    def init_weights(self) -> nn.Module:
        for param in self.model.parameters():
            param.data = torch.randn(param.size())
        return self.model

    def to_vector(self) -> Tensor:
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def to_model(self) -> nn.Module:
        idx = 0
        for param in self.model.parameters():
            param.data = (
                self.genome[idx : idx + param.numel()].view(param.shape).clone()
            )
            idx += param.numel()

        return self.model

    def evaluate(self, x: Tensor, y: Tensor) -> float:
        self.to_model()
        self.model.eval()

        with torch.no_grad():
            loss = self.loss_fn(self.model(x), y)

        self.fitness = -loss.item()
        return self.fitness
