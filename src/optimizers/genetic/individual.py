from copy import deepcopy

import torch
from torch import Tensor

from models.base import BaseModel
from models.losses.base import BaseLoss


class Individual(object):
    def __init__(self, model: BaseModel, loss_fn: BaseLoss):
        self.model = deepcopy(model)
        self.loss_fn = loss_fn
        self.genome = self.to_vector()
        self.fitness = None
        self.init_weights()

    def init_weights(self) -> BaseModel:
        for param in self.model.parameters():
            param.data = torch.randn_like(param)
        return self.model

    def to_vector(self) -> Tensor:
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def to_model(self) -> BaseModel:
        idx = 0
        for param in self.model.parameters():
            param.data = (
                self.genome[idx : idx + param.numel()].view(param.shape).clone()
            )
            idx += param.numel()

        return self.model

    def evaluate(self, x: Tensor, y: Tensor) -> float:
        self.to_model()

        with torch.no_grad():
            y_hat = self.model(x)
            loss = self.loss_fn.loss(y, y_hat)

        self.fitness = -loss
        return self.fitness
