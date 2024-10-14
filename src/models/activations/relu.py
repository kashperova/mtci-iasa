import torch
from torch import Tensor

from models.activations.base import BaseActivation


class ReLU(BaseActivation):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        self.x = x
        return torch.maximum(x, torch.tensor(0.0))

    def backward(self, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        grad_input[self.x <= 0] = 0
        return grad_input
