import torch
from torch import Tensor

from models.layers.base import BaseLayer


class LinearLayer(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = torch.randn(input_size, output_size, dtype=torch.float32) * 0.01
        self.biases = torch.zeros(output_size, dtype=torch.float32)

    def forward(self, x: Tensor):
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, grad_output: Tensor, lr: float) -> Tensor:
        grad_weights = self.input.T @ grad_output
        grad_biases = torch.sum(grad_output, dim=0)

        grad_input = grad_output @ self.weights.T

        self.weights -= lr * grad_weights
        self.biases -= lr * grad_biases

        return grad_input
