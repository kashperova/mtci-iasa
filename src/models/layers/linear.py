import torch
from torch import Tensor

from models.layers.base import BaseLayer


class LinearLayer(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = torch.randn(input_size, output_size, dtype=torch.float32) * 0.01
        self.biases = torch.zeros(output_size, dtype=torch.float32)
        self.velocity_w = torch.zeros_like(self.weights)
        self.velocity_b = torch.zeros_like(self.biases)

        # store gradients manually
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_biases = torch.zeros_like(self.biases)

    def forward(self, x: Tensor):
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, grad_output: Tensor, lr: float, momentum: float) -> Tensor:
        self.grad_weights = self.input.T @ grad_output
        self.grad_biases = grad_output.sum(dim=0)

        grad_input = grad_output @ self.weights.T

        self.velocity_w = momentum * self.velocity_w + lr * self.grad_weights
        self.velocity_b = momentum * self.velocity_b + lr * self.grad_biases

        self.weights -= self.velocity_w
        self.biases -= self.velocity_b

        return grad_input

    def zero_grad(self):
        self.grad_weights.zero_()
        self.grad_biases.zero_()
