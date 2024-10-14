from typing import Union

import torch
from torch import Tensor

from models.losses.base import BaseLoss


class MSELoss(BaseLoss):
    def loss(self, y: Tensor, y_hat: Tensor) -> float:
        return torch.mean((y_hat - y) ** 2).item()

    def backward(self, y: Tensor, y_hat: Tensor) -> Union[Tensor, float]:
        return 2 * (y_hat - y) / y.size(0)
