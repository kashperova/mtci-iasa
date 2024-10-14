from typing import Union

from torch import Tensor


class BaseLoss:
    def loss(self, y: Tensor, y_hat: Tensor) -> float:
        raise NotImplementedError

    def backward(self, y: Tensor, y_hat: Tensor) -> Union[Tensor, float]:
        raise NotImplementedError
