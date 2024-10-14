from torch import Tensor


class BaseActivation:
    x = None

    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self, grad_output: Tensor) -> Tensor:
        raise NotImplementedError
