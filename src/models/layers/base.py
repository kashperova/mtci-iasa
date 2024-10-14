from torch import Tensor


class BaseLayer:
    input: Tensor = None

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, x: Tensor):
        raise NotImplementedError

    def backward(self, grad_output: Tensor, lr: float) -> Tensor:
        raise NotImplementedError
