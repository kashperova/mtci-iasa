from typing import List, Union

from torch import Tensor

from models.activations import ReLU
from models.activations.base import BaseActivation
from models.layers import LinearLayer
from models.layers.base import BaseLayer


class BaseModel:
    layers: List[Union[BaseLayer, BaseActivation]] = []

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params.append(layer.weights)
                params.append(layer.biases)
        return params

    def forward(self, x: Tensor):
        raise NotImplementedError

    def __call__(self, x: Tensor):
        return self.forward(x)

    def backward(self, loss_grad: Tensor, lr: float, momentum: float):
        for layer in reversed(self.layers):
            if isinstance(layer, ReLU):
                loss_grad = layer.backward(loss_grad)
            elif isinstance(layer, LinearLayer):
                loss_grad = layer.backward(loss_grad, lr, momentum)

    def serialize(self, path: str):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, path: str) -> 'BaseModel':
        raise NotImplementedError
