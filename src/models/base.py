from typing import List, Union

from torch import Tensor

from models.activations import ReLU
from models.activations.base import BaseActivation
from models.layers import LinearLayer
from models.layers.base import BaseLayer


class BaseModel:
    layers: List[Union[BaseLayer, BaseActivation]] = []
    _params: List[Tensor] = []

    def parameters(self):
        if len(self._params) > 0:
            return self._params
        self._params = self.init_parameters()
        return self._params

    def init_parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params.append(layer.weights)
                params.append(layer.biases)
        return params

    def set_params(self, params: List[Tensor]):
        self._params = params

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

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.zero_grad()

    def gradients(self):
        grads = []
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                grads.append(layer.grad_weights)
                grads.append(layer.grad_biases)
        return grads

    def serialize(self, path: str):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, path: str) -> 'BaseModel':
        raise NotImplementedError
