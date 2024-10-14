from typing import Optional

import torch
from torch import Tensor

from models.activations import ReLU
from models.base import BaseModel
from models.layers import LinearLayer

from config.model_config import MLPConfig as Config


class MLP(BaseModel):
    def __init__(self, input_size: int, output_size: Optional[int] = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layers.append(LinearLayer(input_size, Config.hidden_size))
        self.layers.append(ReLU())
        self.layers.append(LinearLayer(Config.hidden_size, Config.hidden_size))
        self.layers.append(ReLU())
        self.layers.append(LinearLayer(Config.hidden_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
             x = layer.forward(x)
        return x

    def serialize(self, path: str):
        data = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layers": self.layers
        }
        torch.save(data, path)

    @classmethod
    def deserialize(cls, path: str) -> 'BaseModel':
        data = torch.load(path)
        model = cls(data['input_size'], data['output_size'])
        model.layers = data['layers']
        return model
