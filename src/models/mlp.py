from torch import nn
from typing import Optional

from config.model_config import MLPConfig as Config


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: Optional[int] = 1):
        super(MLP, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, output_size)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self._model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self._model(x)
