from torch import nn
from typing import Optional

from config.model_config import MLPConfig as Config


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: Optional[int] = 1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Linear(Config.hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
