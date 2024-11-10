import os
import random

import numpy as np
import torch

from utils.singleton import Singleton


class SeedSetter(metaclass=Singleton):
    def __init__(self, seed: int) -> None:
        self._seed = seed
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)

        os.environ['PYTHONHASHSEED'] = str(self._seed)
        print(f'Random seed set to {self._seed}')

    @property
    def seed(self) -> int:
        return self._seed
