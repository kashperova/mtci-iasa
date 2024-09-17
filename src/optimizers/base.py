from abc import abstractmethod, ABC
from typing import Callable, Optional, Dict, Any

import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def run(
        self,
        loss: Callable,
        x: np.array,
        y: np.array,
        weights_init: Optional[Callable] = None,
        init_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        raise NotImplementedError
