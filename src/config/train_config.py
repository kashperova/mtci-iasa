from typing import Dict, Any


class BaseTrainConfig:
    @property
    def params(self) -> Dict[str, Any]:
        attrs = set(self.__class__.__dict__.keys()) - set(self.__dict__.keys())
        return {attr: getattr(self, attr) for attr in attrs}


class GeneticConfig(BaseTrainConfig):
    epochs: int = 500
    train_batch_size: int = 32
    eval_batch_size: int = 64
