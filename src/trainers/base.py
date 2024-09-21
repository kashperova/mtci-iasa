from typing import Callable, Union, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.train_config import BaseTrainConfig
from utils import CustomDataset


class BaseTrainer:
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        loss: Callable,
        train_dataset: CustomDataset,
        eval_dataset: CustomDataset,
        config: BaseTrainConfig
    ) -> None:
        self.model = model
        self.loss = loss
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return self.config.params

    @property
    def eval_loader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset, batch_size=self.hyperparams["eval_batch_size"], shuffle=False
        )

    @property
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hyperparams["train_batch_size"], shuffle=True
        )

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
