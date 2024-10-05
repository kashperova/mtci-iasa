import os
from typing import Callable, Union, Dict, Any, Optional

import torch
import plotly.graph_objects as go
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config.train_config import BaseTrainConfig


class BaseTrainer:
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        loss: Callable,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: BaseTrainConfig,
        save_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss = loss
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.save_dir = os.getcwd() if save_dir is None else save_dir

        self.train_losses = []
        self.eval_losses = []

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

    def eval(self, model: nn.Module):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def plot_losses(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.train_losses,
            mode='lines',
            name='Training Loss'
        ))
        fig.add_trace(go.Scatter(
            y=self.eval_losses,
            mode='lines',
            name='Validation Loss'
        ))
        fig.update_layout(
            title='Losses',
            xaxis_title='Epochs',
            yaxis_title='Loss',
            legend=dict(
                x=0,
                y=1,
                traceorder="normal"
            )
        )
        fig.show()
