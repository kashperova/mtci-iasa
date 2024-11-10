import os
from typing import Optional

import torch
import plotly.graph_objects as go
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig

from models.base import BaseModel
from models.losses.base import BaseLoss


class BaseTrainer:
    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: DictConfig,
        save_dir: Optional[str] = None,
        save_name: Optional[str] = "model.pth",
    ) -> None:
        self.model = model
        self.loss = loss
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.save_dir = os.getcwd() if save_dir is None else save_dir
        self.save_name = save_name

        self.train_losses = []
        self.eval_losses = []

    @property
    def eval_loader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset, batch_size=self.config.eval_batch_size, shuffle=False
        )

    @property
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True
        )

    def train(self):
        raise NotImplementedError

    def eval(self, model: BaseModel):
        raise NotImplementedError

    def save(self):
        self.model.serialize(os.path.join(self.save_dir, self.save_name))

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
