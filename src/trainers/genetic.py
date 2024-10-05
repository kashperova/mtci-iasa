import os
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config.train_config import BaseTrainConfig
from optimizers.genetic import GeneticOptimizer
from trainers.base import BaseTrainer


class GeneticTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable,
        optimizer: GeneticOptimizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: BaseTrainConfig,
    ) -> None:

        super(GeneticTrainer, self).__init__(
            model=model,
            loss=loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
        self.optimizer = optimizer

    def train(self, verbose: Optional[bool] = True) -> nn.Module:
        best_loss = float("inf")

        for i in tqdm(range(self.hyperparams["epochs"]), desc="Training"):
            inputs, labels = map(torch.cat, zip(*[(x, y) for x, y in self.train_loader]))
            model, train_loss = self.optimizer.run(inputs, labels)
            self.train_losses.append(train_loss)

            if verbose:
                print(
                    f'Epoch [{i + 1}/{self.hyperparams["epochs"]}] loss: {train_loss}',
                    flush=True,
                )

            eval_loss = self.eval(model, verbose)

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.model = model

        return self.model

    def eval(self, model: nn.Module, verbose: Optional[bool] = True) -> float:
        model.eval()
        inputs, labels = map(torch.cat, zip(*[(x, y) for x, y in self.eval_loader]))
        eval_loss = self.loss(model(inputs), labels).item()
        self.eval_losses.append(eval_loss)

        if verbose is True:
            print(f"Validation loss: {eval_loss}", flush=True)

        return eval_loss

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model.pth"))
