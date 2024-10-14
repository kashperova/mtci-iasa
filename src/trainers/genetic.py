from typing import Optional

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from config.train_config import BaseTrainConfig
from models.base import BaseModel
from models.losses.base import BaseLoss
from optimizers.genetic import GeneticOptimizer
from trainers.base import BaseTrainer


class GeneticTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
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

    def train(self, verbose: Optional[bool] = True) -> BaseModel:
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

    def eval(self, model: BaseModel, verbose: Optional[bool] = True) -> float:
        inputs, labels = map(torch.cat, zip(*[(x, y) for x, y in self.eval_loader]))
        eval_loss = self.loss.loss(model(inputs), labels)
        self.eval_losses.append(eval_loss)

        if verbose is True:
            print(f"Validation loss: {eval_loss}", flush=True)

        return eval_loss
