import os
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from models import MLP
from models.base import BaseModel
from models.losses.base import BaseLoss
from trainers.base import BaseTrainer


class GDTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: DictConfig,
    ):
        super(GDTrainer, self).__init__(
            model=model,
            loss=loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
        self.lr = self.config.lr
        self.momentum = self.config.momentum

    def train(self, verbose: Optional[bool] = True) -> BaseModel:
        best_loss = float("inf")

        for i in tqdm(range(self.config.epochs), desc="Training"):
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                outputs = self.model.forward(inputs)
                loss = self.loss.loss(y=targets, y_hat=outputs)

                loss_grad = self.loss.backward(y=targets, y_hat=outputs)
                self.model.backward(loss_grad, lr=self.lr, momentum=self.momentum)

                train_loss += loss

            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            if verbose:
                print(
                    f'Epoch [{i + 1}/{self.config.epochs}] loss: {train_loss}',
                    flush=True,
                )

            eval_loss = self.eval(self.model, verbose)
            self.eval_losses.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save()

        return MLP.deserialize(os.path.join(self.save_dir, self.save_name))

    def eval(self, model: BaseModel, verbose: Optional[bool] = True) -> float:
        eval_loss = 0
        with torch.no_grad():
            for inputs, targets in self.eval_loader:
                predictions = self.model.forward(inputs)
                eval_loss += self.loss.loss(predictions, targets)

        eval_loss /= len(self.eval_loader)

        if verbose is True:
            print(f"Validation loss: {eval_loss}", flush=True)

        return eval_loss
