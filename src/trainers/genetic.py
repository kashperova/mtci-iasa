from typing import Callable, Optional

from torch import nn
from tqdm.auto import tqdm

from config.train_config import BaseTrainConfig
from optimizers.genetic import GeneticOptimizer
from trainers.base import BaseTrainer
from utils import CustomDataset


class GeneticTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable,
        optimizer: GeneticOptimizer,
        train_dataset: CustomDataset,
        eval_dataset: CustomDataset,
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
        for i in tqdm(range(self.hyperparams["epochs"]), desc="Training"):
            inputs, labels = self.train_dataset.features, self.train_dataset.labels
            self.model = self.optimizer.run(inputs, labels)
            self.model.eval()
            train_loss = self.loss(self.model(inputs), labels).item()
            self.train_losses.append(train_loss)

            if verbose:
                print(
                    f'Epoch [{i + 1}/{self.hyperparams["epochs"]}] loss: {train_loss}',
                    flush=True,
                )

            self.eval(verbose)

        return self.model

    def eval(self, verbose: Optional[bool] = True):
        if verbose is True:
            self.model.eval()
            inputs, labels = self.eval_dataset.features, self.eval_dataset.labels
            eval_loss = self.loss(self.model(inputs), labels).item()
            self.eval_losses.append(eval_loss)
            print(f"Validation loss: {eval_loss}", flush=True)
