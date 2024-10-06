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
        num_of_param_snapshots: int = 10
    ) -> None:

        super(GeneticTrainer, self).__init__(
            model=model,
            loss=loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
        self.optimizer = optimizer
        self.num_of_param_snapshots = num_of_param_snapshots
        self.param_snapshots = []

    def train(self, verbose: Optional[bool] = True) -> nn.Module:
        for i in tqdm(range(self.hyperparams["epochs"]), desc="Training"):
            inputs, labels = self.train_dataset.features, self.train_dataset.labels
            self.model, train_loss = self.optimizer.run(inputs, labels)
            self.model.eval()
            self.train_losses.append(train_loss)

            if verbose:
                print(
                    f'Epoch [{i + 1}/{self.hyperparams["epochs"]}] loss: {train_loss}',
                    flush=True,
                )
            if not i%(self.hyperparams["epochs"]/self.num_of_param_snapshots):
                self.param_snapshots.append({name: param.data.clone() for name, param in self.model.named_parameters()})
            
            self.eval(verbose)

        return self.model

    def eval(self, verbose: Optional[bool] = True):
        self.model.eval()
        inputs, labels = self.eval_dataset.features, self.eval_dataset.labels
        eval_loss = self.loss(self.model(inputs), labels).item()
        self.eval_losses.append(eval_loss)

        if verbose is True:
            print(f"Validation loss: {eval_loss}", flush=True)
