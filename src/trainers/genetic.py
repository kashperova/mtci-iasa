from typing import Callable

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

    def train(self) -> nn.Module:
        for i in tqdm(range(self.hyperparams["num_generations"]), desc="Training"):
            inputs, labels = self.train_dataset.features, self.train_dataset.labels
            self.model = self.optimizer.run(inputs, labels)
            loss = self.loss(self.model(inputs), labels)

            print(
                f'Generation [{i + 1}/{self.hyperparams["num_generations"]}] loss: {loss}',
                flush=True,
            )

        self.eval()

        return self.model

    def eval(self):
        self.model.eval()
        inputs, labels = self.eval_dataset.features, self.eval_dataset.labels
        eval_loss = self.loss(self.model(inputs), labels)
        print(f"Validation loss: {eval_loss}", flush=True)
