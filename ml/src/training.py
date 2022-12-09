import copy
import os
from dataclasses import dataclass
from typing import Dict, Any

import hydra
import torch
import torch.utils.data
from omegaconf import omegaconf
from enum import Enum
import wandb
import ml.src.configuration.configuration as config


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


@dataclass
class TrainingData:
    cfg: config.Config
    wb: wandb
    dataloaders: Dict[Phase, torch.utils.data.DataLoader]
    model: torch.nn.Module
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any  # Base class is private in torch (https://github.com/pytorch/pytorch/pull/88503)

    best_val_loss: float = None
    best_model_weights: dict = None
    epoch: int = 0

    def update(self, val_loss):
        self.epoch += 1
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # save model weights
            self.best_model_weights = copy.deepcopy(self.model.state_dict())


def create_loss(cfg: config.Config) -> torch.nn.Module:
    return None


def create_scheduler(cfg: config.Config) -> Any:
    return None


def create_optimizer(cfg: config.Config) -> torch.optim.Optimizer:
    return None


def create_dataloaders(cfg: config.Config) -> Dict[Phase, torch.utils.data.DataLoader]:
    return None


def create_model(cfg: config.Config) -> torch.nn.Module:
    return None


def train_epoch(training: TrainingData):
    training.model.train()
    for i, data in enumerate(training.dataloaders[Phase.TRAIN]):
        inputs, targets = data

        training.optimizer.zero_grad()

        outputs = training.model(inputs)

        loss_value = training.loss(outputs, targets)
        loss_value.backward()

        training.optimizer.step()
        training.wb.log({"train_loss": loss_value.item()})


def val_epoch(training: TrainingData):
    training.model.eval()
    for i, data in enumerate(training.dataloaders[Phase.VAL]):
        inputs, targets = data
        outputs = training.model(inputs)

        loss_value = training.loss(outputs, targets)
        training.wb.log({"val_loss": loss_value.item()})


root_path = os.path.abspath(f'{os.path.dirname(os.path.abspath(__file__))}/../../')
config_path = f'{root_path}/ml/config'


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: config.Config):
    cfg.root_path = root_path
    with wandb.init(project="tennis-betting",
                    config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)):

        dataloaders = create_dataloaders(cfg)

        model = create_model(cfg)
        if torch.cuda.is_available():
            model.cuda(cfg.training.gpu)

        loss = create_loss(cfg)
        optimizer = create_optimizer(cfg)
        scheduler = create_scheduler(cfg)

        training = TrainingData(cfg, wandb, dataloaders, model, loss, optimizer, scheduler)

        for epoch in range(cfg.training.epochs):
            train_epoch(training)
            val_epoch(training)


if __name__ == "__main__":
    main()
