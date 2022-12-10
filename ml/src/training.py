import copy
import os
from dataclasses import dataclass
from typing import Dict, Any, Union

import torch
import torch.utils.data
from omegaconf import OmegaConf
from enum import Enum
import wandb
from ml.src.configuration.configuration import ConfigStore, Config, Dataset, Model, Training


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


@dataclass
class TrainingData:
    cfg: Config
    dataloaders: Dict[Phase, torch.utils.data.DataLoader]
    model: torch.nn.Module
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    # Base class is private in torch (https://github.com/pytorch/pytorch/pull/88503)
    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]

    best_val_loss: float = None
    best_model_weights: dict = None
    epoch: int = 0

    def update_best(self, val_loss):
        self.epoch += 1
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # save model weights
            self.best_model_weights = copy.deepcopy(self.model.state_dict())


def create_loss(cfg: Config) -> torch.nn.Module:
    return None


def create_scheduler(cfg: Config) -> Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    return None


def create_optimizer(cfg: Config) -> torch.optim.Optimizer:
    return None


def create_dataloaders(cfg: Config) -> Dict[Phase, torch.utils.data.DataLoader]:
    return None


def create_model(cfg: Config) -> torch.nn.Module:
    return None


def train_epoch(training: TrainingData) -> float:
    training.model.train()

    loss_sum = 0
    for i, data in enumerate(training.dataloaders[Phase.TRAIN]):
        inputs, targets = data

        training.optimizer.zero_grad()

        outputs = training.model(inputs)

        loss = training.loss(outputs, targets)
        loss.backward()
        training.optimizer.step()

        loss_sum += loss.item()
        wandb.log({"train_batch_loss": loss.item()})

    return loss_sum / len(training.dataloaders[Phase.TRAIN])


def val_epoch(training: TrainingData) -> float:
    training.model.eval()

    loss_sum = 0
    for i, data in enumerate(training.dataloaders[Phase.VAL]):
        inputs, targets = data
        outputs = training.model(inputs)

        loss = training.loss(outputs, targets)
        loss_sum += loss.item()
        wandb.log({"val_batch_loss": loss.item()})

    return loss_sum / len(training.dataloaders[Phase.VAL])


def train():
    cfg = ConfigStore.cfg
    with wandb.init(entity="pg-pug-tennis-betting", project="tennis-betting", config=cfg):

        if ConfigStore.cfg is None:
            raise Exception("Config not loaded")
        # Override config with sweep values and save whole config in wandb run folder
        ConfigStore.sweep_override(wandb.config)
        ConfigStore.save_config(wandb.run.dir)
        wandb.save(f'all_config.yaml', policy="now")

        cfg: Config = ConfigStore.cfg

        dataloaders = create_dataloaders(cfg)

        model = create_model(cfg)
        if torch.cuda.is_available():
            model.cuda(cfg.training.gpu)

        loss = create_loss(cfg)
        optimizer = create_optimizer(cfg)
        scheduler = create_scheduler(cfg)

        training = TrainingData(cfg, dataloaders, model, loss, optimizer, scheduler)

        for epoch in range(cfg.training.epochs):
            train_epoch_loss = train_epoch(training)
            wandb.log({"train_epoch_loss": train_epoch_loss})

            val_epoch_loss = val_epoch(training)
            wandb.log({"val_epoch_loss": val_epoch_loss})

            training.update_best(val_epoch_loss)


def main():
    root_path = os.path.abspath(f'{os.path.dirname(os.path.abspath(__file__))}/../../')
    config_path = f'{root_path}/ml/config/config.yaml'

    ConfigStore.load(config_path)
    cfg = ConfigStore.cfg

    if cfg.root_path is None:
        cfg.root_path = root_path
    print(cfg.dataset.metadata_path)

    if cfg.sweep is None:
        train()
    else:
        sweep_cfg = OmegaConf.to_container(cfg.sweep, resolve=True)
        sweep_id = wandb.sweep(sweep_cfg, entity="pg-pug-tennis-betting", project="tennis-betting")
        wandb.agent(sweep_id, train, count=1)


if __name__ == "__main__":
    main()
