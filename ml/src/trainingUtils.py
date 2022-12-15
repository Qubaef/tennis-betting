from typing import Dict, Any, Union, Tuple

import torch
import torch.utils.data

from ml.src.trainingData import TrainingData
from ml.src.phase import Phase
import wandb
from torch import nn, optim
from LSTM import LSTMModel
from ANN import ANNModel
from ml.src.configuration.configuration import (
    Config,
)
from ml.src.dataset import TennisDataset


def create_loss(cfg: Config) -> torch.nn.Module:
    return nn.CrossEntropyLoss()


def create_scheduler(
    cfg: Config, optimizer: torch.optim.Optimizer
) -> Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.factor,
        patience=cfg.training.patience,
        verbose=True,
        threshold=cfg.training.threshold,
        threshold_mode="abs",
    )


def create_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    return optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )


def create_dataloaders(cfg: Config) -> Dict[Phase, torch.utils.data.DataLoader]:
    dataloaders = {}
    for phase in Phase:
        dataset = TennisDataset(cfg, phase)
        dataloaders[phase] = dataset.create_data_loader(phase)
    return dataloaders


def create_model(cfg: Config) -> torch.nn.Module:
    if cfg.model.name == "ANN":
        return ANNModel(
            cfg.model.n_features,
            cfg.model.hidden_size,
            cfg.model.n_classes,
            cfg.model.n_layers,
            cfg.model.dropout,
        )
    elif cfg.model.name == "LSTM":
        return LSTMModel(
            cfg.model.n_features,
            cfg.model.hidden_size,
            cfg.model.n_classes,
            cfg.model.n_layers,
            cfg.training.batch_size,
            cfg.model.time_steps,
            cfg.model.dropout,
        )
    else:
        raise Exception("Unknown model")


def train_epoch(training: TrainingData) -> float:
    """
    Train the model for one epoch
    :param training: Dataclass containing all the training data
    :return: Average loss for the epoch
    """
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


def val_epoch(training: TrainingData, phase: Phase = Phase.VAL) -> Tuple[float, float]:
    """
    Validate the model for one epoch
    :param training: Dataclass containing all the training data
    :param phase: phase to validate on (default: val)
    :return: Average loss for the epoch and accuracy
    """
    training.model.eval()

    loss_sum = 0
    guessed_scores_num = 0

    for i, data in enumerate(training.dataloaders[phase]):
        inputs, targets = data
        outputs = training.model(inputs)

        loss = training.loss(outputs, targets)
        loss_sum += loss.item()
        wandb.log({"val_batch_loss": loss.item()})

        guesses = torch.argmax(outputs, dim=1)
        guessed_scores_num += torch.sum(guesses == targets).item()

    acc: float = guessed_scores_num / (len(training.dataloaders[phase].dataset))

    return loss_sum / len(training.dataloaders[phase]), acc
