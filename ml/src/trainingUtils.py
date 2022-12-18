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
from torchmetrics import Accuracy


class reward_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, inputs):
        # Bet of player 0 is on inputs[0] and player 1 is on inputs[1]
        # avg_bet is avarage of chosen player
        # loss = output[target] * odd_target
        odds = inputs[:, targets].diagonal()
        win = outputs[:, targets].diagonal()
        return -torch.mul(win, odds).mean()


def create_loss(cfg: Config) -> torch.nn.Module:
    if cfg.training.loss == "reward":
        return reward_loss()
    elif cfg.training.loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception("Unknown loss")


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


def create_dataloaders(cfg: Config, force_new_instance: bool = False) -> Dict[Phase, torch.utils.data.DataLoader]:

    if not hasattr(create_dataloaders, "datasets"):
        create_dataloaders.datasets = {}

    dataloaders = {}
    for phase in Phase:
        if force_new_instance or phase not in create_dataloaders.datasets:
            create_dataloaders.datasets[phase] = TennisDataset(cfg, phase)
        dataloaders[phase] = create_dataloaders.datasets[phase].create_data_loader(cfg, phase)

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


def calc_metrics(inputs, outputs, targets, metrics, training):
    acc = Accuracy(task="multiclass", num_classes=training.cfg.model.n_classes)
    if torch.cuda.is_available():
        acc = acc.cuda(training.cfg.training.gpu)
    metrics["acc"] += acc(outputs, targets).item()
    if training.cfg.training.loss == "reward":
        metrics["loss"] += training.loss(outputs, targets, inputs).item()
    else:
        metrics["loss"] += training.loss(outputs, targets).item()
    # Bet of player 0 is on inputs[0] and player 1 is on inputs[1]
    # avg_bet is avarage of chosen player
    pick = torch.argmax(outputs, dim=1)
    correct_pick = (pick == targets).int()
    odds = inputs[:, pick].diagonal()
    win = torch.sub(torch.mul(odds, correct_pick).sum(), correct_pick.shape[0])
    metrics["bet_reward"] += win.item()
    # win sum is the sum of all correct picks
    metrics["win_sum"] += correct_pick.sum().item()
    # loss sum is the sum of all incorrect picks
    metrics["loss_sum"] += (1 - correct_pick).sum().item()
    # avg_bet is the average of all odds
    metrics["avg_bet"] += odds.mean().item()
    # avg_bet_win is the average of all odds of correct picks
    metrics["avg_bet_win"] += (odds * correct_pick).sum().item()
    # avg_bet_loss is the average of all odds of incorrect picks
    metrics["avg_bet_loss"] += (odds * (1 - correct_pick)).sum().item()


def print_metrics(metrics, phase_length, wandb, phase):
    wandb.log({f"{phase}_loss": metrics["loss"] / phase_length})
    wandb.log({f"{phase}_acc": metrics["acc"] / phase_length})
    wandb.log({f"{phase}_avg_bet": metrics["avg_bet"] / phase_length})
    wandb.log({f"{phase}_avg_bet_win": metrics["avg_bet_win"] / metrics["win_sum"]})
    wandb.log({f"{phase}_avg_bet_loss": metrics["avg_bet_loss"] / metrics["loss_sum"]})
    wandb.log({f"{phase}_bet_reward": metrics["bet_reward"]})


def train_epoch(training: TrainingData) -> Dict[str, float]:
    """
    Train the model for one epoch
    :param training: Dataclass containing all the training data
    :return: Average loss for the epoch
    """
    training.model.train()

    metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "avg_bet": 0.0,
        "avg_bet_win": 0.0,
        "avg_bet_loss": 0.0,
        "win_sum": 0.0,
        "loss_sum": 0.0,
        "bet_reward": 0.0,
    }

    for i, data in enumerate(training.dataloaders[Phase.TRAIN]):
        inputs, targets = data
        training.optimizer.zero_grad()

        outputs = training.model(inputs)

        if training.cfg.training.loss == "reward":
            loss = training.loss(outputs, targets, inputs)
        else:
            loss = training.loss(outputs, targets)
        loss.backward()
        training.optimizer.step()

        calc_metrics(inputs, outputs, targets, metrics, training)
        wandb.log({"train_batch_loss": loss.item()})

    return metrics


def val_epoch(training: TrainingData, phase: Phase = Phase.VAL) -> Dict[str, float]:
    """
    Validate the model for one epoch
    :param training: Dataclass containing all the training data
    :param phase: phase to validate on (default: val)
    :return: Average loss for the epoch and accuracy
    """
    training.model.eval()

    metrics = {
        "loss": 0.0,
        "acc": 0.0,
        "avg_bet": 0.0,
        "avg_bet_win": 0.0,
        "avg_bet_loss": 0.0,
        "win_sum": 0.0,
        "loss_sum": 0.0,
        "bet_reward": 0.0,
    }
    for i, data in enumerate(training.dataloaders[phase]):
        inputs, targets = data
        outputs = training.model(inputs)

        if training.cfg.training.loss == "reward":
            loss = training.loss(outputs, targets, inputs)
        else:
            loss = training.loss(outputs, targets)
        calc_metrics(inputs, outputs, targets, metrics, training)
        wandb.log({"val_batch_loss": loss.item()})
        # add to win sum if pick is correct

    return metrics
