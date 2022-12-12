import copy
from dataclasses import field, dataclass
from typing import Dict, Union

import torch

from ml.src.configuration.configuration import Config
from ml.src.phase import Phase


@dataclass
class TrainingData:
    cfg: Config
    dataloaders: Dict[Phase, torch.utils.data.DataLoader]
    model: torch.nn.Module
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    # Base class is private in torch (https://github.com/pytorch/pytorch/pull/88503)
    scheduler: Union[
        torch.optim.lr_scheduler._LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ]

    best_val_loss: float = 1e10
    best_model_weights: dict = field(default_factory=dict)
    epoch: int = 0

    def update_best(self, val_loss: float) -> None:
        """
        Update the best model weights if the validation loss is lower than the previous best.
        :param val_loss: Current validation loss.
        """
        self.epoch += 1
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # save model weights
            self.best_model_weights = copy.deepcopy(self.model.state_dict())
