import os
from typing import Dict

import hydra
import torch
from omegaconf import omegaconf
from enum import Enum
import wandb
import ml.src.configuration.configuration as config


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


def create_dataloaders(dataset: config.Dataset) -> Dict[Phase, torch.utils.data.DataLoader]:
    return None


def create_model(model: config.Model) -> torch.nn.Module:
    return None


def train_step():
    pass


def train_epoch():
    pass


def val_epoch():
    pass


root_path = os.path.abspath(f'{os.path.dirname(os.path.abspath(__file__))}/../../')
config_path = f'{root_path}/ml/config'


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: config.Config):
    cfg.root_path = root_path
    with wandb.init(project="tennis-betting", config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)):
        dataloaders = create_dataloaders(cfg.dataset)
        model = create_model(cfg.model)

        print(cfg.dataset.metadata_path)
        print(cfg.dataset.data_path)


if __name__ == "__main__":
    main()