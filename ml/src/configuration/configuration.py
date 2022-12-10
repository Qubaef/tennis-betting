import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import omegaconf
import wandb
from omegaconf import SI, OmegaConf


@dataclass
class Dataset:
    path: Optional[str] = None
    metadata_path: str = SI("${path}/metadata.csv")
    data_path: str = SI("${path}/metadata.csv")


@dataclass
class Model:
    name: str = "model"
    n_layers: int = 3
    n_features: int = 10


@dataclass
class Training:
    gpu: int = 0
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    seed: Optional[int] = None


@dataclass
class Config:
    dataset: Dataset = field(default_factory=Dataset)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    sweep: Optional[dict] = None


class ConfigStore(type):
    cfg: Optional[Config] = None

    @staticmethod
    def load(path: str):
        cfg: Config = OmegaConf.structured(Config)
        cfg_raw = OmegaConf.load(path)
        ConfigStore.cfg = OmegaConf.merge(cfg, cfg_raw)

    @staticmethod
    def sweep_override(sweep_config: wandb.sdk.wandb_config.Config):

        for key, value in sweep_config.items():
            OmegaConf.update(ConfigStore.cfg, key, value, merge=True)

