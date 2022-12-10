import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import omegaconf
import wandb
from omegaconf import SI, OmegaConf


@dataclass
class Dataset:
    path: Optional[str] = SI("${root_path}/data")
    metadata_path: str = SI("${path}/metadata.csv")
    data_path: str = SI("${path}/data")


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
    root_path: Optional[str] = None


class ConfigStore(type):
    cfg: Optional[Config] = None

    @staticmethod
    def load(path: str):
        cfg: Config = OmegaConf.structured(Config)
        cfg_raw = OmegaConf.load(path)
        ConfigStore.cfg = OmegaConf.merge(cfg, cfg_raw)

    @staticmethod
    def sweep_override(sweep_config: wandb.sdk.wandb_config.Config):
        sweep_overrides = OmegaConf.create(dict(sweep_config))
        ConfigStore.cfg = OmegaConf.merge(ConfigStore.cfg, sweep_overrides)

    @staticmethod
    def save_config(path: str, filename: str = "all_config.yaml"):
        OmegaConf.save(ConfigStore.cfg, os.path.join(path, filename))

    @staticmethod
    def print_config():
        print(OmegaConf.to_yaml(ConfigStore.cfg))

    @staticmethod
    def to_yaml_string():
        return OmegaConf.to_yaml(ConfigStore.cfg)
