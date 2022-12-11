import os
from dataclasses import dataclass, field
from typing import Optional

import wandb
from omegaconf import SI, OmegaConf


@dataclass
class Dataset:
    data_path: str = SI("${root_path}/data/own/data.csv")


@dataclass
class Model:
    name: str = "ANN"
    n_layers: int = 3
    n_features: int = 881
    n_classes: int = 2
    hidden_size: int = 256
    dropout: float = 0.1
    time_steps: int = 10


@dataclass
class Training:
    gpu: int = 0
    epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    seed: Optional[int] = None
    lr: float = 0.001
    weight_decay: float = 0
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001


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
    def load(path: str) -> None:
        cfg: Config = OmegaConf.structured(Config)
        cfg_raw = OmegaConf.load(path)
        ConfigStore.cfg = OmegaConf.merge(cfg, cfg_raw)

    @staticmethod
    def sweep_override(sweep_config: wandb.sdk.wandb_config.Config) -> None:
        sweep_overrides = OmegaConf.create(dict(sweep_config))
        ConfigStore.cfg = OmegaConf.merge(ConfigStore.cfg, sweep_overrides)

    @staticmethod
    def save_config(path: str, filename: str = "all_config.yaml") -> None:
        OmegaConf.save(ConfigStore.cfg, os.path.join(path, filename))

    @staticmethod
    def print_config() -> None:
        print(OmegaConf.to_yaml(ConfigStore.cfg))

    @staticmethod
    def to_yaml_string() -> str:
        return OmegaConf.to_yaml(ConfigStore.cfg)
