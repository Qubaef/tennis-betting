import os
from dataclasses import dataclass, field
from typing import Optional, Any

import wandb
from omegaconf import ListConfig, DictConfig, OmegaConf, MissingMandatoryValue, SI, OmegaConf, MISSING


@dataclass
class Dataset:
    data_path: str = SI("${root_path}/data/own/data.csv")


@dataclass
class Model:
    name: str = "ANN"
    n_layers: int = MISSING
    n_features: int = MISSING
    n_classes: int = MISSING
    hidden_size: int = MISSING
    dropout: float = MISSING
    time_steps: int = MISSING


@dataclass
class Training:
    gpu: int = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    seed: Optional[int] = None
    lr: float = MISSING
    weight_decay: float = MISSING
    factor: float = MISSING
    patience: int = MISSING
    threshold: float = MISSING


@dataclass
class Config:
    dataset: Dataset = field(default_factory=Dataset)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    sweep: Optional[dict] = None
    root_path: Optional[str] = None


class ConfigStore(type):
    cfg: Config = OmegaConf.structured(Config)

    @staticmethod
    def load(path: str):
        cfg = OmegaConf.structured(Config)
        cfg_raw = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, cfg_raw)
        ConfigStore.validate(cfg)
        ConfigStore.cfg = cfg

    @staticmethod
    def validate(cfg: Any) -> None:
        if isinstance(cfg, ListConfig):
            for x in cfg:
                ConfigStore.validate(x)
        elif isinstance(cfg, DictConfig):
            for _, v in cfg.items():
                ConfigStore.validate(v)

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
