import os
from dataclasses import dataclass, field
import uuid
from typing import Optional, Any

import wandb
from omegaconf import (
    ListConfig,
    DictConfig,
    SI,
    OmegaConf,
    MISSING, DictKeyType,
)


@dataclass
class Dataset:
    data_path: str = SI(MISSING)


@dataclass
class Model:
    name: str = MISSING
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
    loss: str = MISSING


@dataclass
class Config:
    dataset: Dataset = field(default_factory=Dataset)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    name: Optional[str] = None
    sweep: Optional[dict] = None
    root_path: str = MISSING


class ConfigStore(type):
    cfg: Config = OmegaConf.structured(Config)

    # default project root path
    default_root_path: str = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../../../")
    # default project config directory
    default_config_path: str = f"{default_root_path}/ml/config/"

    run_count: int = 0

    @staticmethod
    def load(path: str) -> None:
        """
        Loads config from yaml file.
        If path is not absolute, it is assumed to be relative to the default config directory.
        param path: relative or absolute path to config file
        """
        if not os.path.isabs(path):
            path = f"{ConfigStore.default_config_path}/{path}"
        cfg_raw = OmegaConf.load(path)
        cfg_typed = OmegaConf.structured(Config)
        ConfigStore.cfg = OmegaConf.merge(cfg_typed, cfg_raw)

        if OmegaConf.is_missing(ConfigStore.cfg, "root_path"):
            ConfigStore.cfg.root_path = ConfigStore.default_root_path

        uid = uuid.uuid4().hex[:5]
        if OmegaConf.is_missing(ConfigStore.cfg, "name") or ConfigStore.cfg.name is None or ConfigStore.cfg.name == "":
            name = f"{uid}"
        else:
            name = f"{ConfigStore.cfg.name}_{uid}"

        if ConfigStore.cfg.sweep is None:
            name = f"run_{name}"
        else:
            name = f"sweep_{name}"
        ConfigStore.cfg.name = name

    @staticmethod
    def handle_wandb(run: wandb.sdk.wandb_run.Run) -> None:
        """
        Synchronize config with wandb and apply overrides from sweep.
        Save config file to wandb run.
        """
        if ConfigStore.cfg is None:
            raise Exception("Config not loaded")

        sweep_mode: bool = ConfigStore.cfg.sweep is not None
        if sweep_mode:
            sweep_overrides = OmegaConf.create(dict(wandb.config))
            ConfigStore.cfg = OmegaConf.merge(ConfigStore.cfg, sweep_overrides)

        if sweep_mode:
            run.name = f"{ConfigStore.cfg.name}_run-{ConfigStore.run_count}"
            ConfigStore.run_count += 1
        else:
            run.name = ConfigStore.cfg.name

        ConfigStore.validate()
        run.config.setdefaults(OmegaConf.to_container(ConfigStore.cfg, resolve=True, throw_on_missing=True))

        ConfigStore.save_config(run.dir)

    @staticmethod
    def validate() -> None:
        """
        Check for missing values in config.
        """
        ConfigStore.validate_recursive(ConfigStore.cfg)

    @staticmethod
    def validate_recursive(cfg: Any = None) -> None:
        if isinstance(cfg, ListConfig):
            for x in cfg:
                ConfigStore.validate_recursive(x)
        elif isinstance(cfg, DictConfig):
            for _, v in cfg.items():
                ConfigStore.validate_recursive(v)

    @staticmethod
    def save_config(path: str, filename: str = "all_config.yaml") -> None:
        path = os.path.abspath(os.path.join(path, filename))
        OmegaConf.save(ConfigStore.cfg, path)
        wandb.save(path)

    @staticmethod
    def print_config() -> None:
        print(OmegaConf.to_yaml(ConfigStore.cfg))

    @staticmethod
    def to_yaml_string() -> str:
        return OmegaConf.to_yaml(ConfigStore.cfg)
