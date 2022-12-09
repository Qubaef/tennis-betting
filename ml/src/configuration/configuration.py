import os
from dataclasses import dataclass, field


@dataclass
class Dataset:
    root_path: str
    metadata_path: str
    data_path: str


@dataclass
class Model:
    name: str
    n_layers: int
    n_features: int


@dataclass
class Training:
    gpu: int
    epochs: int
    batch_size: int
    num_workers: int
    seed: int


@dataclass
class Config:
    dataset: Dataset = field(default_factory=Dataset)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)