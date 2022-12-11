import pandas as pd
from numpy import genfromtxt
from torch.utils.data import Dataset
import numpy as np

from ml.src.configuration.configuration import Config


class TennisDataset(Dataset):
    def __init__(self, cfg: Config):
        self.data = pd.read_csv(cfg.dataset.data_path)
        self.target = self.data['target']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx], self.target.iloc[idx]

