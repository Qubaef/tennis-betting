import pandas as pd
import torch
from torch import device
from torch.utils.data import Dataset
import numpy as np
from ml.src.configuration.configuration import Config
from ml.src.phase import Phase


class TennisDataset(Dataset):
    def __init__(self, cfg: Config, phase: Phase = None):
        self.cfg = cfg
        self.data = pd.read_csv(cfg.dataset.data_path)
        if phase == Phase.TRAIN:
            self.data = self.data[self.data['startDate'] < '2017-01-01']
        elif phase == Phase.VAL:
            self.data = self.data[(self.data['startDate'] >= '2017-01-01') & (self.data['startDate'] < '2018-01-01')]
        elif phase == Phase.TEST:
            self.data = self.data[self.data['startDate'] >= '2018-01-01']
        self.label = self.data['winner'].to_numpy().astype(np.int64)
        self.data = self.data.drop(['winner', 'startDate', 'player1', 'player2'], axis=1)
        self.data = self.data.to_numpy().astype(np.float32)
        self.data = torch.from_numpy(self.data)
        self.label = torch.tensor(self.label)
        if torch.cuda.is_available():
            self.data = self.data.cuda(device(f"cuda:{cfg.training.gpu}"))
            self.label = self.label.cuda(device(f"cuda:{cfg.training.gpu}"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label.data[idx]

    def create_data_loader(self, phase: Phase) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=phase == Phase.TRAIN,
        )