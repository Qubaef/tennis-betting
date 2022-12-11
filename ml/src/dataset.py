import pandas as pd
import torch
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get raw
        raw = self.data.iloc[idx]
        # Get label
        label = raw['winner']
        # drap winner, 'startDate', 'player1', 'player2'
        raw = raw.drop(['winner', 'startDate', 'player1', 'player2'])
        raw = raw.to_numpy().astype(np.float32)
        raw = torch.from_numpy(raw)
        label = torch.tensor(label).long()

        return raw, label

    def create_data_loader(self, phase: Phase) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=phase == Phase.TRAIN,
        )