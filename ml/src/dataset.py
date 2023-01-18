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
        # data = []
        # # there is pairs of matches in data, so we need to remove one of them, choose one randomly
        # for i in range( len(self.data)):
        #     if i % 2 == 0:
        #         if np.random.randint(2) == 1:
        #             data.append(i)
        #         else:
        #             data.append(i+1)

        # select idx from data
        # self.data = self.data.iloc[data]
        self.data = self.data.sort_values(
            by=["startDate", "tournamentRound"], ascending=[True, False]
        )
        # multiply odds by 1.05
        # Train dataset until 2017
        if phase == Phase.TRAIN:
            self.data = self.data[self.data["startDate"] < "2016-01-01"]
        elif phase == Phase.VAL:
            self.data = self.data[
                (self.data["startDate"] >= "2016-01-01")
                & (self.data["startDate"] < "2017-01-01")
            ]
        # Test dataset from 2018 to
        elif phase == Phase.TEST:
            self.data = self.data[self.data["startDate"] >= "2017-01-01"]
        self.label = self.data["winner"].to_numpy().astype(np.int64)

        # Drop columns that are not needed
        self.data = self.drop_columns(self.data)

        # Convert data to tensor and copy to device
        self.data = self.data.to_numpy().astype(np.float32)
        self.data = torch.from_numpy(self.data)
        self.label = torch.tensor(self.label)
        if torch.cuda.is_available():
            self.data = self.data.cuda(device(f"cuda:{cfg.training.gpu}"))
            self.label = self.label.cuda(device(f"cuda:{cfg.training.gpu}"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    @staticmethod
    def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(
            [
                "winner",
                "startDate",
                "player1",
                "player2",
                "setsPlayed",
                "gamesPlayed",
                "oddsAvg1",
                "oddsAvg2",
            ],
            axis=1,
        )

    def create_data_loader(
        self, cfg: Config, phase: Phase
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=phase == Phase.TRAIN if cfg.model.name == "ANN" else False,
        )
