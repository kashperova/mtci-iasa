import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size: int):
        self.features = data.iloc[:, :-1].values
        self.targets = data.iloc[:, -1].values
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size].reshape(-1)
        y = self.targets[idx + self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        ).unsqueeze(0)
