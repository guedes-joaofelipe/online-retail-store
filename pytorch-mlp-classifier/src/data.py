"""
This file contains all data preprocessing and selection.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def data_preprocess(x_path: str, y_path: str):
    
    X = pd.read_csv(x_path).to_numpy()
    y = pd.read_csv(y_path).to_numpy()

    return X, y
    
class SalesDataset(Dataset):

    def __init__(self, X: torch.tensor, y: torch.tensor):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.pos_weight = torch.tensor([len(y[y == 1]) / len(y)])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
