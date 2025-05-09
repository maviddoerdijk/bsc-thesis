import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def create_sliding_dataset(mat: np.ndarray,
                           x_scaler: MinMaxScaler,
                           y_scaler: MinMaxScaler,
                           look_back: int = 20):
    """
    X  -> (samples, look_back, features)
    y  -> (samples, 1)   — the next-step Spread_Close (just 1 day in advance)
    """
    X, y = [], []
    for i in range(len(mat) - look_back):
        X.append(mat[i : i + look_back, :]) # window
        y.append(mat[i + look_back, 0]) # value right after the window
    X, y = np.array(X), np.array(y).reshape(-1, 1)

    # scale per feature (fit on the training set once!)
    X_scaled = x_scaler.fit_transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape)
    y_scaled = y_scaler.fit_transform(y)

    return X, X_scaled, y, y_scaled


class SlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        #  cast to float32 once to avoid repeated conversions
        self.X = torch.tensor(X, dtype=torch.float32)      # (N, L, F)
        self.y = torch.tensor(y, dtype=torch.float32)      # (N, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]                    # each X: (L, F)

def create_sliding_dataset(mat: np.ndarray,
                           x_scaler: MinMaxScaler,
                           y_scaler: MinMaxScaler,
                           look_back: int = 20):
    """
    X  -> (samples, look_back, features)
    y  -> (samples, 1)   — the next-step Spread_Close (just 1 day in advance)
    """
    X, y = [], []
    for i in range(len(mat) - look_back):
        X.append(mat[i : i + look_back, :]) # window
        y.append(mat[i + look_back, 0]) # value right after the window
    X, y = np.array(X), np.array(y).reshape(-1, 1)

    # scale per feature (fit on the training set once!)
    X_scaled = x_scaler.fit_transform(
        X.reshape(-1, X.shape[-1])
    ).reshape(X.shape)
    y_scaled = y_scaler.fit_transform(y)

    return X, X_scaled, y, y_scaled
