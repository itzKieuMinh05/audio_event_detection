import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load spectrogram
        spec = np.load(row['feature_path'])

        # normalize (quan trọng)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        # add channel dim (AST cần)
        spec = np.expand_dims(spec, axis=0)

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(row['label'], dtype=torch.long)