import math
import os
import random

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn
from torch.utils.data import Dataset


class GaitData(Dataset):
    def __init__(self, X, Y):
        self.X = np.round(X, 4)
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GaitData2(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xx = self.X[idx]
        yy = self.y[idx]
        xPos = random.choice([x for i, x in enumerate(self.X) if self.y[i] == yy])
        xneg = random.choice([x for i, x in enumerate(self.X) if self.y[i] != yy])

        return xx, xPos, xneg, yy
