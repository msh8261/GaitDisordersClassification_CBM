import math
import os

import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config.config_train as config
import src.augmentation as aug
from nn.genomap.genoClassification import genoClassification
from src.load import LoadData

# from nn.genomap.genomap import select_random_values



train_dataset_path = config.params["train_dataset_path"]

X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
y_train_path = os.path.join(train_dataset_path, "ytrain.File")
X_test_path = os.path.join(train_dataset_path, "Xtest.File")
y_test_path = os.path.join(train_dataset_path, "ytest.File")


ld_tr = LoadData(X_train_path, y_train_path, config.params["num_augmentation"], True)
# for test dataset augmentation shoud be set to 0
ld_ts = LoadData(X_test_path, y_test_path, 0)

X_train = ld_tr.get_X()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
y_train = ld_tr.get_y()
y_train = y_train.reshape(y_train.shape[0])

X_test = ld_ts.get_X()
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
y_test = ld_ts.get_y()
y_test = y_test.reshape(y_test.shape[0])

colNum = 33
rowNum = 33


est = genoClassification(
    X_train, y_train, X_test, rowNum=rowNum, colNum=colNum, epoch=100
)
print(est)
print(y_test)
print(
    "Classification accuracy of genomap approach:"
    + str(np.sum(est == y_test) / est.shape[0])
)
