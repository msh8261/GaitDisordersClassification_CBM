import math
import os

import numpy as np

import config.config_train as config
import src.augmentation as aug


class LoadData(object):
    def __init__(self, X_path, y_path, num_aug, aug_mode=False):
        super(LoadData, self).__init__()

        self.X_path = X_path
        self.y_path = y_path
        # set True if need augmentation
        self.aug_mode = aug_mode
        # set number of augmentation
        self.num_aug = num_aug

    def get_X(self):
        file = open(self.X_path, "r")
        X = np.array([elem.split(",") for elem in file], dtype=np.float32)
        file.close()

        if self.aug_mode:
            X = self.data_augmentation(X).astype(np.float32)

        blocks = int(len(X) / config.params["sequences"])
        X_ = np.array(np.split(X, blocks))
        return X_

    def get_y(self):
        file = open(self.y_path, "r")
        y = np.array(
            [
                elem
                for elem in [row.replace("  ", " ").strip().split(" ") for row in file]
            ],
            dtype=np.int32,
        )
        file.close()
        y = y - 1

        if self.aug_mode:
            y = self.label_augmentation(y).astype(np.float32)

        return y

    def data_augmentation(self, data0):
        data0 = np.array(data0)
        blocks = int(len(data0) / (config.params["sequences"]))
        data0 = np.array(np.split(data0, blocks))

        collect = []
        for i, data in enumerate(data0):
            data = data.reshape(1, data.shape[0], data.shape[1])
            x1 = aug.jitter(data)
            x2 = aug.scaling(data)
            x3 = aug.magnitude_warp(data)
            x4 = aug.time_warp(data)
            x5 = aug.rotation(data)
            x6 = aug.window_slice(data)

            if self.num_aug == 0:
                X = data
            elif self.num_aug == 1:
                X = np.vstack((data, x1))
            elif self.num_aug == 2:
                X = np.vstack((data, x1, x2))
            elif self.num_aug == 3:
                X = np.vstack((data, x1, x2, x3))
            elif self.num_aug == 4:
                X = np.vstack((data, x1, x2, x3, x4))
            elif self.num_aug == 5:
                X = np.vstack((data, x1, x2, x3, x4, x5))
            elif self.num_aug == 6:
                X = np.vstack((data, x1, x2, x3, x4, x5, x6))
            else:
                assert print("INFO: please select a number between 0 and 6")

            collect.append(X)
        XX = np.vstack(collect)

        XX = XX.reshape(XX.shape[0] * XX.shape[1], XX.shape[2])

        return XX

    def label_augmentation(self, target):
        y = np.array(target)

        collect = []
        for i, y in enumerate(y):
            if self.num_aug == 0:
                yy = y
            elif self.num_aug == 1:
                yy = np.vstack((y, y))
            elif self.num_aug == 2:
                yy = np.vstack((y, y, y))
            elif self.num_aug == 3:
                yy = np.vstack((y, y, y, y))
            elif self.num_aug == 4:
                yy = np.vstack((y, y, y, y, y))
            elif self.num_aug == 5:
                yy = np.vstack((y, y, y, y, y, y))
            elif self.num_aug == 6:
                yy = np.vstack((y, y, y, y, y, y, y))
            else:
                assert print("INFO: please select a number between 0 and 6")

            collect.append(yy)
        yy = np.vstack(collect)

        return yy
