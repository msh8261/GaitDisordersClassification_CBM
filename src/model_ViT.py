# import os
# import math
# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
# import torchvision
import torch.nn.functional as F
import torchmetrics
from sklearn.model_selection import KFold

import config.config_train as config
from nn.AE_vit_mlp import AutoEncoderViTMLP as AEvitmlp
from src.adabound import AdaBound
from src.dataset import GaitData
from src.load import LoadData
# from src.loss import FocalLoss
from src.loss import LogCoshLoss

# from src.loss import XTanhLoss
# from src.loss import XSigmoidLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes_ = config.params["num_class"]
input_size = config.params["input_size"]
sequence_length = config.params["sequences"]


weights = torch.tensor([1.0, 2.0, 2.0]).to(device)
# weights = torch.tensor([0.4, 0.77, 0.8]).to(device)


def matplotlib_imshow(img, one_channel=False):
    img = img.detach().cpu()
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class GaitModel(pl.LightningModule):
    def __init__(
        self,
        k,
        random_state,
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        model_name,
    ):
        super(GaitModel, self).__init__()
        self.save_hyperparameters()
        self.loss_weight_1 = 1
        self.loss_weight_2 = 1

        self.k = k
        self.num_splits = config.params["n_folds"]
        self.split_seed = random_state

        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path

        ld_tr = LoadData(
            X_train_path, y_train_path, config.params["num_augmentation"], True
        )
        # for test dataset augmentation shoud be set to 0
        ld_ts = LoadData(X_test_path, y_test_path, 0)

        self.X_train = ld_tr.get_X()
        self.y_train = ld_tr.get_y()

        self.X_test = ld_ts.get_X()
        self.y_test = ld_ts.get_y()

        self.batch_size = config.params["batch_size"]

        self.model_name = model_name

        if model_name == "vit_mlp":
            self.model = AEvitmlp()
        else:
            raise ValueError(f"{model_name} is nkonw model name")

        optimizers = {
            "Adam": torch.optim.Adam(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "SGD": torch.optim.SGD(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                momentum=0.9,
                weight_decay=config.params["weight_decay"],
            ),
            "RMSprop": torch.optim.RMSprop(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adadelta": torch.optim.Adadelta(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adagrad": torch.optim.Adagrad(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adamax": torch.optim.Adamax(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adamw": torch.optim.AdamW(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "AdaBound": AdaBound(
                self.model.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
        }

        self.optimizer = optimizers[config.params["opt_indx"]]

        # self.critrion1 = nn.MSELoss()
        self.critrion1 = LogCoshLoss()
        # self.critrion1 = XTanhLoss()
        # self.critrion1 = XSigmoidLoss()
        # self.critrion1 = nn.L1Loss(reduction='sum')

        self.critrion2 = nn.CrossEntropyLoss(weight=weights)
        # alpha = 1.0, gamma = 2.0
        alpha = 1.0
        gamma = 2.0
        # self.critrion2 = FocalLoss(alpha, gamma)

    def forward(self, x):
        return self.model(x)

    def metrics(self, pred, y, num_classes):
        torch.use_deterministic_algorithms(False)
        self.acc = torchmetrics.functional.accuracy(pred, y)
        self.f1_score = torchmetrics.functional.f1(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.precision = torchmetrics.functional.precision(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.recall = torchmetrics.functional.recall(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.auc = torchmetrics.functional.auc(pred, y, reorder=True)
        self.specificity = torchmetrics.functional.specificity(
            pred, y, num_classes=num_classes
        )
        self.confmat = torchmetrics.functional.confusion_matrix(
            pred, y, num_classes=num_classes
        )
        return self

    def setup(self, stage=None):
        # for kfold method
        kf = KFold(n_splits=self.num_splits, random_state=self.split_seed, shuffle=True)
        all_splits = [k for k in kf.split(self.X_train)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_set = GaitData(
            self.X_train[train_indexes], self.y_train[train_indexes]
        )
        self.val_set = GaitData(self.X_train[val_indexes], self.y_train[val_indexes])
        #########################################################
        ## if no need kfold method
        # train_percent = 0.7
        # test_percent = 0
        # val_percent = 1 - (train_percent + test_percent)

        # train_size = int(train_percent * len(self.X_train))
        # val_size = int(val_percent * (len(self.X_train)))
        # test_size = int(len(self.X_train) - (train_size + val_size))

        # data_train_val = GaitData(self.X_train, self.y_train)
        # self.train_set, self.val_set, _ = torch.utils.data.random_split(data_train_val, (train_size, val_size, test_size))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        data_test = GaitData(self.X_test, self.y_test)
        return torch.utils.data.DataLoader(
            data_test, batch_size=self.batch_size, shuffle=False
        )

    @property
    def automatic_optimization(self):
        return False

    def set_data_get_data(self, batch):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()

        # if input_size == 70:
        #     key_points = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61,64,65]
        #     x1 = x[:, :, :68]
        #     x1 = torch.index_select(x1,2,torch.tensor(key_points).to(self.device))
        #     x2 = x[:, :, 68:]
        #     x = torch.dstack((x1, x2))

        # elif input_size == 34:
        #     key_points = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61,64,65]
        #     x1 = x[:, :, :68]
        #     x = torch.index_select(x1,2,torch.tensor(key_points).to(self.device))

        # elif input_size == 42:
        #     key_points = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61,64,65]
        #     x1 = x[:, :, :68]
        #     x1 = torch.index_select(x1,2,torch.tensor(key_points).to(self.device))
        #     #x3 = x[:, :, 68+16:68+16+8]
        #     x2 = x[:, :, 68:68+8]
        #     x = torch.dstack((x1, x2))

        # elif input_size == 34:
        #     x = x[:, :, :34]

        # elif input_size == 42:
        #     x1 = x[:, :, :34]
        #     x2 = x[:, :, 52:]
        #     x = torch.dstack((x1, x2))

        # elif input_size == 60:
        #     x1 = x[:, :, :34]
        #     x2 = x[:, :, 52:]
        #     x3 = x[:, :, 34:52]
        #     x = torch.dstack((x1, x2, x3))

        return x, y

    def training_step(self, batch, batch_nb):
        if batch_nb == 0:
            x, y = batch
            self.x_samples = x
            self.reference_image = (batch[0][0]).unsqueeze(0)
            # self.reference_image.resize((1,1,300,104))
            print(self.reference_image.shape)

        x, y = self.set_data_get_data(batch)

        (decoded, cls) = self(x)

        # cls = torch.nn.functional.log_softmax(cls)
        # preds = torch.argmax(cls, dim=1)
        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss1 = self.critrion1(decoded, x)
        loss2 = self.critrion2(cls, y)

        loss = self.loss_weight_1 * loss1 + self.loss_weight_2 * loss2
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)

        self.optimizer.step()

        cl = self.metrics(preds, y, num_classes_)
        acc = cl.acc
        f1_score = cl.f1_score
        precision = cl.precision
        recall = cl.recall
        specificity = cl.specificity
        confmat = cl.confmat

        correct = cls.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        dic = {
            "batch_train_loss": loss,
            "batch_train_acc": acc,
            "batch_train_f1": f1_score,
            "batch_train_precision": precision,
            "batch_train_recall": recall,
            "correct": correct,
            "total": total,
        }
        self.log("batch_train_loss", loss, prog_bar=True)
        self.log("batch_train_acc", acc, prog_bar=True)
        self.log("batch_train_f1", f1_score, prog_bar=True)
        self.log("batch_train_precision", precision, prog_bar=True)
        self.log("batch_train_recall", recall, prog_bar=True)
        return {"loss": loss, "result": dic}

    def training_epoch_end(self, train_step_output):
        ave_loss = torch.tensor(
            [x["result"]["batch_train_loss"] for x in train_step_output]
        ).mean()
        ave_acc = torch.tensor(
            [x["result"]["batch_train_acc"] for x in train_step_output]
        ).mean()
        avg_train_f1 = torch.tensor(
            [x["result"]["batch_train_f1"] for x in train_step_output]
        ).mean()
        avg_train_precision = torch.tensor(
            [x["result"]["batch_train_precision"] for x in train_step_output]
        ).mean()
        avg_train_recall = torch.tensor(
            [x["result"]["batch_train_recall"] for x in train_step_output]
        ).mean()
        self.log("average_train_loss", ave_loss, prog_bar=True)
        self.log("average_train_acc", ave_acc, prog_bar=True)
        self.log("average_train_f1", avg_train_f1, prog_bar=True)
        self.log("average_train_precision", avg_train_precision, prog_bar=True)
        self.log("average_train_recall", avg_train_recall, prog_bar=True)

        avg_loss = torch.stack([x["loss"] for x in train_step_output]).mean()
        print("Loss train= {}".format(avg_loss))
        correct = sum([x["result"]["correct"] for x in train_step_output])
        total = sum([x["result"]["total"] for x in train_step_output])
        # tensorboard_logs = {'loss': avg_loss,"Accuracy": correct/total}

        # Loggig scalars
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Train", correct / total, self.current_epoch
        )

        print("Confusion matrix: ", self.confmat)
        print(
            "Number of Correctly identified Training Set Images {} from a set of {}. \nAccuracy= {} ".format(
                correct, total, correct / total
            )
        )

    def validation_step(self, batch, batch_nb):
        x, y = self.set_data_get_data(batch)

        (decoded, cls) = self(x)

        # cls = torch.nn.functional.log_softmax(cls)
        # preds = torch.argmax(cls, dim=1)
        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss1 = self.critrion1(decoded, x)
        loss2 = self.critrion2(cls, y)
        loss = self.loss_weight_1 * loss1 + self.loss_weight_2 * loss2

        cl = self.metrics(preds, y, num_classes_)
        acc = cl.acc
        f1_score = cl.f1_score
        precision = cl.precision
        recall = cl.recall
        specificity = cl.specificity
        confmat = cl.confmat

        dic = {
            "batch_val_loss": loss,
            "batch_val_acc": acc,
            "batch_val_f1": f1_score,
            "batch_val_precision": precision,
            "batch_val_recall": recall,
        }
        self.log("batch_train_loss", loss, prog_bar=True)
        self.log("batch_train_acc", acc, prog_bar=True)
        self.log("batch_val_f1", f1_score, prog_bar=True, logger=True)
        self.log("batch_val_precision", precision, prog_bar=True, logger=True)
        self.log("batch_val_recall", recall, prog_bar=True, logger=True)

        return dic

    def validation_epoch_end(self, val_step_output):
        ave_loss = torch.tensor([x["batch_val_loss"] for x in val_step_output]).mean()
        ave_acc = torch.tensor([x["batch_val_acc"] for x in val_step_output]).mean()
        avg_val_f1 = torch.tensor([x["batch_val_f1"] for x in val_step_output]).mean()
        avg_val_precision = torch.tensor(
            [x["batch_val_precision"] for x in val_step_output]
        ).mean()
        avg_val_recall = torch.tensor(
            [x["batch_val_recall"] for x in val_step_output]
        ).mean()
        self.log("average_val_loss", ave_loss, prog_bar=True)
        self.log("average_val_acc", ave_acc, prog_bar=True)
        self.log("average_val_f1", avg_val_f1, prog_bar=True)
        self.log("average_val_precision", avg_val_precision, prog_bar=True)
        self.log("average_val_recall", avg_val_recall, prog_bar=True)

    def test_step(self, batch, batch_np):
        x, y = self.set_data_get_data(batch)

        (decoded, cls) = self(x)

        # cls = torch.nn.functional.log_softmax(cls)
        # preds = torch.argmax(cls, dim=1)
        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        cl = self.metrics(preds, y, num_classes_)
        acc = cl.acc
        f1_score = cl.f1_score

        dic = {"batch_test_acc": acc, "batch_test_f1": f1_score}
        self.log("batch_test_acc", acc, prog_bar=True)
        self.log("batch_test_f1", f1_score, prog_bar=True)
        return dic

    def test_epoch_end(self, test_step_output):
        ave_acc = torch.tensor([x["batch_test_acc"] for x in test_step_output]).mean()
        ave_f1 = torch.tensor([x["batch_test_f1"] for x in test_step_output]).mean()
        self.log("average_test_acc", ave_acc, prog_bar=True)
        self.log("average_test_f1", ave_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer
        factor_ = 0.5
        patience_ = 50
        min_lr_ = 1e-15
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor_,
            patience=patience_,
            min_lr=min_lr_,
            verbose=True,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] *= config.params["lr_decay"]
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "average_val_loss",
        }
