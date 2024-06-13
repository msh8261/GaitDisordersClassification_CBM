import os
import pickle

# import numpy as np
# import math
# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import KFold

# torch.use_deterministic_algorithms(True, warn_only=True)
print(torch.__version__)
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision

import config.config_train as config
from nn.AE_vit_mlp import AutoEncoderViTMLP as AEvitmlp
from nn.discriminator import Discriminator_Conv
from src.adabound import AdaBound
from src.dataset import GaitData
from src.load import LoadData
# from src.loss import FocalLoss
from src.loss import LogCoshLoss

# from nn.discriminator import Discriminator_MLP



# from src.loss import XTanhLoss
# from src.loss import XSigmoidLoss
# from src.loss import VGGPerceptualLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes_ = config.params["num_class"]
input_size = config.params["input_size"]
sequence_length = config.params["sequences"]


weights = torch.tensor([1.0, 2.0, 2.0]).to(device)
# weights = torch.tensor([0.4, 0.77, 0.8]).to(device)


def get_features_from_model(model, img):
    img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
    encoder_features_1 = getattr(model.blocks_encoder, "0")(img[None].to(device))[None]
    encoder_features_2 = getattr(model.blocks_encoder, "1")(encoder_features_1[0])[None]
    classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
        encoder_features_2
    )[:, 0]
    mlp = getattr(model, "mlp")(classifier_head)
    mlp_h = (mlp + classifier_head)[0]
    return mlp_h


class GaitModel1(pl.LightningModule):
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
        super(GaitModel1, self).__init__()
        self.save_hyperparameters()
        self.cls_weight_1 = nn.Parameter(torch.abs(torch.randn(1)), requires_grad=True)
        self.cls_weight_2 = nn.Parameter(torch.abs(torch.randn(1)), requires_grad=True)
        self.loss_weight_1 = nn.Parameter(torch.abs(torch.rand(1)), requires_grad=True)
        self.loss_weight_2 = nn.Parameter(torch.abs(torch.rand(1)), requires_grad=True)

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

        self.model_g = AEvitmlp()

        self.model_d = Discriminator_Conv()

        optimizers_g = {
            "Adam": torch.optim.Adam(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "SGD": torch.optim.SGD(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                momentum=0.9,
                weight_decay=config.params["weight_decay"],
            ),
            "RMSprop": torch.optim.RMSprop(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adadelta": torch.optim.Adadelta(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adagrad": torch.optim.Adagrad(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adamax": torch.optim.Adamax(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adamw": torch.optim.AdamW(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "AdaBound": AdaBound(
                self.model_g.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
        }

        optimizers_d = {
            "Adam": torch.optim.Adam(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "SGD": torch.optim.SGD(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                momentum=0.9,
                weight_decay=config.params["weight_decay"],
            ),
            "RMSprop": torch.optim.RMSprop(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adadelta": torch.optim.Adadelta(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adagrad": torch.optim.Adagrad(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adamax": torch.optim.Adamax(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "Adamw": torch.optim.AdamW(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
            "AdaBound": AdaBound(
                self.model_d.parameters(),
                lr=config.params["learning_rate"],
                weight_decay=config.params["weight_decay"],
            ),
        }

        self.optimizer_g = optimizers_g[config.params["opt_indx"]]
        self.optimizer_d = optimizers_d[config.params["opt_indx"]]

        self.critrion_g = LogCoshLoss()

        self.critrion_d = nn.BCELoss()

        self.critrion3 = nn.MSELoss()

        self.critrion_c = nn.CrossEntropyLoss(weight=weights)

        # alpha = 1.0, gamma = 2.0
        alpha = 1.0
        gamma = 2.0
        # self.critrion_c = FocalLoss(alpha, gamma)

    def forward(self, x):
        return self.model_g(x)

    def find_loss_Gan(self, x, decoded):
        x = x.reshape(len(x), -1)
        real_pred = self.model_d(x)
        decoded = decoded.reshape(len(decoded), -1)
        fake_pred = self.model_d(decoded)

        loss_real_d = self.critrion_d(
            real_pred.squeeze(), torch.ones(len(x)).to(device)
        )
        loss_fake_d = self.critrion_d(
            fake_pred.squeeze(), torch.zeros(len(decoded)).to(device)
        )
        loss_fake_g = self.critrion_d(
            fake_pred.squeeze(), torch.ones(len(decoded)).to(device)
        )

        loss_d = loss_real_d + loss_fake_d
        loss_s = (loss_d + loss_fake_g) / 2
        return loss_s

    def find_loss_Gan_class(self, x, y, decoded):
        x = x.reshape(len(x), -1)
        real_pred = self.model_d(x)
        decoded = decoded.reshape(len(decoded), -1)
        fake_pred = self.model_d(decoded)

        loss_real_cls = self.critrion_d(real_pred.squeeze(), y.float())
        loss_fake_cls = self.critrion_d(fake_pred.squeeze(), y.float())

        return loss_real_cls, loss_fake_cls

    def customLoss(self, x, y, decoded, cls, mode):
        loss1 = self.critrion_g(decoded, x)

        loss2_dyn = self.critrion_c(cls, y)

        loss_cls = loss2_dyn

        loss_gan = self.find_loss_Gan(x, decoded)

        loss = (
            self.loss_weight_2 * loss_gan
            + self.cls_weight_1 * loss_cls
            + self.loss_weight_1 * loss1
        )

        if mode == "train":
            loss.backward()
        return loss

    def visualize(self):
        x, xpos, xneg, y = next(iter(self.train_dataloader))
        concat = torch.cat((x, xpos, xneg), 0)
        plt.figure(figsize=(24, 9))
        grid = torchvision.utils.make_grid(concat, nrow=config.params["batch_size"])
        plt.imshow(grid)

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
        kf = KFold(n_splits=self.num_splits, random_state=self.split_seed, shuffle=True)
        all_splits = [k for k in kf.split(self.X_train)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_set = GaitData(
            self.X_train[train_indexes], self.y_train[train_indexes]
        )
        self.val_set = GaitData(self.X_train[val_indexes], self.y_train[val_indexes])

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

        if input_size == 36:  # remove x,y features
            x = x[:, :, 34:]
        elif input_size == 34:  # only x,y features
            x = x[:, :, :34]
        elif input_size == 58:  # remove dist points from nose
            x = x[:, :, :58]
        elif input_size == 62:  # remove 8 symetric angles
            x1 = x[:, :, :34]
            key_points = [0, 2, 4, 6, 8, 10, 12, 14]
            x2 = x[:, :, 34:50]
            x2 = torch.index_select(x2, 2, torch.tensor(key_points).to(self.device))
            x3 = x[:, :, 50:]
            x = torch.dstack((x1, x2, x3))
        # elif input_size == 62: # remove 8 bones angles
        #     x1 = x[:, :, :50]
        #     x2 = x[:, :, 58:]
        #     x = torch.dstack((x1, x2))
        # elif input_size == 62: # remove 8 symetric dist
        #     x1 = x[:, :, :34]
        #     key_points = [1,3,5,7,9,11,13,15]
        #     x2 = x[:, :, 34:50]
        #     x2 = torch.index_select(x2,2,torch.tensor(key_points).to(self.device))
        #     x3 = x[:, :, 50:]
        #     x = torch.dstack((x1, x2, x3))

        return x, y

    def training_step(self, batch, batch_nb):

        x, y = self.set_data_get_data(batch)

        (decoded, cls) = self(x)

        reference_image = x[0]
        extracted_features = get_features_from_model(self.model_g, reference_image)
        dir_ = "./results/feats"
        os.makedirs(dir_, exist_ok=True)
        dir_data = "./results/feats/data"
        os.makedirs(dir_data, exist_ok=True)
        dir_lables = "./results/feats/labels"
        os.makedirs(dir_lables, exist_ok=True)
        dir_models = "./results/feats/models"
        os.makedirs(dir_models, exist_ok=True)
        if self.current_epoch % 10 == 0:
            with open(f"{dir_}/data/ext_feat_{self.current_epoch}.pkl", "wb") as file:
                pickle.dump(extracted_features, file)
            with open(f"{dir_}/labels/y_{self.current_epoch}.pkl", "wb") as file:
                pickle.dump(y[0], file)
            path_model = os.path.join(
                "results/feats/models", f"model_{self.current_epoch}.pt"
            )
            scripted_model = torch.jit.script(self.model_g)
            torch.jit.save(scripted_model, path_model)

        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss = self.customLoss(x, y, decoded, cls, "train")

        self.optimizer_g.step()
        self.optimizer_d.step()

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
        ave_train_loss = torch.tensor(
            [x["result"]["batch_train_loss"] for x in train_step_output]
        ).mean()
        ave_train_acc = torch.tensor(
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
        self.log("average_train_loss", ave_train_loss, prog_bar=True)
        self.log("average_train_acc", ave_train_acc, prog_bar=True)
        self.log("average_train_f1", avg_train_f1, prog_bar=True)
        self.log("average_train_precision", avg_train_precision, prog_bar=True)
        self.log("average_train_recall", avg_train_recall, prog_bar=True)

        avg_loss = torch.stack([x["loss"] for x in train_step_output]).mean()
        print("Loss train= {}".format(avg_loss))
        correct = sum([x["result"]["correct"] for x in train_step_output])
        total = sum([x["result"]["total"] for x in train_step_output])

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
        (decoded, cls) = self.model_g(x)
        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]
        loss = self.customLoss(x, y, decoded, cls, "valid")
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
        ave_val_loss = torch.tensor(
            [x["batch_val_loss"] for x in val_step_output]
        ).mean()
        ave_val_acc = torch.tensor([x["batch_val_acc"] for x in val_step_output]).mean()
        avg_val_f1 = torch.tensor([x["batch_val_f1"] for x in val_step_output]).mean()
        avg_val_precision = torch.tensor(
            [x["batch_val_precision"] for x in val_step_output]
        ).mean()
        avg_val_recall = torch.tensor(
            [x["batch_val_recall"] for x in val_step_output]
        ).mean()
        self.log("average_val_loss", ave_val_loss, prog_bar=True)
        self.log("average_val_acc", ave_val_acc, prog_bar=True)
        self.log("average_val_f1", avg_val_f1, prog_bar=True)
        self.log("average_val_precision", avg_val_precision, prog_bar=True)
        self.log("average_val_recall", avg_val_recall, prog_bar=True)

    def test_step(self, batch, batch_np):

        x, y = self.set_data_get_data(batch)

        (decoded, cls) = self.model_g(x)

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
        ave_test_acc = torch.tensor(
            [x["batch_test_acc"] for x in test_step_output]
        ).mean()
        ave_test_f1 = torch.tensor(
            [x["batch_test_f1"] for x in test_step_output]
        ).mean()
        self.log("average_test_acc", ave_test_acc, prog_bar=True)
        self.log("average_test_f1", ave_test_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer_g = self.optimizer_g
        optimizer_d = self.optimizer_d

        for param_group in optimizer_g.param_groups:
            param_group["lr"] *= config.params["lr_decay"]
        for param_group in optimizer_d.param_groups:
            param_group["lr"] *= config.params["lr_decay"]

        return [optimizer_g, optimizer_d]
