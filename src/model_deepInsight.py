# import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
# import math
import seaborn as sns
import torch
# torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# import torch.optim as optim
import config.config_train as config
from nn.pyDeepInsight.image_transformer import ImageTransformer
from nn.pyDeepInsight.utils import Norm2Scaler
from src.adabound import AdaBound
from src.dataset import GaitData
from src.load import LoadData
# from src.loss import FocalLoss
from src.loss import LogCoshLoss

# from torchsummary import summary

# from nn.AE_vit_mlp import AutoEncoderViTMLP as AEvitmlp




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


class ResNet(torch.nn.Module):
    def __init__(self, type_net="18"):
        super(ResNet, self).__init__()
        self.pixel_size = (224, 224)
        if type_net == "18":
            resnet_pretrained = torchvision.models.resnet18(pretrained=True)
            last_nodes = 512
        elif type_net == "50":
            resnet_pretrained = torchvision.models.resnet50(pretrained=True)
            last_nodes = 2048
        elif type_net == "152":
            resnet_pretrained = torchvision.models.resnet152(pretrained=True)
            last_nodes = 2048
        else:
            raise ValueError(f"{type_net} is nnot ResNet mode type.")
        self.model = resnet_pretrained
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(last_nodes, 128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes_),
        )
        self.couches_before_fc = list(self.model.children())[:-1]
        self.resnet_before_fc = nn.Sequential(*self.couches_before_fc)
        self.resnet_before_fc.fc = nn.Sequential(nn.Flatten(), self.model.fc[0])

    def forward(self, x):
        before_last_fc = self.resnet_before_fc(x)
        x = self.model(x)
        return x


class SqueezeNet(torch.nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        last_nodes = 1000
        self.pixel_size = (227, 227)
        self.model = torchvision.models.squeezenet1_1(weights="DEFAULT")
        self.model.classifier.append(nn.Flatten())
        self.model.classifier.append(nn.Dropout(0.1))
        _ = self.model.classifier.append(nn.Linear(last_nodes, num_classes_))

    def forward(self, x):
        return self.model(x)


class GaitModel_DI(pl.LightningModule):
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
        super().__init__()
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

        X_train = ld_tr.get_X()
        self.X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1] * X_train.shape[2]
        )
        self.y_train = ld_tr.get_y()

        X_test = ld_ts.get_X()
        self.X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
        self.y_test = ld_ts.get_y()

        self.batch_size = config.params["batch_size"]

        self.model_name = model_name

        if model_name == "deepInsight":
            # self.model = SqueezeNet()
            resnet_type = str(config.params["resnet_type"])
            self.model = ResNet(resnet_type).to(device)
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
        # self.acc = torchmetrics.functional.classification.multiclass_accuracy(pred, y, num_classes=num_classes_)
        self.f1_score = torchmetrics.functional.f1(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        # self.f1_score=torchmetrics.functional.classification.multiclass_f1_score(pred, y, num_classes=num_classes_, average="weighted")
        self.precision = torchmetrics.functional.precision(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        self.recall = torchmetrics.functional.recall(
            pred, y, num_classes=num_classes_, average="weighted"
        )
        # self.precision = torchmetrics.functional.classification.multiclass_precision(pred, y, num_classes=num_classes_, average="weighted")
        # self.recall = torchmetrics.functional.classification.multiclass_recall(pred, y, num_classes=num_classes_, average="weighted")
        # self.auc = torchmetrics.functional.auc(pred, y, reorder=True)
        self.specificity = torchmetrics.functional.specificity(
            pred, y, num_classes=num_classes
        )
        self.confmat = torchmetrics.functional.confusion_matrix(
            pred, y, num_classes=num_classes
        )
        # self.specificity = torchmetrics.functional.classification.multiclass_specificity(pred, y, num_classes=num_classes)
        # self.confmat = torchmetrics.functional.classification.multiclass_confusion_matrix(pred, y, num_classes=num_classes)
        return self

    def setup(self, stage=None):
        ln = Norm2Scaler()
        X_train_norm = ln.fit_transform(self.X_train)
        X_test_norm = ln.transform(self.X_test)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(self.y_train)
        y_test_enc = le.transform(self.y_test)
        num_classes = np.unique(y_train_enc).size

        distance_metric = "cosine"
        reducer = TSNE(
            n_components=2,
            metric=distance_metric,
            init="random",
            learning_rate="auto",
            n_jobs=-1,
        )

        pixel_size = self.model.pixel_size

        it = ImageTransformer(feature_extractor=reducer, pixels=pixel_size)

        it.fit(self.X_train, y=self.y_train, plot=False)

        X_train_img = it.transform(X_train_norm)
        X_test_img = it.transform(X_test_norm)

        preprocess = transforms.Compose([transforms.ToTensor()])

        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
        y_train_tensor = torch.from_numpy(le.fit_transform(self.y_train))

        self.X_test_tensor = torch.stack(
            [preprocess(img) for img in X_test_img]
        ).float()
        self.y_test_tensor = torch.from_numpy(le.transform(self.y_test))

        # for kfold method
        kf = KFold(n_splits=self.num_splits, random_state=self.split_seed, shuffle=True)
        all_splits = [k for k in kf.split(X_train_tensor)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_set = GaitData(
            X_train_tensor[train_indexes], y_train_tensor[train_indexes]
        )
        self.val_set = GaitData(
            X_train_tensor[val_indexes], y_train_tensor[val_indexes]
        )
        self.test_set = GaitData(self.X_test_tensor, self.y_test_tensor)
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
        # test_set = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False
        )

    def set_data_get_data(self, batch):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()
        return x, y

    def training_step(self, batch, batch_nb):
        if batch_nb == 0:
            x, y = batch
            self.x_samples = x
            self.reference_image = batch[0][0]  # .unsqueeze(0)
            # self.reference_image.resize((1,1,300,104))
            print(self.reference_image.shape)

        x, y = self.set_data_get_data(batch)

        cls = self(x)

        # cls = torch.nn.functional.log_softmax(cls)
        # preds = torch.argmax(cls, dim=1)
        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss = self.critrion2(cls, y)

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

        # if self.model_name == "deepInsight":
        #     self.show_image(self.reference_image)
        #     # Logging activations
        #     self.show_conv_featureMap(self.reference_image)

        # # # add graph for tesorboard visualization
        # # self.add_graph(self.x_samples)

        # # add images for tesorboard visualization
        # self.add_image(self.x_samples)

        # # Logging histograms
        # self.custom_histogram_adder()

        # self.custom_heatmap_adder(self.confmat, num_classes_)
        #
        print("Confusion matrix: ", self.confmat)
        print(
            "Number of Correctly identified Training Set Images {} from a set of {}. \nAccuracy= {} ".format(
                correct, total, correct / total
            )
        )

    def validation_step(self, batch, batch_nb):
        x, y = self.set_data_get_data(batch)

        cls = self(x)

        # cls = torch.nn.functional.log_softmax(cls)
        # preds = torch.argmax(cls, dim=1)
        cls = F.softmax(cls, dim=1)
        preds = cls.data.max(dim=1)[1]

        loss = self.critrion2(cls, y)

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

        cls = self(x)

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
        patience_ = 30
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

    def custom_heatmap_adder(self, confusion_matrix, num_classes):
        df_cm = pd.DataFrame(
            confusion_matrix.detach().cpu().numpy(),
            index=range(num_classes),
            columns=range(num_classes),
        )
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def custom_histogram_adder(self):
        # A custom defined function that adds Histogram to TensorBoard
        # Iterating over all parameters and logging them
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def add_graph(self, x):
        writer = torch.utils.tensorboard.SummaryWriter("lightning_logs/")
        model = self.model
        writer.add_graph(model, x)
        writer.close()

    def show_conv_featureMap(self, img):
        model_weights = []  # we will save the conv layer weights in this list
        conv_layers = []  # we will save the 49 conv layers in this list
        # get all the model children as list
        model_children = list(self.model.children())
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")

        # transform = transforms.Compose([
        #     transforms.Resize((227, 227)),
        #     transforms.ToTensor()
        # ])

        # # image = transform(img)
        # image = img
        # print(f"Image shape before: {image.shape}")
        # image = image.unsqueeze(0)
        # print(f"Image shape after: {image.shape}")
        # image = image.to(device)

        # outputs = []
        # names = []
        # for layer in conv_layers[0:]:
        #     image = layer(image)
        #     outputs.append(image)
        #     names.append(str(layer))
        # print(len(outputs))
        # #print feature_maps
        # for feature_map in outputs:
        #     print(feature_map.shape)

        # processed = []
        # for feature_map in outputs:
        #     feature_map = feature_map.squeeze(0)
        #     gray_scale = torch.sum(feature_map,0)
        #     gray_scale = gray_scale / feature_map.shape[0]
        #     processed.append(gray_scale.data.cpu().numpy())

        # fig = plt.figure(figsize=(30, 50))
        # for i in range(len(processed)):
        #     a = fig.add_subplot(5, 4, i+1)
        #     imgplot = plt.imshow(processed[i])
        #     a.axis("off")
        #     a.set_title(names[i].split('(')[0], fontsize=30)

        # # take a look at the conv layers and the respective weights
        # for weight, conv in zip(model_weights, conv_layers):
        #     # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        #     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

        # # visualize the first conv layer filters
        # plt.figure(figsize=(20, 17))
        # for i, filter in enumerate(model_weights[0]):
        #     plt.subplot(5, 5, i+1) # (8, 8) because in conv0 we have 4x4 filters and total of 64 (see printed shapes)
        #     plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
        #     plt.axis('off')
        #     #plt.savefig('../outputs/filter.png')
        # plt.show()

        # pass the image through all the layers
        results = [conv_layers[0](img)]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results

        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())
            for i, filter in enumerate(layer_viz):
                if i == 25:  # we will visualize only 8x8 blocks from each layer
                    break
                plt.subplot(5, 5, i + 1)
                plt.imshow(filter, cmap="gray")
                plt.axis("off")
            print(f"Saving layer {num_layer} feature maps...")
            # plt.savefig(f"../outputs/layer_{num_layer}.png")
            # plt.show()
            plt.close()

    def show_image(self, img):
        x = img
        # Evaluating the batch data as it moves forward in the netowrk
        # Custom made function for this model to log activations
        print("shape of input: ", x[0].shape)
        # plt.imshow(torch.Tensor.cpu(x[0][0]))
        plt.imshow(torch.Tensor.cpu(x[0]))

        # Logging the input image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x[0]), self.current_epoch, dataformats="HW"
        )
        # self.logger.experiment.add_image("input",torch.Tensor.cpu(x[0][0]),self.current_epoch,dataformats="HW")

        plt.show()
        plt.clf()

        # print("layer input: ", x.shape)
        # #out = self.model.layers(x)
        # out = self.model.layers[0](x)
        # print("shape of out: ", out.shape)
        # outer=(torch.Tensor.cpu(out).detach())
        # print("shape of outer layer1: ", outer.shape)

    def showActivations_conv2d(self, img):
        x = img
        # Evaluating the batch data as it moves forward in the netowrk
        # Custom made function for this model to log activations
        print("shape of input: ", x[0].shape)
        # plt.imshow(torch.Tensor.cpu(x[0][0]))
        plt.imshow(torch.Tensor.cpu(x[0]))

        # Logging the input image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x[0]), self.current_epoch, dataformats="HW"
        )
        # self.logger.experiment.add_image("input",torch.Tensor.cpu(x[0][0]),self.current_epoch,dataformats="HW")

        plt.show()
        plt.clf()

        print("layer input: ", x.shape)
        # out = self.model.layers(x)
        out = self.model.layers[0](x)
        print("shape of out: ", out.shape)
        outer = torch.Tensor.cpu(out).detach()
        print("shape of outer layer1: ", outer.shape)

        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(4 * outer.shape[2], 0)

        # Plotting for layer 1
        i = 0
        j = 0
        while i < 32:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == 4:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1

        plt.imshow(c)
        plt.show()
        plt.clf()
        self.logger.experiment.add_image(
            "layers", c, self.current_epoch, dataformats="HW"
        )

        out, _ = self.model.lstm(out)
        out = out[:, -1, :]
        outer = torch.Tensor.cpu(out).detach()
        print("shape of outer lstm: ", outer.shape)

        plt.figure(figsize=(10, 10))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(8 * outer.shape[2], 0)

        # Plotting for layer2
        i = 0
        j = 0
        while i < 64:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == 8:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1

        self.logger.experiment.add_image(
            "lstm", c, self.current_epoch, dataformats="HW"
        )
        plt.imshow(c)
        plt.show()
        plt.clf()

        # print(out.shape)
        out = self.model.fc(out)
        outer = torch.Tensor.cpu(out).detach()
        print("shape of outer fc: ", outer.shape)

        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(8 * outer.shape[2], 0)

        # Plotting for layer3
        j = 0
        i = 0
        while i < 128:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == 8:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        # print(c.shape)

        self.logger.experiment.add_image("fc", c, self.current_epoch, dataformats="HW")
        plt.imshow(c)
        plt.show()

    def add_image(self, x):
        writer = torch.utils.tensorboard.SummaryWriter("lightning_logs/")
        # create grid of images
        img_grid = torchvision.utils.make_grid(x)
        # show images
        matplotlib_imshow(img_grid, one_channel=True)
        # write to tensorboard
        writer.add_image("sample_images", img_grid)
        writer.close()

    # def show_train_images():
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #     for i in range(0, 3):
    #         ax[i].imshow(X_train_img[i])
    #         ax[i].title.set_text(f"Train[{i}] - class '{y_train[i]}'")
    #     plt.tight_layout()
    #     # plt.show()

    # def show_test_images():
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #     for i in range(0, 3):
    #         ax[i].imshow(X_test_img[i])
    #         ax[i].title.set_text(f"Test[{i}] - class '{y_test[i]}'")
    #     plt.tight_layout()
    #     # plt.show()
