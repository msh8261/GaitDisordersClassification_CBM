import os
import pickle as pk
import random
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchcam.methods import (CAM, ISCAM, SSCAM, GradCAM, GradCAMpp, LayerCAM,
                              ScoreCAM, SmoothGradCAMpp, XGradCAM)
from torchcam.utils import overlay_mask
from torchsummary import summary
from torchvision.transforms.functional import normalize, resize, to_pil_image

import config.config_train as config
from nn.pyDeepInsight.image_transformer import ImageTransformer
from nn.pyDeepInsight.utils import Norm2Scaler
from src.load import LoadData

warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(train_dataset_path):
    X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
    y_train_path = os.path.join(train_dataset_path, "ytrain.File")
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")
    y_test_path = os.path.join(train_dataset_path, "ytest.File")

    ld_tr = LoadData(
        X_train_path, y_train_path, config.params["num_augmentation"], True
    )
    # for test dataset augmentation shoud be set to 0
    ld_ts = LoadData(X_test_path, y_test_path, 0)

    X_train = ld_tr.get_X()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    y_train = ld_tr.get_y()

    X_test = ld_ts.get_X()
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    y_test = ld_ts.get_y()
    return X_train, y_train, X_test, y_test


def transform_data(X_train, y_train, X_test, y_test, batch_size, pixel_size):
    ln = Norm2Scaler()
    X_train_norm = ln.fit_transform(X_train)
    X_test_norm = ln.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = np.unique(y_train_enc).size

    distance_metric = "cosine"
    reducer = TSNE(
        n_components=2,
        metric=distance_metric,
        init="random",
        learning_rate="auto",
        n_jobs=-1,
    )

    it = ImageTransformer(feature_extractor=reducer, pixels=pixel_size)

    it.fit(X_train, y=y_train, plot=True)

    X_train_img = it.transform(X_train_norm)
    X_test_img = it.transform(X_test_norm)

    return X_train_img, X_test_img, le


class ResNet(torch.nn.Module):
    def __init__(self, type_net="18", num_classes=3):
        super(ResNet, self).__init__()
        self.pixel_size = (224, 224)
        if type_net == "18":
            self.name = "ResNet" + type_net
            resnet_pretrained = torchvision.models.resnet18(pretrained=True)
            last_nodes = 512
            # self.pixel_size = (224,224)
        elif type_net == "50":
            self.name = "ResNet" + type_net
            resnet_pretrained = torchvision.models.resnet50(pretrained=True)
            last_nodes = 2048
            # self.pixel_size = (224,224)
        elif type_net == "152":
            self.name = "ResNet" + type_net
            resnet_pretrained = torchvision.models.resnet152(pretrained=True)
            last_nodes = 2048
            # self.pixel_size = (256, 256)
        else:
            raise ValueError(f"{type_net} is not ResNet mode type.")
        self.model = resnet_pretrained
        self.model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(last_nodes, 128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.couches_before_fc = list(self.model.children())[:-1]
        self.resnet_before_fc = nn.Sequential(*self.couches_before_fc)
        self.resnet_before_fc.fc = nn.Sequential(nn.Flatten(), self.model.fc[0])

    def forward(self, x):
        before_last_fc = self.resnet_before_fc(x)
        x = self.model(x)
        return x


class SqueezeNet(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SqueezeNet, self).__init__()
        self.name = "SqueezeNet"
        last_nodes = 1000
        self.pixel_size = (227, 227)
        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(last_nodes, 128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )
        self.couches_before_fc = list(self.model.children())[:-1]
        self.model_before_fc = nn.Sequential(*self.couches_before_fc)
        self.model_before_fc.fc = nn.Sequential(nn.Flatten(), self.model.fc[0])

    def forward(self, x):
        before_last_fc = self.model_before_fc(x)
        x = self.model(x)
        return x


def get_network(num_classes, model_name="SqueezeNet"):
    if model_name == "SqueezeNet":
        net = SqueezeNet(num_classes)
    elif model_name == "ResNet18":
        net = ResNet("18", num_classes)
    elif model_name == "ResNet50":
        net = ResNet("50", num_classes)
    elif model_name == "ResNet152":
        net = ResNet("152", num_classes)
    else:
        raise ValueError(f"{model_name} is unkonw model name")
    return net


def train(model, le, X_train_img, X_test_img, epochs, batch_size):

    preprocess = transforms.Compose([transforms.ToTensor()])

    X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
    y_train_tensor = torch.from_numpy(le.fit_transform(y_train))

    X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
    y_test_tensor = torch.from_numpy(le.transform(y_test))

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model.train()
    model.to(device)
    weights = torch.tensor([1.0, 2.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-03,
        # momentum=0.8,
        weight_decay=1e-05,
    )

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # print epoch statistics
        if not (epoch % 20):
            print(
                f"[{epoch}] loss: {running_loss / len(X_train_tensor) * batch_size:.3f}"
            )
    print(f"[{epoch}] loss: {running_loss / len(X_train_tensor) * batch_size:.3f}")

    model.to("cpu")
    model = model.eval()
    with torch.no_grad():
        y_hat = model(X_train_tensor)
    train_predicted = torch.max(y_hat, 1)[1]

    with torch.no_grad():
        y_hat = model(X_test_tensor)
    test_predicted = torch.max(y_hat, 1)[1]

    print(
        f"The train accuracy was {accuracy_score(train_predicted, y_train_tensor):.3f}"
    )
    print(f"The test accuracy was {accuracy_score(test_predicted, y_test_tensor):.3f}")

    disp = ConfusionMatrixDisplay.from_predictions(
        test_predicted, y_test_tensor, display_labels=le.classes_
    )
    disp.plot()
    plt.show()


def show_conv_featureMap(net, X_img, pixel_size):
    model = net.model
    model_name = net.name
    random.seed(8)
    ix = np.random.choice(25)
    img = X_img[ix]
    # summary(model, input_size=(img.shape[2], img.shape[0], img.shape[1]), batch_size=1)
    # print(model)
    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    # get all the model children as list
    model_children = list(model.children())
    # model_children = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        model_children[i] = model_children[i].double()
        if type(model_children[i]) == nn.Conv2d:
            if model_name == "SqueezeNet" and model_children[i].padding != (0, 0):
                continue
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

    img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
    image = torch.from_numpy(img)
    if image.dim() < 4:
        # add batch with size 1
        image = image[None]

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        print("=============")
        print(layer)
        print(image.shape)
        image = layer(image)
        print(image.shape)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    # print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    processed[:0] = [img.reshape(img.shape[1], img.shape[2], img.shape[0])]
    names[:0] = ["Input Image"]
    print(len(processed))
    if model_name == "ResNet152":
        fig = plt.figure(figsize=(20, 32))
    else:
        fig = plt.figure(figsize=(6, 8))
    fig.suptitle(f"Result of {model_name}")
    for i in range(len(processed)):
        if model_name == "ResNet18":
            a = fig.add_subplot(5, 4, i + 1)
        elif model_name == "ResNet50":
            a = fig.add_subplot(8, 7, i + 1)
        elif model_name == "ResNet152":
            a = fig.add_subplot(10, 16, i + 1)
        elif model_name == "SqueezeNet":
            a = fig.add_subplot(6, 5, i + 1)
        plt.imshow(processed[i] * 255, interpolation="nearest", aspect="auto")
        a.axis("off")
        if i == 0:
            if model_name == "ResNet152":
                a.set_title(names[i].split("(")[0], fontsize=6)
            else:
                a.set_title(names[i].split("(")[0], fontsize=10)
        else:
            if model_name == "ResNet152":
                a.set_title(names[i].split("(")[0] + f"_{i-1}", fontsize=6)
            else:
                a.set_title(names[i].split("(")[0] + f"_{i-1}", fontsize=9)
    plt.tight_layout()
    plt.savefig("deepInsight_FMap.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_created_images(X_img, y, mode="Train"):
    import random

    random.seed(8)
    samples = np.random.randint(0, 20, size=3)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(len(samples)):
        ax[i].imshow(X_img[samples[i]] * 255, interpolation="nearest", aspect="auto")
        ax[i].title.set_text(f"{mode}[{i}] - class '{y[i]}'")
    plt.tight_layout()
    plt.show()


def show_activation_map(net, X_img, y_test):
    model = net.model
    model_name = net.name
    # CAM, GradCAM, GradCAMMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
    cam_extractor = SmoothGradCAMpp(model, "layer4")
    n = 25
    am = []
    targets = []
    for ix in range(n):
        img = X_img[ix]
        y = y_test[ix]
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        img = torch.from_numpy(img)

        if img.dim() < 4:
            # add batch with size 1
            img4 = img[None]

        model_children = list(model.children())
        for i in range(len(model_children)):
            model_children[i] = model_children[i].double()

        out = model(img4)
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        am.append(activation_map[0].squeeze(0).numpy())
        targets.append(y)

    indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        # Visualize the raw CAM
        plt.imshow(X_img[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
    plt.tight_layout()
    plt.savefig("deepInsight_images.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Result of activation map")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        # Visualize the raw CAM
        plt.imshow(am[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            # a.set_title(f'class {targets[indexes[i]]}', fontsize=10)
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
        # # Resize the CAM and overlay it
        # result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # # Display it
        # plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    plt.tight_layout()
    plt.savefig("deepInsight_CAM.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_all_process(model_, model, X_img, y_test):
    # print(model)
    # CAM, GradCAM, GradCAMMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
    cam_extractor = SmoothGradCAMpp(model, "layer4")

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    # get all the model children as list
    model_children = list(model.children())
    # model_children = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        model_children[i] = model_children[i].double()
        if type(model_children[i]) == nn.Conv2d:
            if model_name == "SqueezeNet" and model_children[i].padding != (0, 0):
                continue
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

    n = 25
    lbs = []
    processed = []
    names = []
    indexes = sorted(range(len(y_test)), key=lambda k: y_test[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    indexes_2 = [indexes[0], indexes[7], indexes[12]]
    for ix in indexes_2:
        im = X_img[ix]
        y = y_test[ix]
        img = im.reshape(im.shape[2], im.shape[0], im.shape[1])
        img = torch.from_numpy(img)

        if img.dim() < 4:
            # add batch with size 1
            img4 = img[None]
        else:
            img4 = img

        out = model(img4)
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        names.append("Input sample")
        names.append("Smooth Grad CAMpp")
        processed.append(im)
        processed.append(activation_map[0].squeeze(0).numpy())

        image = img
        outputs = []
        for i, layer in enumerate(conv_layers[0:]):
            image = layer(image)
            # if i==20 or i==51 or i==101 or i==150:
            if i == 150:
                outputs.append(image)
                names.append(f"Conv layer{i}")

        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        last_conv_res = torch.stack(outputs[-1:])
        flt = nn.Flatten()(last_conv_res).float()
        fc = nn.Linear(flt.shape[1], 100)(flt).detach().cpu().numpy()

        # names.append('FC layer')
        # processed.append(fc.T)
        lbs.append(y)

    num_fig_per_row = int(len(processed) / 3)

    sns.set_theme(style="ticks", font_scale=1)
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f"Result of deepInsight method")

    for i in range(len(processed)):
        a = fig.add_subplot(3, num_fig_per_row, i + 1)
        # Visualize the raw CAM
        plt.imshow(processed[i] * 255, interpolation="nearest", aspect="auto")
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        a.set_title(names[i].split("(")[0], fontsize=8)
        if i == 0:
            a.set_ylabel(f"class {lbs[0]+1}", rotation=90, weight="bold", fontsize=10)
        elif i == num_fig_per_row:
            a.set_ylabel(f"class {lbs[1]+1}", rotation=90, weight="bold", fontsize=10)
        elif i == num_fig_per_row * 2:
            a.set_ylabel(f"class {lbs[2]+1}", rotation=90, weight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig("deepInsight_all_process.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("deepInsight_all_process.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def show_tsne(model_, model, X_img, y_):
    # print(net)
    # CAM, GradCAM, GradCAMMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
    cam_extractor = SmoothGradCAMpp(model, "layer4")

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    # get all the model children as list
    model_children = list(model.children())
    # model_children = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        model_children[i] = model_children[i].double()
        if type(model_children[i]) == nn.Conv2d:
            if model_name == "SqueezeNet" and model_children[i].padding != (0, 0):
                continue
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

    lbs = []
    proc_imgs = []
    proc_convs = []
    proc_flts = []
    proc_fcs = []

    for ix in range(len(X_img)):
        im = X_img[ix]
        y = y_[ix]
        img = im.reshape(im.shape[2], im.shape[0], im.shape[1])
        img = torch.from_numpy(img)

        if img.dim() < 4:
            # add batch with size 1
            img4 = img[None]
        else:
            img4 = img

        out = model(img4)
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        proc_imgs.append(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]))

        image = img
        outputs = []
        for i, layer in enumerate(conv_layers[0:]):
            image = layer(image)
            if i == 20 or i == 51 or i == 101 or i == 150:
                outputs.append(image)

        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]

        last_conv_res = torch.stack(outputs[-1:])
        proc_convs.append(
            last_conv_res.detach()
            .cpu()
            .numpy()
            .reshape(
                last_conv_res.shape[0]
                * last_conv_res.shape[1]
                * last_conv_res.shape[2]
                * last_conv_res.shape[3]
            )
        )
        flt = nn.Flatten()(last_conv_res).float()
        proc_flts.append(
            flt.detach().cpu().numpy().reshape(flt.shape[0] * flt.shape[1])
        )
        fc = nn.Linear(flt.shape[1], 100)(flt).detach().cpu().numpy()
        proc_fcs.append(fc.reshape(fc.shape[0] * fc.shape[1]))

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_convs = np.array(proc_convs)
    X_flts = np.array(proc_flts)
    X_fcs = np.array(proc_fcs)

    tsns_inputs = tsne_(X_imgs, y, "Inputs")
    tsns_flatten = tsne_(X_flts, y, "Flattens")

    # plot the result
    sns.set_theme(style="ticks", font_scale=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.scatter(
        tsns_inputs["Dimension 1"],
        tsns_inputs["Dimension 2"],
        marker="o",
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    fig.colorbar(
        im1, ax=ax1, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    ax1.set_title("t-SNE Visualization of input", fontweight="bold")
    # ax1.figure.set_size_inches(6.5, 4.5)
    # ax1.ax.margins(.15)
    # ax1.despine(trim=True)
    im2 = ax2.scatter(
        tsns_flatten["Dimension 1"],
        tsns_flatten["Dimension 2"],
        marker="o",
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    # plt.colorbar(ticks=range(3), label='Classes', boundaries=np.arange(4)-0.5)
    fig.colorbar(
        im2, ax=ax2, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    ax2.set_title("t-SNE Visualization of output features", fontweight="bold")
    # ax2.figure.set_size_inches(6.5, 4.5)
    # ax2.ax.margins(.15)
    # ax2.despine(trim=True)
    plt.suptitle("TSNE Result (deepInsight method)")
    plt.savefig(f"tsne_deepInsight.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"tsne_deepInsight.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def pca_(X, y, name):
    # set the components to 2
    pca = PCA(n_components=2, whiten=True)
    # fit the model to our data and extract the results
    X_pca = pca.fit_transform(X)

    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_pca, columns=["Dimension 1", "Dimension 2"])

    df["class"] = y
    # plot the resulting data from two dimensions
    sns.set_theme(style="ticks", font_scale=1)
    # d = sns.displot(data=df, x="Dimension 1", col="Dimension 2", kde=True)
    # sns.catplot(data=df, kind="bar", x="Dimension 1", y="Dimension 2", hue="class")
    g = sns.relplot(
        data=df,
        x="Dimension 1",
        y="Dimension 2",
        hue="class",
        palette="crest",
        # marker="x",
        s=50,
    )
    g.set_axis_labels("Dimension 1", "Dimension 2", labelpad=10)
    g.fig.suptitle("PCA Result")
    g.legend.set_title("class")
    g.figure.set_size_inches(6.5, 4.5)
    g.ax.margins(0.15)
    g.despine(trim=True)
    g.savefig(f"pca_{name}.pdf")


def tsne_(X, y, name):
    # create the model
    tsne = TSNE(
        n_components=2,
        perplexity=3,
        learning_rate="auto",
        init="random",
        random_state=21,
        n_iter=5000,
        n_jobs=-1,
    )
    # apply it to the data
    X_dimensions = tsne.fit_transform(X)
    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_dimensions, columns=["Dimension 1", "Dimension 2"])

    df["class"] = y

    return df


def show_data_distribution(X, y_train):
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_rs = X.copy()

    # set the components to 2
    pca = PCA(n_components=2, whiten=True)
    # fit the model to our data and extract the results
    X_pca = pca.fit_transform(X_rs)
    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_pca, columns=["Dimension 1", "Dimension 2"])

    df["class"] = y_train
    # plot the resulting data from two dimensions
    sns.set_theme(style="ticks", font_scale=1)
    # d = sns.displot(data=df, x="Dimension 1", col="Dimension 2", kde=True)
    # sns.catplot(data=df, kind="bar", x="Dimension 1", y="Dimension 2", hue="class")
    g = sns.relplot(
        data=df,
        x="Dimension 1",
        y="Dimension 2",
        hue="class",
        palette="crest",
        # marker="x",
        s=50,
    )
    g.set_axis_labels("Dimension 1", "Dimension 2", labelpad=10)
    g.fig.suptitle("PCA Result")
    g.legend.set_title("class")
    g.figure.set_size_inches(6.5, 4.5)
    g.ax.margins(0.15)
    g.despine(trim=True)
    g.savefig("pca_data.pdf")

    # set the hyperparmateres
    keep_dims = 2
    lrn_rate = 700
    prp = 40
    # extract the data as a cop
    tsnedf = X_rs.copy()
    # creae the model
    tsne = TSNE(
        n_components=keep_dims, perplexity=prp, random_state=42, n_iter=5000, n_jobs=-1
    )
    # apply it to the data
    X_dimensions = tsne.fit_transform(tsnedf)
    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_dimensions, columns=["Dimension 1", "Dimension 2"])

    df["class"] = y_train

    # plot the result
    sns.set_theme(style="ticks", font_scale=1)
    # d = sns.catplot(data = df,
    #             kind="bar",
    #             x="Dimension 1",
    #             y="Dimension 2",
    #             hue="class",
    #             )
    g = sns.relplot(
        data=df,
        x="Dimension 1",
        y="Dimension 2",
        hue="class",
        palette="crest",
        # marker="x",
        s=50,
    )
    g.set_axis_labels("Dimension 1", "Dimension 2", labelpad=10)
    g.fig.suptitle("TSNE Result")
    g.legend.set_title("class")
    g.figure.set_size_inches(6.5, 4.5)
    g.ax.margins(0.15)
    g.despine(trim=True)
    g.savefig("tsne_data.pdf")


if __name__ == "__main__":
    train_dataset_path = config.params["train_dataset_path"]
    batch_size = 16
    epochs = 120
    num_classes = 3
    # ResNet18, ResNet50, ResNet152, SqueezeNet
    model_name = "ResNet152"

    net = get_network(num_classes, model_name)

    X_train, y_train, X_test, y_test = get_data(train_dataset_path)

    with open("xtrain.pkl", "rb") as f:
        X_train_img = pk.load(f)

    with open("xtest.pkl", "rb") as f:
        X_test_img = pk.load(f)

    data_df = pd.DataFrame(np.concatenate((X_train, X_test), axis=0))
    y_df = pd.DataFrame(np.concatenate((y_train, y_test), axis=0))
    y_df = y_df.astype(int)

    model_saved = torch.jit.load("./results/deepInsight/152/deepInsight_k0_rand21.pt")
    model_saved.to(device)
    model_saved.eval()

    show_all_process(model_saved, net.model, X_test_img, y_test)
