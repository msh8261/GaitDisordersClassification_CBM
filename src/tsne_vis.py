""" export CUBLAS_WORKSPACE_CONFIG=:4096:8 """

import os
# import random
import pickle

import genomap as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Please install pandas and matplotlib before you run this example
import scipy
import seaborn as sns
import torch
# import torchvision.transforms as transforms
# from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torchvision
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# import src.augmentation as aug
import config.config_train as config
from nn.pyDeepInsight.image_transformer import ImageTransformer
from nn.pyDeepInsight.utils import Norm2Scaler
from src.load import LoadData

# import matplotlib






# from torchcam.methods import SmoothGradCAMpp, CAM, GradCAM, GradCAMpp, ScoreCAM
# from torchcam.methods import SSCAM, ISCAM, XGradCAM, LayerCAM
# from torchcam.utils import overlay_mask
# from torchvision.transforms.functional import normalize, resize, to_pil_image
# import torch.nn.functional as F


# from genomap.genomap import construct_genomap
# import genomap.genoNet as gNet


# from nn.AE_vit_mlp import AutoEncoderViTMLP as AEvitmlp
# from nn.discriminator import Discriminator_MLP, Discriminator_Conv


# from src.model_ViT_GAN import GaitModel1


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
    y_train = ld_tr.get_y()

    X_test = ld_ts.get_X()
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


def show_tsne(model_pr, model_vit, model_di, model_gn, X_, X_di, X_gn, y_, nump):
    conv_layers = []
    conv1 = getattr(model_pr.model_d.block_conv1, "0")
    conv_layers.append(conv1)
    conv2 = getattr(model_pr.model_d.block_conv2, "0")
    conv_layers.append(conv2)
    conv3 = getattr(model_pr.model_d.block_conv3, "0")
    conv_layers.append(conv3)
    last_conv = conv_layers[0:][2]

    model_weights = []
    conv_layers_di = []
    model_children = list(model_di.children())
    counter = 0
    for i in range(len(model_children)):
        model_children[i] = model_children[i].double()
        if type(model_children[i]) == nn.Conv2d:
            if model_name == "SqueezeNet" and model_children[i].padding != (0, 0):
                continue
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers_di.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers_di.append(child)
    print(f"Total convolutional layers: {counter}")

    imgs = []
    proc_imgs = []
    proc_gen_fcs = []
    proc_pr_fcs = []
    proc_vit_fcs = []
    proc_di_fcs = []
    lbs = []
    for i in range(len(y_)):
        img = X_[i]
        img_gn = X_gn[i]
        img_di = X_di[i]
        lbs.append(y_[i])

        img_gn = img_gn.reshape(img_gn.shape[2], img_gn.shape[0], img_gn.shape[1])
        img_gn = torch.from_numpy(img_gn)

        if img_gn.dim() < 4:
            # add batch with size 1
            img4 = img_gn[None]
        else:
            img4 = img_gn

        model_children = list(model_gn.children())
        for i in range(len(model_children)):
            model_children[i] = model_children[i].double()

        out_gn = model_gn(img4)
        conv1_gn = getattr(model_gn, "conv1")
        fc1_gn = getattr(model_gn, "fc1")
        # conv1_gn = getattr(model_gn.model, 'conv1')
        # fc1_gn = getattr(model_gn.model, 'fc1')

        out_gn_conv = conv1_gn(img4)
        gn_fc1 = fc1_gn(out_gn_conv.contiguous().view(-1, nump * 8)).detach().numpy()

        img_pr = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_pr, out) = model_pr(img_pr)
        encoder_features_1 = getattr(model_pr.model_g.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model_pr.model_g.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        classifier_head = nn.LayerNorm(img_pr.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        mlp = getattr(model_pr.model_g, "mlp")(classifier_head)
        pr_fc = (mlp + classifier_head)[0]  # remove channel

        out, _ = model_vit(img_pr)
        encoder_features_1 = getattr(model_vit.model.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model_vit.model.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        vit_classifier_head = nn.LayerNorm(img_pr.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        vit_mlp = getattr(model_vit.model, "mlp")(classifier_head)
        vit_fc = (vit_mlp + vit_classifier_head)[0]  # remove channel

        image = img_di.reshape(img_di.shape[2], img_di.shape[0], img_di.shape[1])
        image = torch.from_numpy(image)
        outputs = []
        for i, layer in enumerate(conv_layers_di[0:]):
            image = layer(image)
            if i == 20 or i == 51 or i == 101 or i == 150:
                outputs.append(image)

        last_conv_res = torch.stack(outputs[-1:])
        di_fc = nn.Flatten()(last_conv_res).float()

        print("@@@@@@@@@@@@@")
        print(img.shape)
        print(pr_fc.shape)
        print(di_fc.shape)
        print(gn_fc1.shape)
        print("@@@@@@@@@@@@@")

        proc_imgs.append(img.reshape(img.shape[0] * img.shape[1]))
        proc_pr_fcs.append(
            pr_fc.reshape(pr_fc.shape[0] * pr_fc.shape[1]).detach().cpu().numpy()
        )
        proc_vit_fcs.append(
            vit_fc.reshape(vit_fc.shape[0] * vit_fc.shape[1]).detach().cpu().numpy()
        )
        proc_di_fcs.append(
            di_fc.reshape(di_fc.shape[0] * di_fc.shape[1]).detach().cpu().numpy()
        )
        proc_gen_fcs.append(gn_fc1.reshape(gn_fc1.shape[0] * gn_fc1.shape[1]))

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_pr_fcs = np.array(proc_pr_fcs)
    X_vit_fcs = np.array(proc_vit_fcs)
    X_di_fcs = np.array(proc_di_fcs)
    X_gen_fcs = np.array(proc_gen_fcs)

    tsns_inputs = tsne_(X_imgs, y)
    tsns_pr_fcs = tsne_(X_pr_fcs, y)
    tsns_vit_fcs = tsne_(X_vit_fcs, y)
    tsns_di_fcs = tsne_(X_di_fcs, y)
    tsns_gen_fcs = tsne_(X_gen_fcs, y)

    # plot the result
    sns.set_theme(style="ticks", font_scale=1)
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    plt.rcParams["axes.facecolor"] = "mistyrose"  # 'lanvender'
    plt.style.context("seaborn")
    im1 = axs[0][0].scatter(
        tsns_inputs["Dimension 1"],
        tsns_inputs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=1,
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    fig.colorbar(
        im1,
        ax=axs[0][0],
        ticks=range(3),
        label="Classes",
        boundaries=np.arange(4) - 0.5,
    )
    axs[0][0].set_title("t-SNE Visualization of input", fontweight="bold", fontsize=10)
    axs[0][0].get_xaxis().set_ticks([])
    axs[0][0].get_yaxis().set_ticks([])

    im2 = axs[0][1].scatter(
        tsns_pr_fcs["Dimension 1"],
        tsns_pr_fcs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=1,
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    fig.colorbar(
        im2,
        ax=axs[0][1],
        ticks=range(3),
        label="Classes",
        boundaries=np.arange(4) - 0.5,
    )
    axs[0][1].set_title(
        "t-SNE Visualization of proposed method features",
        fontweight="bold",
        fontsize=10,
    )
    axs[0][1].get_xaxis().set_ticks([])
    axs[0][1].get_yaxis().set_ticks([])

    im3 = axs[1][0].scatter(
        tsns_vit_fcs["Dimension 1"],
        tsns_vit_fcs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=1,
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    fig.colorbar(
        im3,
        ax=axs[1][0],
        ticks=range(3),
        label="Classes",
        boundaries=np.arange(4) - 0.5,
    )
    axs[1][0].set_title(
        "t-SNE Visualization of ViT method features", fontweight="bold", fontsize=10
    )
    axs[1][0].get_xaxis().set_ticks([])
    axs[1][0].get_yaxis().set_ticks([])

    im4 = axs[1][1].scatter(
        tsns_di_fcs["Dimension 1"],
        tsns_di_fcs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=1,
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    fig.colorbar(
        im4,
        ax=axs[1][1],
        ticks=range(3),
        label="Classes",
        boundaries=np.arange(4) - 0.5,
    )
    axs[1][1].set_title(
        "t-SNE Visualization of deepInsight features", fontweight="bold", fontsize=10
    )
    axs[1][1].get_xaxis().set_ticks([])
    axs[1][1].get_yaxis().set_ticks([])

    im5 = axs[2][0].scatter(
        tsns_gen_fcs["Dimension 1"],
        tsns_gen_fcs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=1,
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    fig.colorbar(
        im5,
        ax=axs[2][0],
        ticks=range(3),
        label="Classes",
        boundaries=np.arange(4) - 0.5,
    )
    axs[2][0].set_title(
        "t-SNE Visualization of genmap features", fontweight="bold", fontsize=10
    )
    axs[2][0].get_xaxis().set_ticks([])
    axs[2][0].get_yaxis().set_ticks([])

    plt.suptitle("TSNE Results")
    plt.savefig("tsne_all.pdf")
    plt.show()


def tsne_(X, y):
    print("==============")
    print(X.shape)
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


def select_n_features(X, n):
    # calculate the variance of each feature
    variances = np.var(X, axis=0)

    # get the indices of the features sorted by variance (in descending order)
    indices = np.argsort(variances)[::-1]

    # select the indices of the top 10 most variable features
    top_n_indices = indices[:n]

    # select the top 10 most variable features
    X_top_n = X[:, top_n_indices]
    return X_top_n, top_n_indices


if __name__ == "__main__":
    train_dataset_path = config.params["train_dataset_path"]
    batch_size = 16
    epochs = 120
    num_classes = 3
    # ResNet18, ResNet50, ResNet152, SqueezeNet
    model_name = "ResNet152"

    X_train, y_train, X_test, y_test = get_data(train_dataset_path)

    # model_deepInsight = torch.jit.load('./results/deepInsight/152/deepInsight_k0_rand21.pt')
    # model_deepInsight.to(device)
    # model_deepInsight.eval()

    num_classes = 3
    # ResNet18, ResNet50, ResNet152, SqueezeNet
    model_name = "ResNet152"
    model_deepInsight = get_network(num_classes, model_name).model

    # model_gnmap = torch.jit.load('./results/gnclassify/gnClassify_k2_rand21.pt')
    # model_gnmap.to('cpu')
    # model_gnmap.eval()

    with open("geno.pkl", "rb") as f:
        model_gnmap = pickle.load(f)

    model_proposed = torch.jit.load("./results/final/gan_k2_rand21.pt")
    model_proposed.to(device)
    model_proposed.eval()

    model_vit = torch.jit.load("./results/final/vit_mlp_k1_rand21.pt")
    model_vit.to(device)
    model_vit.eval()

    X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    X_train_img, X_test_img, labelencoder = transform_data(
        X_train_r, y_train, X_test_r, y_test, batch_size, (224, 224)
    )

    colNum = 36  # Column number of genomap
    rowNum = 36  # Row number of genomap

    data_test = X_test_r

    nump = colNum * rowNum
    if nump < data_test.shape[1]:
        data_test, index = select_n_features(data_test, nump)

    # Normalization of the data
    dataNorm_test = scipy.stats.zscore(data_test, axis=0, ddof=1)
    # Construction of genomaps
    genoMaps_test = gp.construct_genomap(
        dataNorm_test, rowNum, colNum, epsilon=0.0, num_iter=200
    )

    X_proposed = X_test
    X_deepInsight = X_test_img
    X_genomaps = genoMaps_test

    show_tsne(
        model_proposed,
        model_vit,
        model_deepInsight,
        model_gnmap,
        X_proposed,
        X_deepInsight,
        X_genomaps,
        y_test,
        nump,
    )
