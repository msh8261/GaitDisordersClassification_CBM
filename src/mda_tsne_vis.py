""" export CUBLAS_WORKSPACE_CONFIG=:4096:8 """

import os

import matplotlib.pyplot as plt
import numpy as np
# import random
# import pickle
import pandas as pd  # Please install pandas and matplotlib before you run this example
import seaborn as sns
import torch
# import torchvision.transforms as transforms
# from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torchvision
from mda import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# import src.augmentation as aug
import config.config_train as config
from nn.pyDeepInsight.image_transformer import ImageTransformer
from nn.pyDeepInsight.utils import Norm2Scaler
from src.load import LoadData

# import matplotlib





# import scipy
# import genomap as gp



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


def show_comp_tsne(model, model_vit, X_img, y_, mode):
    conv_layers = []  # we will save the 49 conv layers in this list
    conv1 = getattr(model.model_d.block_conv1, "0")
    conv_layers.append(conv1)
    conv2 = getattr(model.model_d.block_conv2, "0")
    conv_layers.append(conv2)
    conv3 = getattr(model.model_d.block_conv3, "0")
    conv_layers.append(conv3)

    # summary(model, input_size=(img.shape[2], img.shape[0], img.shape[1]), batch_size=1)
    # print(model)

    last_conv = conv_layers[0:][2]

    n = 25
    imgs = []
    targets = []
    out_attns = []
    out_imgs = []
    proc_imgs = []
    proc_mlps = []
    proc_encs = []
    vit_proc_mlps = []
    vit_proc_encs = []
    lbs = []
    for i in range(len(X_img)):
        img = X_img[i]
        lbs.append(y_[i])
        img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_c, out) = model(img_c)
        generated_image = (
            img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        )
        encoder_features_1 = getattr(model.model_g.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model.model_g.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        mlp = getattr(model.model_g, "mlp")(classifier_head)
        mlp_h = (mlp + classifier_head)[0]  # remove channel
        mlp = (
            nn.Linear(img.shape[0] * img.shape[1], 70)
            .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
            .reshape(70, 1)
            .detach()
            .cpu()
            .numpy()
        )
        encoder_features_2 = encoder_features_2.detach().cpu().numpy()[0][0]
        proc_imgs.append(img.reshape(img.shape[0] * img.shape[1]))
        proc_mlps.append(
            mlp_h.detach().cpu().numpy().reshape(mlp_h.shape[0] * mlp_h.shape[1])
        )
        proc_encs.append(
            encoder_features_2.reshape(
                encoder_features_2.shape[0] * encoder_features_2.shape[1]
            )
        )

        img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_c, out) = model_vit(img_c)
        generated_image = (
            img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        )
        encoder_features_1 = getattr(model_vit.model.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model_vit.model.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        mlp = getattr(model_vit.model, "mlp")(classifier_head)
        mlp_h = (mlp + classifier_head)[0]  # remove channel
        mlp = (
            nn.Linear(img.shape[0] * img.shape[1], 70)
            .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
            .reshape(70, 1)
            .detach()
            .cpu()
            .numpy()
        )
        encoder_features_2 = encoder_features_2.detach().cpu().numpy()[0][0]
        vit_proc_mlps.append(
            mlp_h.detach().cpu().numpy().reshape(mlp_h.shape[0] * mlp_h.shape[1])
        )
        vit_proc_encs.append(
            encoder_features_2.reshape(
                encoder_features_2.shape[0] * encoder_features_2.shape[1]
            )
        )

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_mlps = np.array(proc_mlps)
    X_ens = np.array(proc_encs)
    X_vit_mlps = np.array(vit_proc_mlps)
    X_vit_ens = np.array(vit_proc_encs)

    tsns_inputs = tsne_(X_imgs, y, mode)
    tsns_mlps = tsne_(X_mlps, y, mode)
    tsns_vit_mlps = tsne_(X_vit_mlps, y, mode)

    corr = tsns_mlps.corr("pearson")

    # plot the result
    sns.set_theme(style="ticks", font_scale=1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 12))
    im1 = ax1.scatter(
        tsns_inputs["Dimension 1"],
        tsns_inputs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        c=y,
        cmap=plt.cm.get_cmap("plasma", 3),
    )
    cbar = plt.colorbar(
        im1, ax=ax1, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    cbar.set_ticklabels(["1", "2", "3"])
    ax1.set_title("t-SNE Visualization of test samples", fontweight="bold")
    ax1.set_facecolor("lavender")
    ax1.grid(color="white", linewidth=1)
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors="white")
    # ax1.get_xaxis().set_ticks([])
    # ax1.get_yaxis().set_ticks([])

    im2 = ax2.scatter(
        tsns_mlps["Dimension 1"],
        tsns_mlps["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        c=y,
        cmap=plt.cm.get_cmap("plasma", 3),
    )
    # plt.colorbar(ticks=range(3), label='Classes', boundaries=np.arange(4)-0.5)
    cbar = plt.colorbar(
        im2, ax=ax2, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    cbar.set_ticklabels(["1", "2", "3"])
    ax2.set_title("t-SNE Visualization of proposed method", fontweight="bold")
    ax2.set_facecolor("lavender")
    ax2.grid(color="white", linewidth=1)
    ax2.tick_params(axis="x", colors="white")
    ax2.tick_params(axis="y", colors="white")

    im3 = ax3.scatter(
        tsns_vit_mlps["Dimension 1"],
        tsns_mlps["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        c=y,
        cmap=plt.cm.get_cmap("plasma", 3),
    )
    # plt.colorbar(ticks=range(3), label='Classes', boundaries=np.arange(4)-0.5)
    cbar = plt.colorbar(
        im3, ax=ax3, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    cbar.set_ticklabels(["1", "2", "3"])
    ax3.set_title("t-SNE Visualization of Encoder Transformer", fontweight="bold")
    ax3.set_facecolor("lavender")
    ax3.grid(color="white", linewidth=1)
    ax3.tick_params(axis="x", colors="white")
    ax3.tick_params(axis="y", colors="white")

    # plt.suptitle('TSNE Result')
    plt.savefig(f"tsne_compare_{mode}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"tsne_compare_{mode}.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def tsne_(X, y, mode):
    print("==============")
    print(X.shape)
    # create the model
    tsne = TSNE(
        n_components=2,
        perplexity=3,
        learning_rate="auto",
        init="random",
        random_state=21,
        n_iter=6000,
        n_jobs=-1,
    )
    # apply it to the data
    X_dimensions = tsne.fit_transform(X)
    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_dimensions, columns=["Dimension 1", "Dimension 2"])

    df["class"] = y
    return df


def show_comp_mda(model, model_vit, X_img, y_, mode):
    conv_layers = []  # we will save the 49 conv layers in this list
    conv1 = getattr(model.model_d.block_conv1, "0")
    conv_layers.append(conv1)
    conv2 = getattr(model.model_d.block_conv2, "0")
    conv_layers.append(conv2)
    conv3 = getattr(model.model_d.block_conv3, "0")
    conv_layers.append(conv3)

    # summary(model, input_size=(img.shape[2], img.shape[0], img.shape[1]), batch_size=1)
    # print(model)

    last_conv = conv_layers[0:][2]

    n = 25
    imgs = []
    targets = []
    out_attns = []
    out_imgs = []
    proc_imgs = []
    proc_mlps = []
    proc_encs = []
    vit_proc_mlps = []
    vit_proc_encs = []
    lbs = []
    for i in range(len(X_img)):
        img = X_img[i]
        lbs.append(y_[i])
        img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_c, out) = model(img_c)
        generated_image = (
            img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        )
        encoder_features_1 = getattr(model.model_g.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model.model_g.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        mlp = getattr(model.model_g, "mlp")(classifier_head)
        mlp_h = (mlp + classifier_head)[0]  # remove channel
        mlp = (
            nn.Linear(img.shape[0] * img.shape[1], 70)
            .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
            .reshape(70, 1)
            .detach()
            .cpu()
            .numpy()
        )
        encoder_features_2 = encoder_features_2.detach().cpu().numpy()[0][0]
        proc_imgs.append(img.reshape(img.shape[0] * img.shape[1]))
        proc_mlps.append(
            mlp_h.detach().cpu().numpy().reshape(mlp_h.shape[0] * mlp_h.shape[1])
        )
        proc_encs.append(
            encoder_features_2.reshape(
                encoder_features_2.shape[0] * encoder_features_2.shape[1]
            )
        )

        img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_c, out) = model_vit(img_c)
        generated_image = (
            img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        )
        encoder_features_1 = getattr(model_vit.model.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model_vit.model.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        mlp = getattr(model_vit.model, "mlp")(classifier_head)
        mlp_h = (mlp + classifier_head)[0]  # remove channel
        mlp = (
            nn.Linear(img.shape[0] * img.shape[1], 70)
            .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
            .reshape(70, 1)
            .detach()
            .cpu()
            .numpy()
        )
        encoder_features_2 = encoder_features_2.detach().cpu().numpy()[0][0]
        vit_proc_mlps.append(
            mlp_h.detach().cpu().numpy().reshape(mlp_h.shape[0] * mlp_h.shape[1])
        )
        vit_proc_encs.append(
            encoder_features_2.reshape(
                encoder_features_2.shape[0] * encoder_features_2.shape[1]
            )
        )

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])

    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_mlps = np.array(proc_mlps)
    X_ens = np.array(proc_encs)
    X_vit_mlps = np.array(vit_proc_mlps)
    X_vit_ens = np.array(vit_proc_encs)

    mda_inputs = mda_(X_imgs, y, mode)
    mda_mlps = mda_(X_mlps, y, mode)
    mda_vit_mlps = mda_(X_vit_mlps, y, mode)

    # plot the result
    sns.set_theme(style="ticks", font_scale=1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 12))
    im1 = ax1.scatter(
        mda_inputs["Dimension 1"],
        mda_inputs["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        c=y,
        cmap=plt.cm.get_cmap("plasma", 3),
    )
    cbar = plt.colorbar(
        im1, ax=ax1, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    cbar.set_ticklabels(["1", "2", "3"])
    ax1.set_title("MDA Visualization of test samples", fontweight="bold")
    ax1.set_facecolor("lavender")
    ax1.grid(color="white", linewidth=1)
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors="white")

    im2 = ax2.scatter(
        mda_mlps["Dimension 1"],
        mda_mlps["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        c=y,
        cmap=plt.cm.get_cmap("plasma", 3),
    )
    # plt.colorbar(ticks=range(3), label='Classes', boundaries=np.arange(4)-0.5)
    cbar = plt.colorbar(
        im2, ax=ax2, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    cbar.set_ticklabels(["1", "2", "3"])
    ax2.set_title("MDA Visualization of proposed method", fontweight="bold")
    ax2.set_facecolor("lavender")
    ax2.grid(color="white", linewidth=1)
    ax2.tick_params(axis="x", colors="white")
    ax2.tick_params(axis="y", colors="white")

    im3 = ax3.scatter(
        mda_vit_mlps["Dimension 1"],
        mda_mlps["Dimension 2"],
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        c=y,
        cmap=plt.cm.get_cmap("plasma", 3),
    )
    # plt.colorbar(ticks=range(3), label='Classes', boundaries=np.arange(4)-0.5)
    cbar = plt.colorbar(
        im3, ax=ax3, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    cbar.set_ticklabels(["1", "2", "3"])
    ax3.set_title("MDA Visualization of Encoder Transformer", fontweight="bold")
    ax3.set_facecolor("lavender")
    ax3.grid(color="white", linewidth=1)
    ax3.tick_params(axis="x", colors="white")
    ax3.tick_params(axis="y", colors="white")

    plt.savefig(f"mda_compare_{mode}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"mda_compare_{mode}.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def mda_(X, y, mode):
    n_components = 2
    if mode == "train":
        n_components = 20
    print("==============")
    print(X.shape)
    print(y.shape)
    print(n_components)
    # Number of neighbors in MDA analyses
    neighborNum = 3
    Y_pred = y.reshape(y.shape[0], -1)
    # Compute the outline of the output manifold
    # clusterIdx_pred = discoverManifold(Y_pred, neighborNum)

    # # Reshape the output for PCA
    # n1, h1 = X.shape
    # X = X.reshape(-1, h1)
    y = y.reshape(y.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components, svd_solver="arpack")
    X_pca = pca.fit_transform(X)

    # Use the outline of the output manifold to generate the MDA visualization of the ResNet50 features
    X_dimensions = mda(X_pca, y)
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

    X_train, y_train, X_test, y_test = get_data(train_dataset_path)

    model_proposed = torch.jit.load("./results/final/gan_k2_rand21.pt")
    model_proposed.to(device)
    model_proposed.eval()

    model_vit = torch.jit.load("./results/final/vit_mlp_k1_rand21.pt")
    model_vit.to(device)
    model_vit.eval()

    # show_comp_tsne(model_proposed, model_vit, X_train, y_train, 'train')
    # show_comp_mda(model_proposed, model_vit, X_train, y_train, 'train')

    # show_comp_tsne(model_proposed, model_vit, X_test, y_test, 'test')

    show_comp_mda(model_proposed, model_vit, X_test, y_test, "test")
