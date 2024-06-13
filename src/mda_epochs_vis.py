""" export CUBLAS_WORKSPACE_CONFIG=:4096:8 """

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# import glob
# import random
# import pickle
import numpy as np
import pandas as pd  # Please install pandas and matplotlib before you run this example
import torch
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
from mda import *

import config.config_train as config
from src.load import LoadData

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib





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


def mda_(X, y):
    # Number of neighbors in MDA analyses
    neighborNum = 3
    Y_pred = y.reshape(y.shape[0], -1)
    # Compute the outline of the output manifold
    clusterIdx_pred = discoverManifold(Y_pred, neighborNum)

    # # Reshape the output for PCA
    n1, h1, w1 = X.shape
    X = X.reshape(-1, h1 * w1)
    y = y.reshape(y.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2, svd_solver="arpack")
    X_pca = pca.fit_transform(X)

    # Use the outline of the output manifold to generate the MDA visualization of the ResNet50 features
    X_dimensions = mda(X_pca, y)

    return X_dimensions


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


def get_features_from_model(model, img):
    img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
    encoder_features_1 = getattr(model.blocks_encoder, "0")(
        torch.from_numpy(img[None]).to(device)
    )[None]
    encoder_features_2 = getattr(model.blocks_encoder, "1")(encoder_features_1[0])[None]
    classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
        encoder_features_2
    )[:, 0]
    mlp = getattr(model, "mlp")(classifier_head)
    mlp_h = (mlp + classifier_head)[0]
    return (
        mlp_h.detach().cpu().numpy(),
        encoder_features_1.detach().cpu().numpy(),
        encoder_features_2.detach().cpu().numpy(),
    )


def mda_show_epochs(models, X, labels):
    dfs_mlp = []
    dfs_h1 = []
    dfs_h2 = []
    dfs = [[], [], []]
    for model_path in models:
        model = torch.jit.load(model_path)
        model.to(device)
        model.eval()
        features_mlp = []
        features_head1 = []
        features_head2 = []
        for data in X:
            model_feature, head_1_feature, head_2_feature = get_features_from_model(
                model, data
            )
            features_mlp.append(model_feature)
            features_head1.append(head_1_feature)
            features_head2.append(head_2_feature)

        features_mlp_arr = np.array(features_mlp)
        features_head_1_arr = np.array(features_head1)
        features_head_1_arr = features_head_1_arr.reshape(
            features_head_1_arr.shape[0],
            features_head_1_arr.shape[3],
            features_head_1_arr.shape[4],
        )
        features_head_2_arr = np.array(features_head2)
        features_head_2_arr = features_head_2_arr.reshape(
            features_head_2_arr.shape[0],
            features_head_2_arr.shape[3],
            features_head_2_arr.shape[4],
        )
        labels = np.array(labels)
        mda_features_mlp = mda_(features_mlp_arr, labels)
        print(features_mlp_arr.shape)
        print(features_head_1_arr.shape)
        print(features_head_2_arr.shape)
        mda_features_head_1 = mda_(features_head_1_arr, labels)
        mda_features_head_2 = mda_(features_head_2_arr, labels)
        # create a dataframe from the dataset
        df_mlp = pd.DataFrame(
            data=mda_features_mlp, columns=["Dimension 1", "Dimension 2"]
        )
        df_h1 = pd.DataFrame(
            data=mda_features_head_1, columns=["Dimension 1", "Dimension 2"]
        )
        df_h2 = pd.DataFrame(
            data=mda_features_head_2, columns=["Dimension 1", "Dimension 2"]
        )
        dfs_mlp.append(df_mlp)
        dfs_h1.append(df_h1)
        dfs_h2.append(df_h2)
        dfs = [dfs_mlp, dfs_h1, dfs_h2]

    plot_show(dfs, labels)


def plot_show(dfs, labels):
    n = len(dfs[0])
    fig, axes = plt.subplots(n, 1, figsize=(10, 8))
    outer = gridspec.GridSpec(n, 1, wspace=0.2, hspace=0.2)
    for i in range(n):
        axe = axes[i]
        axe.set_axis_off()
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[i], wspace=0.1, hspace=0.1
        )
        for j in range(3):
            ax = plt.Subplot(fig, inner[j])
            df = dfs[j][i]
            im = ax.scatter(
                df["Dimension 1"],
                df["Dimension 2"],
                marker="o",
                edgecolors="black",
                linewidths=0.5,
                c=labels,
                cmap=plt.cm.get_cmap("plasma", 3),
            )
            ax.set_title(f"Epoch_{i*20}, layer_{j+1}", fontsize=10, fontweight="bold")
            ax.set_facecolor("lavender")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            fig.add_subplot(ax)
            if j == 2:
                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    ticks=range(3),
                    label="Classes",
                    boundaries=np.arange(4) - 0.5,
                )
                cbar.set_ticklabels(["1", "2", "3"])

    plt.tight_layout()
    fig.subplots_adjust(right=0.9)  # adjust location of plot
    # fig.colorbar(im, ax=axes, ticks=range(3), label='Classes', location='right', shrink=0.5)
    plt.savefig(f"mda_epochs_vis.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"mda_epochs_vis.jpg", dpi=600, bbox_inches="tight")
    plt.show()


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


if __name__ == "__main__":
    train_dataset_path = config.params["train_dataset_path"]
    batch_size = 16
    epochs = 120
    num_classes = 3

    X_train, y_train, X_test, y_test = get_data(train_dataset_path)

    dir_ = "./results/feats/models"
    all_models = [f"{dir_}/{file}" for file in os.listdir(dir_)]
    print(all_models)

    mda_show_epochs(all_models, X_test, y_test)
