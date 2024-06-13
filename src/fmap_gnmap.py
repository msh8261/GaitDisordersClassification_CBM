import os
# import random
import pickle

import genomap as gp
import genomap.genoNet as gNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Please install pandas and matplotlib before you run this example
import scipy
import seaborn as sns
# import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from genomap.genomap import construct_genomap
from sklearn.decomposition import PCA
# from torchcam.methods import CAM, GradCAM, GradCAMpp, ScoreCAM
# from torchcam.methods import SSCAM, ISCAM, XGradCAM, LayerCAM
# from torchcam.utils import overlay_mask
# from torchvision.transforms.functional import normalize, resize, to_pil_image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torchcam.methods import SmoothGradCAMpp

# import src.augmentation as aug
import config.config_train as config
from src.load import LoadData

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


def findConv2dOutShape(hin, win, conv, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor(
        (hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    wout = np.floor(
        (win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if pool:
        hout /= pool
        wout /= pool
    print("----->>>> ", hout, wout)
    return int(hout), int(wout)


class genoNet(nn.Module):
    # Define the convolutional neural network architecture
    def __init__(self, input_shape, class_num):
        super(genoNet, self).__init__()
        input_dim = input_shape[2]
        Cin, Hin, Win = 1, input_dim, input_dim
        init_f = 8
        num_fc1 = 100
        # Cin = 3
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3, padding=1)

        self.num_flatten = 1089 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, class_num)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.contiguous().view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def show_activation_map(model, X_img, y_test):
    # CAM, GradCAM, GradCAMMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
    cam_extractor = SmoothGradCAMpp(
        model, input_shape=[X_img.shape[3], X_img.shape[1], X_img.shape[2]]
    )
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
    sns.set_theme(style="ticks", font_scale=1)
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
            a.set_ylabel(f"class {targets[indexes[i]]}", fontsize=10)

    plt.tight_layout()
    plt.savefig("gen_CAM.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_genoMap(genoMaps, y_test):
    indexes = sorted(range(len(y_test)), key=lambda k: y_test[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    sns.set_theme(style="ticks", font_scale=1)
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Result of genoMap")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        plt.imshow(genoMaps[i] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {y_test[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
    plt.tight_layout()
    plt.savefig("genMAP.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_all_process(model, X_img, y_test, nump):
    # print(model)
    # CAM, GradCAM, GradCAMMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
    cam_extractor = SmoothGradCAMpp(
        model, input_shape=[X_img.shape[3], X_img.shape[1], X_img.shape[2]]
    )

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
        conv1 = getattr(model, "conv1")
        fc1 = getattr(model, "fc1")
        out_conv = conv1(img4)
        out_fc1 = fc1(out_conv.contiguous().view(-1, nump * 8)).detach().numpy()
        gray_scale = torch.sum(out_conv, 0)
        out_conv = gray_scale / out_conv.shape[0]
        out_conv = out_conv.data.max(dim=0)[1].numpy()

        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        names.append("Input sample")
        names.append("CAMpp")
        names.append("Conv layer")

        processed.append(img.reshape(img.shape[1], img.shape[2]).detach().numpy())
        processed.append(activation_map[0].squeeze(0).numpy())
        processed.append(out_conv)
        # names.append('FC layer')
        # processed.append(out_fc1)
        lbs.append(y)

    sns.set_theme(style="ticks", font_scale=1)
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f"Result of genoNet method")
    num_fig_per_row = int(len(processed) / 3)
    for i in range(len(processed)):
        a = fig.add_subplot(3, num_fig_per_row, i + 1)
        # Visualize the raw CAM
        print(processed[i].T.shape)
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
    plt.savefig("gen_all_process.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("gen_all_process.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def show_tsne(model, X_img, y_, nump):
    n = 25
    imgs = []
    targets = []
    out_attns = []
    out_imgs = []
    proc_imgs = []
    proc_fcs = []
    lbs = []
    for i in range(len(X_img)):
        img = X_img[i]
        lbs.append(y_[i])
        conv1 = getattr(model, "conv1")
        fc1 = getattr(model, "fc1")
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        img = torch.from_numpy(img)

        if img.dim() < 4:
            # add batch with size 1
            img4 = img[None]

        model_children = list(model.children())
        for i in range(len(model_children)):
            model_children[i] = model_children[i].double()

        out = model(img4)
        out_conv = conv1(img4)
        out_fc1 = fc1(out_conv.contiguous().view(-1, nump * 8)).detach().numpy()
        gray_scale = torch.sum(out_conv, 0)
        out_conv = gray_scale / out_conv.shape[0]
        out_conv = out_conv.data.max(dim=0)[1].numpy()
        print("@@@@@@@@@@@@@")
        print(img.shape)
        print(out_fc1.shape)
        proc_imgs.append(
            img.reshape(img.shape[0] * img.shape[1] * img.shape[2]).detach().numpy()
        )
        proc_fcs.append(out_fc1.reshape(out_fc1.shape[0] * out_fc1.shape[1]))

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_fcs = np.array(proc_fcs)

    tsns_inputs = tsne_(X_imgs, y)
    tsns_fcs = tsne_(X_fcs, y)
    # tsns_ens = tsne_(X_ens, y)

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
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    im2 = ax2.scatter(
        tsns_fcs["Dimension 1"],
        tsns_fcs["Dimension 2"],
        marker="o",
        c=y,
        cmap=plt.cm.get_cmap("viridis", 3),
    )
    # plt.colorbar(ticks=range(3), label='Classes', boundaries=np.arange(4)-0.5)
    fig.colorbar(
        im2, ax=ax2, ticks=range(3), label="Classes", boundaries=np.arange(4) - 0.5
    )
    ax2.set_title("t-SNE Visualization of output features", fontweight="bold")
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    plt.suptitle("TSNE Result (gnmap method)")
    plt.savefig(f"tsne_proposed.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"tsne_proposed.jpg", dpi=600, bbox_inches="tight")
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


train_dataset_path = config.params["train_dataset_path"]

X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
y_train_path = os.path.join(train_dataset_path, "ytrain.File")
X_test_path = os.path.join(train_dataset_path, "Xtest.File")
y_test_path = os.path.join(train_dataset_path, "ytest.File")


ld_tr = LoadData(X_train_path, y_train_path, config.params["num_augmentation"], False)
# for test dataset augmentation shoud be set to 0
ld_ts = LoadData(X_test_path, y_test_path, 0)

X_train = ld_tr.get_X()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
y_train = ld_tr.get_y()

X_test = ld_ts.get_X()
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
y_test = ld_ts.get_y()


colNum = 36  # Column number of genomap
rowNum = 36  # Row number of genomap

data = np.concatenate((X_train, X_test), axis=0)
y = pd.DataFrame(np.concatenate((y_train, y_test), axis=0))
data_test = X_test

num = colNum * rowNum
if num < data.shape[1]:
    data, index = select_n_features(data, num)
    data_test, index = select_n_features(data_test, num)

print(data.shape)

# Normalization of the data
dataNorm_test = scipy.stats.zscore(data_test, axis=0, ddof=1)
# Construction of genomaps
genoMaps_test = gp.construct_genomap(
    dataNorm_test, rowNum, colNum, epsilon=0.0, num_iter=200
)


# Normalization of the data
dataNorm = scipy.stats.zscore(data, axis=0, ddof=1)
# Construction of genomaps
genoMaps = gp.construct_genomap(dataNorm, rowNum, colNum, epsilon=0.0, num_iter=200)
print(genoMaps.shape)
findI = genoMaps[10, :, :, :]

# Plot the first genomap
plt.figure(1)
plt.imshow(findI, origin="lower", extent=[0, 10, 0, 10], aspect=1)
plt.title("Genomap")
plt.show()

# show genoMaps of each classes
show_genoMap(genoMaps, y_test)

input_shape = genoMaps
class_num = 3
# Construction of genomaps
nump = rowNum * colNum
if nump < data.shape[1]:
    data, index = select_n_features(data, nump)

dataNorm = scipy.stats.zscore(data, axis=0, ddof=1)
genoMaps = construct_genomap(dataNorm, rowNum, colNum, epsilon=0.0, num_iter=200)

training_labels = y_train.reshape(y_train.shape[0])
# Split the data for training and testing
dataMat_CNNtrain = genoMaps[: training_labels.shape[0]]
dataMat_CNNtest = genoMaps[training_labels.shape[0] :]

groundTruthTrain = training_labels
classNum = len(np.unique(groundTruthTrain))

# Preparation of training and testing data for PyTorch computation
XTrain = dataMat_CNNtrain.transpose([0, 3, 1, 2])
XTest = dataMat_CNNtest.transpose([0, 3, 1, 2])
yTrain = groundTruthTrain

model = gNet.traingenoNet(XTrain, yTrain, maxEPOCH=100, batchSize=16, verbose=True)


with open("geno.pkl", "wb") as f:
    pickle.dump(model, f)

with open("geno.pkl", "rb") as f:
    model = pickle.load(f)

model_saved = torch.jit.load("./results/gnclassify/gnClassify_k2_rand21.pt")
model_saved.to("cpu")
model_saved.eval()

show_all_process(model, genoMaps_test, y_test, nump)

show_tsne(model, genoMaps_test, y_test, nump)
