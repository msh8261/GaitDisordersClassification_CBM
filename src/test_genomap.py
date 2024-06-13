import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from itertools import cycle

import cv2
import matplotlib
import matplotlib.pyplot as plt
# import glob
import numpy as np
import pandas as pd
import torch
# import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cycler
from sklearn import metrics
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report
# from sklearn.utils import shuffle
from sklearn.metrics import (PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import label_binarize

# import torchmetrics


colors = cycler(
    "color", ["#EE6666", "#3388BB", "#9988DD", "#EECC55", "#88BB44", "#FFBBBB"]
)
plt.rc(
    "axes",
    facecolor="#E6E6E6",
    edgecolor="none",
    titlesize=14,
    axisbelow=True,
    grid=True,
    prop_cycle=colors,
)
plt.rc("grid", color="w", linestyle="solid")
plt.rc("xtick", direction="out", color="gray", labelsize=14)
plt.rc("ytick", direction="out", color="gray", labelsize=14)
plt.rc("patch", edgecolor="#E6E6E6")
plt.rc("lines", linewidth=2, linestyle="-.")
plt.rcParams["figure.figsize"] = (8, 6)
matplotlib.rc("font", size=14)
matplotlib.rc("lines", linewidth=3, linestyle="-.")
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

import genomap as gp
import scipy
from torchvision import transforms

import config.config_train as config
from nn.genomap.genomap import construct_genomap
from nn.pyDeepInsight.image_transformer import ImageTransformer
from nn.pyDeepInsight.utils import Norm2Scaler
from src.dataset import GaitData
from src.load import LoadData
# from pretty_confusion_matrix import pp_matrix
from src.pp_matrix import pp_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cup")

input_size = config.params["input_size"]


def filters(im, mode="sharpen"):
    # remove noise
    im = cv2.GaussianBlur(im, (3, 3), 0)
    if mode == "laplacian":
        # convolute with proper kernels
        im_out = cv2.Laplacian(im, cv2.CV_32F)
    elif mode == "sobelx":
        im_out = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)  # x
    elif mode == "sobely":
        im_out = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=5)  # y
    elif mode == "sharpen":
        # kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        im_out = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
    return im_out


def convert_to_spect(img):
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    return magnitude_spectrum


def plot_precision_recall(
    recall, precision, f_scores, average_precision, save_file_name
):

    _, ax = plt.subplots(figsize=(7, 8))

    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, color="gold")
    _ = display.ax_.set_title("Micro-averaged over all classes")

    for i, color in zip(range(config.params["num_class"]), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i+1}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    fig = display.figure_
    fig.tight_layout()
    fig.savefig(save_file_name, dpi=600)

    plt.show()


def save_accuracy_in_file(labels_gt, labels):
    with open(
        "results/test_results_genomap_cls" + str(config.params["num_class"]) + ".txt",
        "w",
    ) as f:
        acc = accuracy_score(labels_gt, labels)
        print(f"Test accuracy is: {acc.round(2)}")
        f.write(f"Test accuracy is: {acc.round(2)} \n")

        precision, recall, fscore, support = score(labels_gt, labels)

        print("precision: {} ".format(precision.round(2)))
        f.write("precision: {} \n".format(precision.round(2)))
        print("recall: {}".format(recall.round(2)))
        f.write("recall: {} \n".format(recall.round(2)))
        print("fscore: {}".format(fscore.round(2)))
        f.write("fscore: {} \n".format(fscore.round(2)))
        print("support: {}".format(support))
        f.write("support: {} \n".format(support))

        print("ave f1_score: ", f1_score(labels_gt, labels, average="macro").round(2))
        f.write(
            "ave f1_score: {} \n".format(
                f1_score(labels_gt, labels, average="macro").round(2)
            )
        )
        print(
            "ave precision: ",
            precision_score(labels_gt, labels, average="macro").round(2),
        )
        f.write(
            "ave precision: {} \n".format(
                precision_score(labels_gt, labels, average="macro").round(2)
            )
        )
        print(
            "ave recall: {}".format(
                recall_score(labels_gt, labels, average="macro").round(2)
            )
        )
        f.write(
            "ave recall: {} \n".format(
                recall_score(labels_gt, labels, average="macro").round(2)
            )
        )

        conf_mat = confusion_matrix(labels_gt, labels)
        print(conf_mat)

        df_cm = pd.DataFrame(conf_mat, index=range(1, 4), columns=range(1, 4))
        cmap = "tab20b"  # 'gist_yarg' #'gnuplot' #'gist_yarg' #'gnuplot'  #cmap=plt.cm.Blues #'coolwarm_r' #'PuRd'

        pp_matrix(
            df_cm,
            cmap=cmap,
            fz=14,
            lw=0.5,
            figsize=[8, 6],
            save_file_name="results/metrics_confusion_genomap.png",
        )

        precision = dict()
        recall = dict()
        average_precision = dict()

        labels_gt = label_binarize(
            labels_gt, classes=[*range(config.params["num_class"])]
        )
        labels = label_binarize(labels, classes=[*range(config.params["num_class"])])

        for i in range(config.params["num_class"]):
            precision[i], recall[i], _ = precision_recall_curve(
                labels_gt[:, i], labels[:, i]
            )
            average_precision[i] = average_precision_score(
                labels_gt[:, i], labels[:, i]
            )

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            labels_gt.ravel(), labels.ravel()
        )
        average_precision["micro"] = average_precision_score(
            labels_gt, labels, average="micro"
        )

        save_file_name = "results/precision_recall_genomap.png"
        plot_precision_recall(
            recall, precision, fscore, average_precision, save_file_name
        )


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


def get_genomap_data(X_test, pixel_size=(36, 36)):
    colNum = pixel_size[1]  # Column number of genomap
    rowNum = pixel_size[0]  # Row number of genomap
    data = X_test
    # Construction of genomaps
    nump = rowNum * colNum
    if nump < data.shape[1]:
        data, index = select_n_features(data, nump)

    dataNorm = scipy.stats.zscore(data, axis=0, ddof=1)
    genoMaps = construct_genomap(dataNorm, rowNum, colNum, epsilon=0.0, num_iter=200)

    XTest = genoMaps.transpose([0, 3, 1, 2])

    X_test_img = XTest.reshape(XTest.shape[0], XTest.shape[2], XTest.shape[3])

    return X_test_img


def test_saved_model(path_model, model_name, X_test, y_test, pixle_size):
    labels, preds, ims, _ims, specs, sharps = [], [], [], [], [], []
    model_saved = torch.jit.load(path_model)
    model_saved.to(device)
    model_saved.eval()
    print("============ Test Results =============")
    print(f"model path {model_name} is running for test....")
    X_test = get_genomap_data(X_test, pixle_size)
    for ix in range(len(X_test)):
        im = X_test[ix]
        label = y_test[ix]
        if input_size == 36:  # remove x,y features
            im = torch.tensor(im[:, 34:]).numpy()
        elif input_size == 34:  # only x,y features
            im = torch.tensor(im[:, :34]).numpy()
        elif input_size == 58:  # remove dist points from nose
            im = torch.tensor(im[:, :58]).numpy()
        elif input_size == 62:  # remove 8 symetric angles
            x1 = torch.tensor(im[:, :34])
            key_points = [0, 2, 4, 6, 8, 10, 12, 14]
            x2 = torch.tensor(im[:, 34:50])
            x2 = torch.index_select(x2, 1, torch.tensor(key_points))
            x3 = torch.tensor(im[:, 50:])
            im = (torch.cat((x1, x2, x3), dim=1)).numpy()
        # elif input_size == 62: # remove 8 bones angles
        #     x1 = torch.tensor(im[:, :50])
        #     x2 = torch.tensor(im[:, 58:])
        #     im = (torch.cat((x1, x2), dim=1)).numpy()
        # elif input_size == 62: # remove 8 symetric dist
        #     x1 = torch.tensor(im[:, :34])
        #     key_points = [1,3,5,7,9,11,13,15]
        #     x2 = torch.tensor(im[:, 34:50])
        #     x2 = torch.index_select(x2,1,torch.tensor(key_points))
        #     x3 = torch.tensor(im[:, 50:])
        #     im = (torch.cat((x1, x2, x3), dim=1)).numpy()

        data_input = torch.autograd.Variable(torch.tensor(im[None])).to(device)
        (_im, pred) = model_saved(data_input)

        # pred = torch.nn.functional.log_softmax(pred)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().detach().numpy()
        preds.extend(pred)
        labels.extend(label)

    # print(labels)
    # print(preds)
    acc = metrics.accuracy_score(labels, preds)
    print(f"---->>>> acc: {acc} <<<<-------")

    save_accuracy_in_file(labels, preds)


if __name__ == "__main__":
    train_dataset_path = config.params["train_dataset_path"]
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")
    y_test_path = os.path.join(train_dataset_path, "ytest.File")

    ld_ts = LoadData(X_test_path, y_test_path, 0)
    X_test = ld_ts.get_X()
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    y_test = ld_ts.get_y()

    model_name = config.params["models_name"][0]
    dir_path = "saved_models/"
    for file_name in os.listdir(dir_path):
        print("===================================================")
        print(f"Test of {file_name}.")
        path_model = os.path.join(dir_path, file_name)
        test_saved_model(path_model, model_name, X_test, y_test, (config.params["genmap_colNum"], config.params["genmap_rowNum"]))
