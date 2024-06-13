import os

import numpy as np
import shap
import torch
# import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax

import config.config_train as config
from src.dataset import GaitData
from src.load import LoadData

# import torchmetrics


device = torch.device("cuda" if torch.cuda.is_available() else "cup")

input_size = config.params["input_size"]


def print_feature_importances_shap_values(shap_values, features):
    """
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    """
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {
        fea: imp for imp, fea in zip(importances_norm, features)
    }
    # Sorts the dictionary
    feature_importances = {
        k: v
        for k, v in sorted(
            feature_importances.items(), key=lambda item: item[1], reverse=True
        )
    }
    feature_importances_norm = {
        k: v
        for k, v in sorted(
            feature_importances_norm.items(), key=lambda item: item[1], reverse=True
        )
    }
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")


train_dataset_path = config.params["train_dataset_path"]
X_test_path = os.path.join(train_dataset_path, "Xtest.File")
y_test_path = os.path.join(train_dataset_path, "ytest.File")

ld_ts = LoadData(X_test_path, y_test_path, 0)
X_test = ld_ts.get_X()
y_test = ld_ts.get_y()
test_dataset = GaitData(X_test, y_test)

path_model = "./results/final/gan_k2_rand21.pt"
model_saved = torch.jit.load(path_model)
model_saved.to(device)
model_saved.eval()
preds = []
for ix in range(len(test_dataset)):
    im, label = test_dataset[ix]
    data_input = torch.autograd.Variable(torch.tensor(im[None])).to(device)
    (_im, pred) = model_saved(data_input)
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().detach().numpy()
    preds.append(pred)

# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
# preds = np.array(preds)
# print(X_test.shape, preds.shape)
# Fits the explainer
explainer = shap.KernelExplainer(preds, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)

# # Evaluate SHAP values
# shap_values = explainer.shap_values(X)

shap.plots.bar(shap_values)

shap.summary_plot(shap_values)

shap.summary_plot(shap_values, plot_type="violin")

shap.plots.bar(shap_values[0])

shap.plots.waterfall(shap_values[0])

# shap.plots.force(shap_test[0])
