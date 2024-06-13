import os
import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import thop
import torch
from torchvision.models import resnet50

import config.config_train as config
from src.model import GaitModel
from src.model1 import GaitModel1

model = resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = thop.profile(model, inputs=(input,))
print("=========================================")
# nicer result view
macs, params = thop.clever_format([macs, params], "%.3f")
print(f"Params: {params} , Macs: {macs} of resnet50 model.")
print("=========================================")

# models_name = ["vit_mlp", "gan"]
models_name = ["vit_mlp", "gan"]

num_features = 70  #
num_class = 3
sequences = 300
random_state = 21
k = 5

train_dataset_path = config.params["train_dataset_path"]
n_folds = config.params["n_folds"]
random_state_list = config.params["random_state_list"]
num_classes = config.params["num_class"]


X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
y_train_path = os.path.join(train_dataset_path, "ytrain.File")
X_test_path = os.path.join(train_dataset_path, "Xtest.File")
y_test_path = os.path.join(train_dataset_path, "ytest.File")

with open("results/params_macs.txt", "w") as f:
    for model_name in models_name:
        if model_name == "gan":
            model = GaitModel1(
                k,
                random_state,
                X_train_path,
                y_train_path,
                X_test_path,
                y_test_path,
                model_name,
            )
        else:
            model = GaitModel(
                k,
                random_state,
                X_train_path,
                y_train_path,
                X_test_path,
                y_test_path,
                model_name,
            )

        device = "cuda"
        input = torch.randn(1, sequences, num_features)  # .to(device)
        macs0, params0 = thop.profile(model, inputs=(input,))
        flops0 = macs0 * 2

        macs, params = thop.clever_format([macs0, params0], "%.3f")
        flops, params = thop.clever_format([flops0, params0], "%.3f")
        print("=========================================")
        print(f"Params: {params} , MACs: {macs}, FLOPs: {flops} of {model_name} model.")
        print("=========================================")
        f.write(
            f"Params: {params} , MACs: {macs}, FLOPs: {flops} of {model_name} model.\n"
        )
