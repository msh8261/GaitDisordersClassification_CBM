import os
# import glob
# import numpy as np
# import math
import shutil
import sys

# import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
# torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn

import config.config_train as config

# from sklearn import metrics



# from torch.autograd import Variable
# import torch.nn.functional as F


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from argparse import ArgumentParser

from pytorch_lightning.callbacks import EarlyStopping

from src.model_deepInsight import GaitModel_DI
from src.model_gnClassify import GaitModel_GNC
from src.model_ViT import GaitModel
from src.model_ViT_GAN import GaitModel1
from src.model_ViT_GAN_genmaps import GaitModel2
from src.model_ViT_GAN_deepInsight import GaitModel3

if config.params["models_name"][0] == "gnClassify":
    device = "cpu"
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_size = config.params["input_size"]
sequence_length = config.params["sequences"]


def train(
    k, random_state, X_train_path, y_train_path, X_test_path, y_test_path, model_name
):
    pl.seed_everything(random_state)

    parser = ArgumentParser()
    args, unknown = parser.parse_known_args()
    parent_parser = pl.Trainer.add_argparse_args(parser)
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

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
    elif model_name == "gan_genomaps":
        model = GaitModel2(
            k,
            random_state,
            X_train_path,
            y_train_path,
            X_test_path,
            y_test_path,
            model_name,
        )
    elif model_name == "gan_deepInsight":
        model = GaitModel3(
            k,
            random_state,
            X_train_path,
            y_train_path,
            X_test_path,
            y_test_path,
            model_name,
        )
    elif model_name == "deepInsight":
        model = GaitModel_DI(
            k,
            random_state,
            X_train_path,
            y_train_path,
            X_test_path,
            y_test_path,
            model_name,
        )
    elif model_name == "gnClassify":
        model = GaitModel_GNC(
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint_" + model_name,
        save_top_k=1,
        verbose=True,
        monitor="average_val_loss",
        mode="min",
    )

    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs", name="model_run_" + model_name
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    if model_name == "gnClassify":
        trainer = pl.Trainer.from_argparse_args(
            args,
            max_epochs=config.params["epochs"],
            deterministic=False,
            gpus=0,
            progress_bar_refresh_rate=1,
            logger=logger,
            callbacks=[
                EarlyStopping(monitor="average_val_loss", patience=50),
                checkpoint_callback,
                lr_monitor,
            ],
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            max_epochs=config.params["epochs"],
            deterministic=False,
            gpus=1,
            progress_bar_refresh_rate=1,
            logger=logger,
            callbacks=[
                EarlyStopping(monitor="average_val_loss", patience=50),
                checkpoint_callback,
                lr_monitor,
            ],
        )

    tuner = pl.tuner.tuning.Tuner(trainer)

    trainer.fit(model)

    trainer.test(model)

    return model, trainer


def save_scripted_module(model, save_model_path):
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, save_model_path)


def save_traced_module(model, save_model_path):
    traced_model = torch.jit.trace(
        model,
        torch.rand(1, sequence_length, input_size, dtype=torch.float32, device="cuda"),
    )
    torch.jit.save(traced_model, save_model_path)


def convert_ckp_to_pt_model(
    best_k,
    best_r,
    path_model,
    X_train_path,
    y_train_path,
    X_test_path,
    y_test_path,
    model_name,
):
    pl.seed_everything(best_r)
    if model_name == "gan":
        model = GaitModel1(
            best_k,
            best_r,
            X_train_path,
            y_train_path,
            X_test_path,
            y_test_path,
            model_name,
        )
    else:
        model = GaitModel(
            best_k,
            best_r,
            X_train_path,
            y_train_path,
            X_test_path,
            y_test_path,
            model_name,
        )
    model.to(device)
    model.eval()
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint["state_dict"])
    save_model_path = f"saved_models/{model_name}.pt"

    save_scripted_module(model, save_model_path)


def save_metrics_to_file(
    best_k, best_r, model_name, model_scores, nums_folds, num_classes
):
    with open(
        f"results/{str(model_name)}_fold{best_k}_random{best_r}_{str(num_classes)}classes_results_check.txt",
        "w",
    ) as f:
        for j, score in enumerate(model_scores):
            f.write("\n")
            for key, value in score:
                f.write(f"{key}: {value.cpu().detach().numpy()}, ")
            f.write(
                "\n======================================================================="
            )
    with open(
        f"results/{str(model_name)}_fold{best_k}_random{best_r}_{str(num_classes)}classes_results.txt",
        "w",
    ) as f:
        for j, score in enumerate(model_scores):
            for key, value in score:
                if j == 0:
                    f.write("%s," % (key))
            f.write("\n")
            for key, value in score:
                f.write("%s," % (value.cpu().detach().numpy()))


def _metrics(trainer, model_name):
    print("===================================================================")
    metrics = trainer.callback_metrics

    print(f'average_train_loss: {metrics["average_train_loss"]:.2f}')
    print(f'average_train_acc: {metrics["average_train_acc"]*100:.2f} %')
    print(f'average_train_f1: {metrics["average_train_f1"]*100:.2f} %')
    print(f'average_val_loss: {metrics["average_val_loss"]:.2f}')
    print(f'average_val_acc: {metrics["average_val_acc"]*100:.2f} %')
    print(f'average_val_f1: {metrics["average_val_f1"]*100:.2f} %')
    print(f'average_test_acc: {metrics["average_test_acc"]*100:.2f} %')
    print(f'average_test_f1: {metrics["average_test_f1"]*100:.2f} %')

    return metrics.items(), metrics["average_test_f1"], metrics["average_test_acc"]


def delete_all_files(dir_path):
    for file_name in os.listdir(dir_path):
        # construct full file path
        file = os.path.join(dir_path, file_name)
        if os.path.isfile(file):
            print("Deleting file:", file)
            os.remove(file)


if __name__ == "__main__":

    train_dataset_path = config.params["train_dataset_path"]
    n_folds = config.params["n_folds"]
    random_state_list = config.params["random_state_list"]
    models_name = config.params["models_name"]
    num_classes = config.params["num_class"]

    X_train_path = os.path.join(train_dataset_path, "Xtrain.File")
    y_train_path = os.path.join(train_dataset_path, "ytrain.File")
    X_test_path = os.path.join(train_dataset_path, "Xtest.File")
    y_test_path = os.path.join(train_dataset_path, "ytest.File")

    train_mode = True

    if train_mode:
        for j, model_name in enumerate(models_name):
            print("==========================================")
            print(f"trian for {model_name} model")
            print("==========================================")
            os.makedirs(f"lightning_logs/model_run_{model_name}", exist_ok=True)
            best_r = 0
            for i, r in enumerate(random_state_list):
                model_scores = []
                best_score = 0
                best_k = 0
                for k in range(n_folds):
                    delete_all_files("./checkpoints")
                    shutil.rmtree(f"./lightning_logs/model_run_{model_name}")
                    model, trainer = train(
                        k,
                        r,
                        X_train_path,
                        y_train_path,
                        X_test_path,
                        y_test_path,
                        model_name,
                    )

                    metrics_items, test_f1, test_acc = _metrics(trainer, model_name)
                    model_scores.append(metrics_items)

                    if best_score < (test_f1 + test_acc) / 2:
                        dir_path = "saved_models/"
                        delete_all_files(dir_path)
                        path_model = os.path.join(
                            dir_path, f"{model_name}_k{k}_rand{r}.pt"
                        )

                        save_scripted_module(model, path_model)
                        best_score = test_f1
                        best_k = k
                        best_r = r

                save_metrics_to_file(
                    best_k, best_r, model_name, model_scores, n_folds, num_classes
                )
