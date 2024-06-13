import os
import random

import cv2
import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
import pandas as pd
import scipy
# import genomap as gp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchcam.methods import SmoothGradCAMpp
from torchvision import transforms
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

# import matplotlib
import config.config_train as config
# from nn.AE_vit_mlp import AutoEncoderViTMLP as AEvitmlp
from nn.discriminator import Discriminator_Conv
from nn.genomap.genomap import construct_genomap
from nn.pyDeepInsight.image_transformer import ImageTransformer
from nn.pyDeepInsight.utils import Norm2Scaler
# from nn.discriminator import Discriminator_MLP
# from timm.models import create_model
from src.load import LoadData

# import src.augmentation as aug

# from matplotlib import cycler
# from itertools import cycle





# from torchcam.methods import CAM, GradCAM, GradCAMpp, ScoreCAM
# from torchcam.methods import SSCAM, ISCAM, XGradCAM, LayerCAM




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


def get_attention_map(model, img, get_mask=False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    x = transform(img)
    x.size()

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

    return result


def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax2.set_title("Attention Map Last Layer")
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)


def show_activation_map(model, X_img, Y):
    net = Discriminator_Conv()
    # CAM, GradCAM, GradCAMMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
    cam_extractor = SmoothGradCAMpp(net, input_shape=[X_img.shape[1], X_img.shape[2]])
    n = 25
    am = []
    targets = []
    out_imgs = []
    for ix in range(n):
        img = X_img[ix]
        y = Y[ix]
        # img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        img = torch.autograd.Variable(torch.tensor(img[None])).to(device)

        (img_c, out) = model(img)

        model_children = list(net.children())
        for i in range(len(model_children)):
            model_children[i] = model_children[i].double()

        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        am.append(activation_map[0].squeeze(0).numpy())

        targets.append(y)
        img_c = img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        out_imgs.append(img_c)

    indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    fig = plt.figure(figsize=(3, 6))
    fig.suptitle(f"input images")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        # Visualize the raw CAM
        plt.imshow(X_img[indexes[i]])
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(3, 6))
    fig.suptitle(f"output images of AE_ViT")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        # Visualize the raw CAM
        plt.imshow(out_imgs[i])
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
    plt.tight_layout()
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
    plt.savefig("AE_ViT_CAM.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_conv_featureMap(model, X_img):
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
    out_imgs = []
    for ix in range(n):
        y = y_test[ix]
        img = X_img[ix]
        img_c = torch.from_numpy(img[None]).to(device)
        feature_map = last_conv(conv_layers[0:][1](conv_layers[0:][0](img_c)))
        targets.append(y)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        out_imgs.append(gray_scale.data.detach().cpu().numpy())

    # sns.set_theme(style="ticks", font_scale=1)
    indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    fig = plt.figure(figsize=(10, 6))
    sns.set_theme(style="ticks", font_scale=1)
    fig.suptitle(f"Result of last conv layer of GAN")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        a.imshow(out_imgs[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
        # # Resize the CAM and overlay it
    plt.tight_layout()
    plt.savefig("GAN_Fmap_classes.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    random.seed(8)
    ix = np.random.choice(25)
    img0 = X_img[ix]

    image = torch.from_numpy(img0[None]).to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        print("=============")
        print(layer)
        print(image.shape)
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    # print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        # feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.detach().cpu().numpy())

    processed[:0] = [img.reshape(img.shape[0], img.shape[1], 1)]
    names[:0] = ["Input Image"]
    print(len(processed))
    sns.set_theme(style="ticks", font_scale=1)
    fig = plt.figure(figsize=(4, 2))
    fig.suptitle(
        f"Result of GAN (different conv layers results on an input)", fontsize=8
    )
    for i in range(len(processed)):
        a = fig.add_subplot(1, 4, i + 1)
        plt.imshow(processed[i] * 255, interpolation="nearest", aspect="auto")
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        # a.axis("off")
        if i == 0:
            a.set_title(names[i].split("(")[0], fontsize=6)
        else:
            a.set_title("Conv" + f"_{i}", fontsize=6)
    plt.tight_layout()
    plt.savefig("GAN_FMap_one_input.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_created_images(model, X_img, y_test, mode="Train"):
    n = 25
    imgs = []
    targets = []
    out_imgs = []
    for ix in range(n):
        y = y_test[ix]
        img = X_img[ix]
        img = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_c, out) = model(img)
        img_c = img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        out_imgs.append(img_c)
        targets.append(y)

    sns.set_theme(style="ticks", font_scale=1)
    indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Input images")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        a.imshow(X_img[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
        # # Resize the CAM and overlay it
    plt.tight_layout()
    plt.savefig("GAN_input_images.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Generated images")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        a.imshow(out_imgs[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
        # # Resize the CAM and overlay it
    plt.tight_layout()
    plt.savefig("GAN_output_images.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def show_atten_images(model, X_img, y_test, mode="Train"):
    n = 25
    imgs = []
    targets = []
    out_imgs = []
    for ix in range(n):
        y = y_test[ix]
        img = X_img[ix][None]
        model_attn = my_forward_wrapper()(torch.from_numpy(img))
        attn_map = model_attn.mean(dim=1).detach()  # .squeeze(0)
        out_imgs.append(attn_map)
        targets.append(y)

    sns.set_theme(style="ticks", font_scale=1)
    indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Attention images")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        a.imshow(out_imgs[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
        # # Resize the CAM and overlay it
    plt.tight_layout()
    plt.savefig("GAN_atten_images.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("GAN_atten_images.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def to_tensor(img):
    transform_fn = Compose(
        [
            Resize(249, 3),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform_fn(img)


def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def show_2_img(img1, img2, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(3, 3))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis("off")
    plt.show()


class Attention(nn.Module):
    """Attention mechanism.
    the original code is from link below but its for linux
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

    this code is for windows
    """

    def __init__(self, dim, n_heads=2, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches , 3 * dim)
        # print(qkv.shape)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches , 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches )
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches , n_patches )
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches , n_patches )
        attn = self.attn_drop(attn)
        # print(attn.shape)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches , head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches , n_heads, head_dim)
        # print(weighted_avg.shape)
        # weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches , dim)
        weighted_avg = weighted_avg.reshape(
            weighted_avg.shape[0],
            weighted_avg.shape[1],
            weighted_avg.shape[2] * weighted_avg.shape[3],
        )
        x = self.proj(weighted_avg)  # (n_samples, n_patches , dim)
        x = self.proj_drop(x)  # (n_samples, n_patches , dim)

        return weighted_avg


def my_forward_wrapper():
    def my_forward(x):
        B, N, C = x.shape
        x = Attention(C, n_heads=2, qkv_bias=True)(x)
        return x

    return my_forward


def visualize_attention(model, X_img, y_test, device):

    n = 25
    imgs = []
    targets = []
    out_imgs = []
    for ix in range(n):
        y = y_test[ix]
        img = X_img[ix][None]

        # get attention results with trained model
        attentions = getattr(model.model_g.blocks_encoder, "0").attn

        attentions = attentions(torch.from_numpy(img).to(device))[None]
        print(attentions.shape)

        nh = 2  # attentions.shape[1]  # number of head
        # keep only the output patch attention
        attentions = attentions[0, :, 0, :].reshape(nh, -1)
        print(attentions.shape)

        patch_size = 300

        attentions = attentions.reshape(nh, 7, 5)
        # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0),
                scale_factor=patch_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        attentions = np.mean(attentions, 0)
        out_imgs.append(attentions)
        targets.append(y)

    sns.set_theme(style="ticks", font_scale=1)
    indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Attention images (proposed method)")
    for i in range(len(indexes)):
        a = fig.add_subplot(3, 5, i + 1)
        a.axes.get_xaxis().set_ticks([])
        a.axes.get_yaxis().set_ticks([])
        a.imshow(out_imgs[indexes[i]] * 255, interpolation="nearest", aspect="auto")
        if i == 0 or i == 5 or i == 10:
            a.set_ylabel(
                f"class {targets[indexes[i]]}", rotation=90, weight="bold", fontsize=10
            )
        # # Resize the CAM and overlay it
    plt.tight_layout()
    plt.savefig("GAN_atten_images.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("GAN_atten_images.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def visualize_proposed_all_process(model, X_img, y_test, device, mode="proposed"):
    print(model)
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

    # indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = sorted(range(len(y_test)), key=lambda k: y_test[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    indexes_2 = [indexes[0], indexes[7], indexes[12]]
    names = []
    processed = []
    lbs = []
    for i in indexes_2:
        img = X_img[i]
        lbs.append(y_test[i])
        img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        (img_c, out) = model(img_c)
        generated_image = (
            img_c.reshape(img_c.shape[1], img_c.shape[2]).detach().cpu().numpy()
        )

        if type(img) != torch.Tensor:
            input_tensor = torch.from_numpy(img[None]).to(device)
        else:
            input_tensor = img[None].to(device)
        encoder_features_1 = getattr(model.model_g.blocks_encoder, "0")(input_tensor)[
            None
        ]
        encoder_features_2 = getattr(model.model_g.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        if mode == "genomap":
            classifier_head = (
                nn.LayerNorm(img_c.shape[2], eps=1e-6)
                .double()
                .to(device)(encoder_features_2)[:, 0]
            )
        else:
            classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
                encoder_features_2
            )[:, 0]
        mlp = getattr(model.model_g, "mlp")(classifier_head)
        mlp_h = (mlp + classifier_head)[0]  # remove channel
        print(mlp_h.shape)
        if mode == "genomap":
            mlp = (
                nn.Linear(img.shape[0] * img.shape[1], 70)
                .double()
                .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
                .reshape(70, 1)
                .detach()
                .cpu()
                .numpy()
            )
        else:
            mlp = (
                nn.Linear(img.shape[0] * img.shape[1], 70)
                .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
                .reshape(70, 1)
                .detach()
                .cpu()
                .numpy()
            )
        clf = getattr(model.model_g, "classifier_mlp")(mlp_h)
        clf = (
            F.softmax(clf, dim=0)
            .data.max(dim=0)[1]
            .reshape(3, 1)
            .detach()
            .cpu()
            .numpy()
        )
        encoder_features_1 = encoder_features_1.detach().cpu().numpy()[0][0]
        encoder_features_2 = encoder_features_2.detach().cpu().numpy()[0][0]
        # get attention results with trained model
        attentions_1 = getattr(model.model_g.blocks_encoder, "0").attn
        attentions_2 = getattr(model.model_g.blocks_encoder, "1").attn
        attentions_1 = attentions_1(input_tensor)[None]
        attentions_2 = attentions_2(attentions_1[0])[None]
        nh = 2  # attentions.shape[1]  # number of head
        # keep only the output patch attention
        attentions_1 = attentions_1[0, :, 0, :].reshape(nh, -1)
        attentions_2 = attentions_2[0, :, 0, :].reshape(nh, -1)
        patch_size = 300
        attentions_1 = attentions_1.reshape(nh, 7, 5)
        attentions_2 = attentions_2.reshape(nh, 7, 5)
        # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
        attentions_1 = (
            nn.functional.interpolate(
                attentions_1.unsqueeze(0),
                scale_factor=patch_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        attentions_2 = (
            nn.functional.interpolate(
                attentions_2.unsqueeze(0),
                scale_factor=patch_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        attentions_1 = np.mean(attentions_1, 0)
        attentions_2 = np.mean(attentions_2, 0)
        attn_1_image = attentions_1
        attn_2_image = attentions_2

        outputs = []

        names.append("Input sample")
        names.append("Attention layer1")
        names.append("Attention layer2")
        names.append("Encoder layer1 feature")
        names.append("Encoder layer2 feature")

        names.append("Generated fake sample")
        processed.append(img.reshape(img.shape[0], img.shape[1], 1))
        processed.append(attn_1_image)
        processed.append(attn_2_image)
        processed.append(encoder_features_1)
        processed.append(encoder_features_2)
        # names.append('Classifier MLP')
        # names.append('Classifier')
        # processed.append(clf)
        # processed.append(mlp)
        processed.append(generated_image)

        image = input_tensor
        for i, layer in enumerate(conv_layers[0:]):
            image = layer(image)
            outputs.append(image)
            names.append(f"Conv1_Block{i}")

        for feature_map in outputs:
            # feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.detach().cpu().numpy())

        print(len(processed))

    sns.set_theme(style="ticks", font_scale=1)
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Result of proposed method with {mode} input", fontsize=12)
    num_fig_per_row = int(len(processed) / 3)
    for i in range(len(processed)):
        a = fig.add_subplot(3, num_fig_per_row, i + 1)
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
    plt.savefig(
        f"{mode}_input_with_proposed_all_process.pdf", format="pdf", bbox_inches="tight"
    )
    plt.savefig(
        f"{mode}_input_with_proposed_all_process.jpg", dpi=600, bbox_inches="tight"
    )
    plt.show()


def visualize_vit_all_process(model, X_img, y_test, device):

    n = 25
    imgs = []
    targets = []
    out_attns = []
    out_imgs = []

    # indexes = sorted(range(len(targets)), key=lambda k: targets[k])
    indexes = sorted(range(len(y_test)), key=lambda k: y_test[k])
    indexes = [
        indexes[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24]
    ]
    indexes_2 = [indexes[0], indexes[7], indexes[12]]
    names = []
    processed = []
    lbs = []
    for i in indexes_2:
        img = X_img[i]
        lbs.append(y_test[i])
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
        print(mlp_h.shape)
        # mlp = nn.Linear(img_c.shape[2], img_c.shape[2]).to(device)(mlp_h).detach().cpu().numpy()
        mlp = (
            nn.Linear(img.shape[0] * img.shape[1], 70)
            .to(device)(mlp_h.reshape(img.shape[0] * img.shape[1]))
            .reshape(70, 1)
            .detach()
            .cpu()
            .numpy()
        )
        clf = getattr(model.model_g, "classifier_mlp")(mlp_h)
        clf = (
            F.softmax(clf, dim=0)
            .data.max(dim=0)[1]
            .reshape(3, 1)
            .detach()
            .cpu()
            .numpy()
        )
        encoder_features_1 = encoder_features_1.detach().cpu().numpy()[0][0]
        encoder_features_2 = encoder_features_2.detach().cpu().numpy()[0][0]
        # get attention results with trained model
        attentions_1 = getattr(model.model_g.blocks_encoder, "0").attn
        attentions_2 = getattr(model.model_g.blocks_encoder, "1").attn
        attentions_1 = attentions_1(torch.from_numpy(img[None]).to(device))[None]
        attentions_2 = attentions_2(attentions_1[0])[None]
        nh = 2  # attentions.shape[1]  # number of head
        # keep only the output patch attention
        attentions_1 = attentions_1[0, :, 0, :].reshape(nh, -1)
        attentions_2 = attentions_2[0, :, 0, :].reshape(nh, -1)
        patch_size = 300
        attentions_1 = attentions_1.reshape(nh, 7, 5)
        attentions_2 = attentions_2.reshape(nh, 7, 5)
        # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
        attentions_1 = (
            nn.functional.interpolate(
                attentions_1.unsqueeze(0),
                scale_factor=patch_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        attentions_2 = (
            nn.functional.interpolate(
                attentions_2.unsqueeze(0),
                scale_factor=patch_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        attentions_1 = np.mean(attentions_1, 0)
        attentions_2 = np.mean(attentions_2, 0)
        attn_1_image = attentions_1
        attn_2_image = attentions_2

        outputs = []

        names.append("Input sample")
        names.append("Attention layer1")
        names.append("Attention layer2")
        names.append("Encoder layer1 feature")
        names.append("Encoder layer2 feature")

        names.append("mlp")
        processed.append(img.reshape(img.shape[0], img.shape[1], 1))
        processed.append(attn_1_image)
        processed.append(attn_2_image)
        processed.append(encoder_features_1)
        processed.append(encoder_features_2)
        # names.append('Classifier MLP')
        # names.append('Classifier')
        # processed.append(clf)
        processed.append(mlp_h.detach().cpu().numpy())

        image = torch.from_numpy(img[None]).to(device)

        print(len(processed))

    sns.set_theme(style="ticks", font_scale=1)
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"Result of proposed method", fontsize=12)
    num_fig_per_row = int(len(processed) / 3)
    for i in range(len(processed)):
        a = fig.add_subplot(3, num_fig_per_row, i + 1)
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
    plt.savefig("vit_all_process.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("vit_all_process.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def show_proposed_tsne(model, X_img, y_):
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

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_mlps = np.array(proc_mlps)
    X_ens = np.array(proc_encs)

    tsns_inputs = tsne_(X_imgs, y)
    tsns_mlps = tsne_(X_mlps, y)
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
    # ax1.figure.set_size_inches(6.5, 4.5)
    # ax1.ax.margins(.15)
    # ax1.despine(trim=True)
    im2 = ax2.scatter(
        tsns_mlps["Dimension 1"],
        tsns_mlps["Dimension 2"],
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
    # ax2.figure.set_size_inches(6.5, 4.5)
    # ax2.ax.margins(.15)
    # ax2.despine(trim=True)
    plt.suptitle("TSNE Result (proposed method)")
    plt.savefig(f"tsne_proposed.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"tsne_proposed.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def show_vit_tsne(model, X_img, y_):
    n = 25
    imgs = []
    targets = []
    out_attns = []
    out_imgs = []
    proc_imgs = []
    proc_mlps = []
    proc_encs = []
    lbs = []
    for i in range(len(X_img)):
        img = X_img[i]
        lbs.append(y_[i])
        img_c = torch.autograd.Variable(torch.tensor(img[None])).to(device)
        img_c, _ = model(img_c)
        encoder_features_1 = getattr(model.model.blocks_encoder, "0")(
            torch.from_numpy(img[None]).to(device)
        )[None]
        encoder_features_2 = getattr(model.model.blocks_encoder, "1")(
            encoder_features_1[0]
        )[None]
        classifier_head = nn.LayerNorm(img_c.shape[2], eps=1e-6).to(device)(
            encoder_features_2
        )[:, 0]
        mlp = getattr(model.model, "mlp")(classifier_head)
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

    y = np.array(y_)
    y = y.reshape(y.shape[0] * y.shape[1])
    # initialise the standard scaler
    sc = StandardScaler()
    # create a copy of the original dataset
    X_imgs = np.array(proc_imgs)
    X_mlps = np.array(proc_mlps)
    X_ens = np.array(proc_encs)

    tsns_inputs = tsne_(X_imgs, y)
    tsns_mlps = tsne_(X_mlps, y)
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
        tsns_mlps["Dimension 1"],
        tsns_mlps["Dimension 2"],
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

    plt.suptitle("TSNE Result (vit method)")
    plt.savefig(f"tsne_vit.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"tsne_vit.jpg", dpi=600, bbox_inches="tight")
    plt.show()


def show_comp_tsne(model, model_vit, X_img, y_):
    conv_layers = []  # we will save the 49 conv layers in this list
    conv1 = getattr(model.model_d.block_conv1, "0")
    conv_layers.append(conv1)
    conv2 = getattr(model.model_d.block_conv2, "0")
    conv_layers.append(conv2)
    conv3 = getattr(model.model_d.block_conv3, "0")
    conv_layers.append(conv3)

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

    tsns_inputs = tsne_(X_imgs, y)
    tsns_mlps = tsne_(X_mlps, y)
    tsns_vit_mlps = tsne_(X_vit_mlps, y)
    # tsns_ens = tsne_(X_ens, y)

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
    ax1.set_title("Visualization of test samples", fontweight="bold")
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
    ax2.set_title("Visualization of proposed method", fontweight="bold")
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
    ax3.set_title("Visualization of Encoder Transformer", fontweight="bold")
    ax3.set_facecolor("lavender")
    ax3.grid(color="white", linewidth=1)
    ax3.tick_params(axis="x", colors="white")
    ax3.tick_params(axis="y", colors="white")

    # plt.suptitle('TSNE Result')
    plt.savefig(f"tsne_compare.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"tsne_compare.jpg", dpi=600, bbox_inches="tight")
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
        n_iter=6000,
        n_jobs=-1,
    )
    # apply it to the data
    X_dimensions = tsne.fit_transform(X)
    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_dimensions, columns=["Dimension 1", "Dimension 2"])

    df["class"] = y
    return df


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

    preprocess = transforms.Compose([transforms.ToTensor()])

    X_test_img = torch.stack([preprocess(img) for img in X_test_img]).float()
    X_train_img = torch.stack([preprocess(img) for img in X_train_img]).float()

    X_test_img = X_test_img[:, 0, :, :]

    return X_train_img, X_test_img, le


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


def get_genomap_data(X_train, X_test, y_train, y_test, pixel_size=(36, 36)):
    colNum = pixel_size[1]  # Column number of genomap
    rowNum = pixel_size[0]  # Row number of genomap
    # Construction of genomaps
    nump = rowNum * colNum
    data = np.concatenate((X_train, X_test), axis=0)
    if nump < data.shape[1]:
        data, index = select_n_features(data, nump)

    dataNorm = scipy.stats.zscore(data, axis=0, ddof=1)
    genoMaps = construct_genomap(dataNorm, rowNum, colNum, epsilon=0.0, num_iter=200)

    # Split the data for training and testing
    dataMat_CNNtrain = genoMaps[: y_train.shape[0]]
    dataMat_CNNtest = genoMaps[y_train.shape[0] :]

    # Preparation of training and testing data for PyTorch computation
    XTrain = dataMat_CNNtrain.transpose([0, 3, 1, 2])
    XTest = dataMat_CNNtest.transpose([0, 3, 1, 2])

    # preprocess = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # X_test_img = torch.stack([preprocess(img) for img in dataMat_CNNtest]).float()

    X_test_img = XTest.reshape(XTest.shape[0], XTest.shape[2], XTest.shape[3])

    return XTrain, X_test_img


train_dataset_path = config.params["train_dataset_path"]

X_train, y_train, X_test, y_test = get_data(train_dataset_path)

model_saved = torch.jit.load("./results/final/gan_k2_rand21.pt")
model_saved.to(device)
model_saved.eval()

model_vit_saved = torch.jit.load("./results/final/vit_mlp_k1_rand21.pt")
model_vit_saved.to(device)
model_vit_saved.eval()

# show_created_images(model_saved, X_test, y_test, mode='Test')

# # # show_activation_map(model_saved, X_test, y_test)

# net = Discriminator_Conv()
# show_conv_featureMap(net, X_test)

# show_conv_featureMap(model_saved, X_test)

# visualize_attention(model_saved, X_test, y_test, device)
# # show_atten_images(model_saved, X_test, y_test, mode='Test')

# visualize_proposed_all_process(model_saved, X_test, y_test, device)

# show_proposed_tsne(model_saved, X_test, y_test)

# visualize_vit_all_process(model_saved, X_test, y_test, device)

# show_vit_tsne(model_vit_saved, X_test, y_test)

show_comp_tsne(model_saved, model_vit_saved, X_test, y_test)


X_train_ = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test_ = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# model_deepInsight_saved = torch.jit.load('./results/deepInsight/gan_7070/gan_k1_rand21.pt')
# model_deepInsight_saved.to(device)
# model_deepInsight_saved.eval()
# X_train_deepInsight, X_test_deepInsight, labelencoder = transform_data(X_train_, y_train, X_test_,
#                                                                     y_test, 16,
#                                                                     (70,70)
#                                                                 )

# visualize_proposed_all_process(model_deepInsight_saved, X_test_deepInsight, y_test, device, 'deepInsight')


# model_genomap_saved = torch.jit.load('./results/gnmap/70_70/gan_k2_rand21.pt')
# model_genomap_saved.to(device)
# model_genomap_saved.eval()
# X_train_genomap, X_test_genomap = get_genomap_data(X_train_, X_test_, y_train, y_test, (70,70))

# visualize_proposed_all_process(model_genomap_saved, X_test_genomap, y_test, device, 'genomap')
