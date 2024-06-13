import os
import random

import cv2
import numpy as np
from skimage.segmentation import clear_border


def apply_Enhance_image_filter(frame):
    img_res = frame.copy()
    img_res = cv2.detailEnhance(img_res, sigma_s=5, sigma_r=0.15)
    return img_res


def apply_GaussianBlur_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    color = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    return color


def apply_equalhist_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return color


def apply_colormap_filter(frame):
    img_res = frame.copy()
    # colormap_image = cv2.applyColorMap(adjusted_img, cv2.COLORMAP_TWILIGHT_SHIFTED)
    colormap_image = cv2.applyColorMap(np.uint8(img_res), cv2.COLORMAP_JET)
    return colormap_image


def apply_hist_colormap_filter(frame):
    img_res = frame.copy()
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hist_colormap_image = cv2.applyColorMap(np.uint8(color), cv2.COLORMAP_JET)
    return hist_colormap_image


def apply_adjust_gamma_filter(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def apply_complex_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    blur = cv2.GaussianBlur(edges, (9, 9), 0)
    # Otsu's thresholding after Gaussian filtering
    retval, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # clear any foreground pixels touching the border of the image
    threshold = clear_border(threshold)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=threshold)
    return cartoon


def random_colour_masks(image):
    colours = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask
