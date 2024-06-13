"""
link of code:
https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
"""

import numpy as np
from tqdm import tqdm


def jitter(x, sigma=0.03):
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [
                CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                for dim in range(x.shape[2])
            ]
        ).T
        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(
                warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim]
            )(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(
                orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]
            ).T
    return ret


def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1] - target_len, size=(x.shape[0])
    ).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]),
                np.arange(target_len),
                pat[starts[i] : ends[i], dim],
            ).T
    return ret
