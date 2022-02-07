import numpy as np


def black_level_correction(img, black_level):
    img = img - black_level
    img[img < 0] = 0
    return img


def white_level_correction(img, black_level, white_level, saturation_scale):
    sat = (white_level - black_level) * saturation_scale
    img[img > sat] = sat
    return img


def pre_process(img, black_level, white_level, saturation_scale):
    img = black_level_correction(img, black_level)
    img = white_level_correction(img, black_level, white_level, saturation_scale)
    img = np.clip(img, 0, 65535)
    return img
