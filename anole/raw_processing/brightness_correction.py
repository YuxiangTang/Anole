import numpy as np


def brightness_correction(img):
    gray = np.sum(img * np.array([0.299, 0.587, 0.114, ]), 2)
    gray_scale = 0.25 / (np.mean(gray) + 1e-15)
    bright = img * gray_scale
    return bright
