import numpy as np


def apply_tone_map(x):
    # simple tone curve
    # return 3 * x ** 2 - 2 * x ** 3
    x = x.astype(np.float32)
    x = 3 * np.power(x, 2) - 2 * np.power(x, 3)
    x = np.clip(x, 0., 1.)
    return x
