import numpy as np


def apply_tone_map(x):
    # simple tone curve
    # return 3 * x ** 2 - 2 * x ** 3
    x = x.astype(np.float32)
    x = 3 * np.power(x, 2) - 2 * np.power(x, 3)
    x = np.round(x).astype(np.int16)
    x = np.clip(0, 65536)
    return x
