import numpy as np
from .gamma_correction import remove_gamma, apply_gamma, remove_gamma_torch, apply_gamma_torch


def white_balance(img, awb_gain):
    img = img / (awb_gain + 1e-8)
    return img


def white_balance_nonlinear(img, awb_gain):
    img = remove_gamma(img)
    img = img / (awb_gain + 1e-8)
    img = apply_gamma(img)
    return img


def white_balance_nonlinear_torch(img, awb_gain):
    img = remove_gamma_torch(img)
    img = img / (awb_gain + 1e-8)
    img = apply_gamma_torch(img)
    return img
