import numpy as np
import torch


def apply_gamma(img):
    T = 0.0031308
    return np.where(img < T, 12.92 * img, (1.055 * np.power(np.abs(img), 1 / 2.4) - 0.055))


def remove_gamma(img):
    T = 0.04045
    return np.where(img < T, img / 12.92, np.power(np.abs(img + 0.055) / 1.055, 2.4))


def apply_gamma_torch(img):
    T = 0.0031308
    return torch.where(img < T, 12.92 * img, (1.055 * torch.pow(torch.abs(img), 1 / 2.4) - 0.055))


def remove_gamma_torch(img):
    T = 0.04045
    return torch.where(img < T, img / 12.92, torch.pow(torch.abs(img + 0.055) / 1.055, 2.4))
