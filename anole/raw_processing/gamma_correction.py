import numpy as np

def apply_gamma(img):
    T = 0.0031308
    # rgb1 = torch.max(rgb, rgb.new_tensor(T))
    return np.where(img < T, 12.92 * img, (1.055 * np.power(np.abs(img), 1 / 2.4) - 0.055))

def remove_gamma(img):
    T = 0.04045
    #img1 = np.max(img, T)
    #print(img1.shape)
    return np.where(img < T, img / 12.92, np.power(np.abs(img + 0.055) / 1.055, 2.4))