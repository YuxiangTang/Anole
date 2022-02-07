import numpy as np
import torch


def color_correction(img, ccm):
    return img.dot(ccm.T)


def color_correction_torch(img, ccm):
    img = img.permute(0, 2, 3, 1)
    img = img.matmul(ccm.transpose(3, 2))
    img = img.permute(0, 3, 1, 2)
    return img


def camera2sRGB(img, Camera):
    camera2XYZ = get_XYZ2camera_mat(Camera, False)
    XYZ2lin = get_lin2XYZ_mat(False)
    img_xyz = img.dot(camera2XYZ.T)
    img_lin = img_xyz.dot(XYZ2lin.T)
    img_lin = np.clip(img_lin, 0., 1.)
    return img_lin


def sRGB2camera(img, Camera):
    if Camera in ['sRGB', None]:
        return img
    lin2XYZ = get_lin2XYZ_mat(True)
    XYZ2camera = get_XYZ2camera_mat(Camera, True)
    img_xyz = img.dot(lin2XYZ.T)
    img_cam = img_xyz.dot(XYZ2camera.T)
    img_cam = np.clip(img_cam, 0., 1.)
    return img_cam


# some defalut camera matrices
def get_XYZ2camera_mat(camera_model, ToCamera=True):
    # extracted from dcraw.c
    matrices = {'Canon5D':      (6347,-479,-972,-8297,15954,2480,-1968,2131,7649),  # Canon 5D
                'Canon1D':      (4374,3631,-1743,-7520,15212,2472,-2892,3632,8161), # Canon 1Ds
                'Canon550D':    (6941,-1164,-857,-3825,11597,2534,-416,1540,6039),  # Canon 550D
                'Canon1DsMkIII':(5859,-211,-930,-8255,16017,2353,-1732,1887,7448),  # Canon 1Ds Mark III
                'Canon600D':    (6461,-907,-882,-4300,12184,2378,-819,1944,5931),   # Canon 600D
                'FujifilmXM1':  (10413,-3996,-993,-3721,11640,2361,-733,1540,6011), # FujifilmXM1
                'NikonD5200':   (8322,-3112,-1047,-6367,14342,2179,-988,1638,6394), # Nikon D5200
                'OlympusEPL6':  (8380,-2630,-639,-2887,10725,2496,-627,1427,5438),  # Olympus E-PL6
                'PanasonicGX1': (6763,-1919,-863,-3868,11515,2684,-1216,2387,5879), # Panasonic GX1
                'SamsungNX2000':(7557,-2522,-739,-4679,12949,1894,-840,1777,5311),  # SamsungNX2000
                'SonyA57':      (5991,-1456,-455,-4764,12135,2980,-707,1425,6701)}  # Sony SLT-A57
    xyz2cam = np.asarray(matrices[camera_model]) / 10000
    xyz2cam = xyz2cam.reshape(3, 3)
    xyz2cam = mat_norm(xyz2cam)
    if ToCamera:
        return xyz2cam
    else:
        return np.linalg.inv(xyz2cam)


def get_lin2XYZ_mat(ToXYZ=True):
    linsRGB2XYZ = np.array(((0.4124564, 0.3575761, 0.1804375),
                            (0.2126729, 0.7151522, 0.0721750),
                            (0.0193339, 0.1191920, 0.9503041)))
    linsRGB2XYZ = mat_norm(linsRGB2XYZ)
    if ToXYZ:
        return linsRGB2XYZ
    else:
        return np.linalg.inv(linsRGB2XYZ)
    

def mat_norm(mat):
    mat = mat / np.sum(mat, axis=1, keepdims=True)
    return mat


def mat_norm_torch(mat):
    eps = torch.tensor(1e-9)
    mat_norm = mat / (torch.sum(mat, 3, keepdim=True) + eps) + eps
    return mat_norm


def mat_inverse_torch(mat):
    ccm_inv = torch.linalg.inv(mat)
    return ccm_inv
