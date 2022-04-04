"""
Basic class for loading dataset (Pytorch)

Reference:
[1] Tang, Yuxiang, et al. "Transfer Learning for Color Constancy via Statistic Perspective." (2022).
[2] Hu, Yuanming, Baoyuan Wang, and Stephen Lin. "Fc4: Fully convolutional color constancy with
confidence-weighted pooling." Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition. 2017.
"""
import math
from typing import Optional, List
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .utils import augment_ill, augment_img

__all__ = ['BaseDataset', 'generate_dataset']


class BaseDataset(Dataset):
    """
    Provides the base functions that all subclasses depend on
    and denotes the abstract methods that all subclasses need
    to implement.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        input_size: int,
        training: bool,
        fold_idx: int,
        aug_num: int = 4,
        statistic_mode: bool = False,
        minik: Optional[int] = 1,
        defalut_list: List[str] = [],
    ):
        """
        :param dataset_name: eg. CCD, NUS, Cube+.
        :param data_dir: the father path of the dataset.
            the data_dir + dataset_name is the full path
            that saves the raw images (processed ".npy").
        :param input_size: the size of training images.
        :param training: if not, the images will not be augmented.
        :param statistic_mode: illuminant label or statistic label.
        :param minik: serve for the generation of statistic label.
        :param defalut_list: build dataloader from list.
        """
        self._dataset_name = dataset_name
        self.data_dir = data_dir
        self.input_size = input_size
        self.training = training
        self.aug_num = aug_num
        self.statistic_mode = statistic_mode
        self.minik = minik
        if defalut_list == []:
            self.img_list = self.three_fold(fold_idx)
        else:
            self.img_list = defalut_list

        self.angle = 60
        self.scale = [0.5, 1.0]
        self.aug_color = 0.8

        # the camera types are contained the CCD/NUS8/Cube+ dataset.
        self.camera_type = [
            'Canon5D', 'Canon1D', 'Canon550D', 'Canon1DsMkIII', 'Canon600D', 'FujifilmXM1',
            'NikonD5200', 'OlympusEPL6', 'PanasonicGX1', 'SamsungNX2000', 'SonyA57', 'sRGB'
        ]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # process image_path
        img_path = self.data_dir + self.img_list[idx]

        # load image/ illumination/ camera type
        img, ill, camera = self.load_data(img_path)
        # convert camera type from string to the one-hot vector
        device_id = self.camera2onehot(camera)

        # Trick One: Set 0 to a very small value
        img = img * 65535.0  # 0 ~ 1 --> 0 ~ 65535
        img[img == 0] = 1e-5

        if self.training:
            img_batch = []
            gt_batch = []
            # Trick Two: do augment multiple times per image can improve training efficiency
            for _ in range(self.aug_num):
                img_aug = augment_img(img, self.angle, self.scale, self.input_size)
                # Convert from illumination form to statis form, details in [1]
                if not self.statistic_mode:
                    remove_stat, stat, si = self.generate_statistic(img_aug, ill)
                else:
                    remove_stat, stat, si = img_aug, ill, np.ones_like(ill)
                # do label augment
                img_aug, gt_aug = augment_ill(remove_stat, stat, self.aug_color)
                img_aug = img_aug / np.max(img_aug)
                img_batch.append(img_aug)
                gt_batch.append(gt_aug)
            img = np.stack(img_batch)
            gt = np.stack(gt_batch)
            img = np.power(img, (1.0 / 2.2))
            img = img.transpose(0, 3, 1, 2)
        else:
            remove_stat, gt, si = self.generate_statistic(img, ill)
            # Trick Three: for obtain more point estimation: reference in [2]
            remove_stat = cv2.resize(remove_stat, (0, 0), fx=0.5, fy=0.5)
            img = remove_stat / np.max(remove_stat)
            img = np.power(img, (1.0 / 2.2))
            img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(gt.copy()).float()
        si = torch.from_numpy(si.copy()).float()
        return OrderedDict(img=img, gt=gt, statis=si, device_id=device_id)

    @property
    def dataset_name(self):
        return self._dataset_name

    @abstractmethod
    def load_data(self, img_path):
        """
        Each dataset requires its own special read method.
        """
        pass

    def generate_statistic(self, img, ill):
        """
        Core operation that the illuminant into statistic form
        """
        stat_i = self.statis_norm(img)
        # mapping from ill to statisic
        stat_img = img * stat_i * math.sqrt(3)
        stat_label = self.L2_norm(ill * stat_i)
        return stat_img, stat_label, stat_i

    def statis_norm(self, img):
        """
        Core operation that calculate the h(I, n, sigma, p)
        """
        # compute h(~)
        # img = self.gradient(img)
        val = np.mean(np.power(img, self.minik), (0, 1))
        stat = np.power(val, 1 / self.minik)
        stat = self.L2_norm(stat)

        # compute h(~)^-1
        stat = 1 / (stat + 1e-20)
        stat = self.L2_norm(stat)
        return stat

    def gradient(self, img):
        """
        TODO: Give statistic label more choice
        """
        img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)
        img_grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
        img_grad = np.sqrt(np.power(img_grad_x, 2) + np.power(img_grad_y, 2))
        return img_grad

    def camera2onehot(self, camera):
        return torch.tensor([[int(c == camera) for c in self.camera_type] for _ in range(len(self.camera_type))]).float()

    def resize(self, img):
        return cv2.resize(img, (self.input_size, self.input_size))

    def L2_norm(self, vec):
        return vec / (np.linalg.norm(vec, 2) + 1e-20)

    @staticmethod
    def load_nameseq(dir_path):
        """
        load image name from txt.
        """
        img_list = []
        with open(dir_path, "r") as f:
            for line in f:
                line = line.rstrip()
                img_list.append(line)
        return img_list

    def three_fold(self, idx):
        """
        two folds for test and one fold for training.
        """
        img_list = []
        if self.training:
            for i in range(3):
                if i == idx:
                    continue
                img_list += self.load_nameseq(f'{self.data_dir}/{self.dataset_name}_fold{i}.txt')
        else:
            img_list = self.load_nameseq(f'{self.data_dir}/{self.dataset_name}_fold{idx}.txt')

        return img_list


def generate_dataset(dataset_type, **kwargs):
    return dataset_type(training=True, **kwargs), dataset_type(training=False, **kwargs)
