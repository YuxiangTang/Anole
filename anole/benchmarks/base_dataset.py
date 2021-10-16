"""
Some operations about SE-Scheme and other basic operations.
"""
from abc import abstractmethod
import math
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import random

from collections import OrderedDict

from .utils import augment_ill, augment_img

__all__ = ['BaseDataset']

class BaseDataset(Dataset):
    """
    Provides the base functions that all subclasses depend on.
    """
    def __init__(self, 
                 dataset_name: str,
                 data_dir: str, 
                 input_size: int, 
                 training: bool, 
                 fold_idx: int,
                 aug_num: int = 4,
                 statistic_mode: bool = False,
                 minik: Optional[int] = 1,
                 ):
        """
        :param data_dir: the path that saves the raw images (processed ".npy")
        :param input_size: the size of training images
        :param training: if not, the images will not augment 
        :param label_mode: illuminant label or statistic label
        :param minik: serve for the generation of statistic label
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.input_size = input_size
        self.training = training
        self.aug_num = aug_num
        self.statistic_mode = statistic_mode
        self.minik = minik
        self.img_list = self.three_fold(fold_idx)
        
        self.angle = 60
        self.scale = [0.5, 1.0]
        self.aug_color = 0.8
        
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        # process image_path
        img_path = self.data_dir + self.img_list[idx]
        
        img, ill, camera = self.load_data(img_path)

        # One trick: Set 0 to a very small value
        img = img * 65535.0 # 0 ~ 1 --> 0 ~ 65535
        img[img == 0] = 1e-5
        
        if self.training:
            img_batch = []
            gt_batch = []
            # For a single image, augmenting multiple times can improve training efficiency 
            for _ in range(self.aug_num):
                img_aug = augment_img(img) # self.transform(image=img)['image']
                remove_stat, stat, _ = self.generate_statistic_gt(img_aug, ill) # Convert to SET
                img_aug, gt_aug = augment_ill(remove_stat, stat, self.aug_color, self.angle, self.scale, self.input_size)
                img_aug = img_aug / np.max(img_aug)
                img_batch.append(img_aug)
                gt_batch.append(gt_aug)
            img = np.stack(img_batch)
            gt = np.stack(gt_batch)
            img = np.power(img, (1.0/2.2))
            img = img.transpose(0, 3, 1, 2)
        else:
            remove_stat, gt, si = self.generate_statistic_gt(img, ill)
            # for obtain more estimation
            remove_stat = cv2.resize(remove_stat, (0,0), fx=0.5, fy=0.5)
            img = remove_stat / np.max(remove_stat)
            img = np.power(img, (1.0/2.2))
            img = img.transpose(2, 0, 1)
            
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(gt.copy()).float()
        si = torch.from_numpy(si.copy()).float()
        return OrderedDict(img=img, gt=gt, statis=si)
    
    @abstractmethod
    def load_data(self, img_path):
        pass
    
    def generate_statistic_gt(self, img, ill):
        if not self.statistic_mode:
            return img, ill, np.ones_like(ill)
        remove_stat, stat = self.mapping_into_statis(img)
        # mapping ill to statisic
        true_stat = self.L2_norm(ill * stat)
        return remove_stat, true_stat, stat

    def mapping_into_statis(self, img):
        """
        Core operation that the illuminant into statistic form
        """
        stat = self.statis_norm(img) 
        remove = img * stat * math.sqrt(3)
        return remove, stat
    
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
        img_list = []
        if self.training:
            for i in range(3):
                if i == idx:
                    continue
                img_list += self.load_nameseq(self.data_dir + '/{}_fold{}.txt'.format(self.dataset_name, i))
        else:
            img_list = self.load_nameseq(self.data_dir + '/{}_fold{}.txt'.format(self.dataset_name, idx))
            
        return img_list
    