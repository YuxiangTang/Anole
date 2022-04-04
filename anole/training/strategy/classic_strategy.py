from typing import List

import torch
from .base_strategy import BaseStrategy
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..builder import STRATEGY

__all__ = ['classic_stragtegy']


class ClassicStragtegy(BaseStrategy):

    def __init__(
        self,
        device_mode: bool = False,
        statis_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device_mode = device_mode
        self.statis_mode = statis_mode

    def _unpack_train_minibatch(self):
        img = self.data_format(self.train_batch['img'])
        gt = self.data_format(self.train_batch['gt'])
        _, _, c, h, w = img.shape
        self.img, self.gt = img.view((-1, c, h, w)), gt.view((-1, 3))

        if self.device_mode:
            self.device_id = self.data_format(self.train_batch['device_id'])
        if self.statis_mode:
            self.si = self.data_format(self.train_batch['statis'])

    def _unpack_eval_minibatch(self):
        self.img = self.data_format(self.eval_batch['img'])
        self.gt = self.data_format(self.eval_batch['gt'])
        if self.device_mode:
            self.device_id = self.data_format(self.eval_batch['device_id'])
        if self.statis_mode:
            self.si = self.data_format(self.eval_batch['statis'])

    def forward(self):
        attachment = {}
        if self.device_mode:
            attachment['device_id'] = self.device_id
        if self.statis_mode:
            attachment['si'] = self.si
        return self.model(self.img, **attachment)

    def criterion(self):
        per_loss = self._criterion(self.pred, self.gt)
        return torch.mean(per_loss)


@STRATEGY.register_obj
def classic_stragtegy(**kwargs):
    return ClassicStragtegy(**kwargs)
