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
        model: Module,
        optimizer: Optimizer,
        criterion: Module,
        plugins: List = [],
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            plugins=plugins,
            **kwargs,
        )

    def _unpack_train_minibatch(self):
        img = self.train_batch['img']
        gt = self.train_batch['gt']
        si = self.train_batch['statis']
        img = img.to(self.device).float()
        gt = gt.to(self.device).float()
        self.si = si.to(self.device).float()

        _, _, c, h, w = img.shape

        self.img, self.gt = img.view((-1, c, h, w)), gt.view((-1, 3))

    def _unpack_eval_minibatch(self):
        img = self.eval_batch['img']
        gt = self.eval_batch['gt']
        si = self.eval_batch['statis']
        self.img = img.to(self.device).float()
        self.gt = gt.to(self.device).float()
        self.si = si.to(self.device).float()

    def forward(self):
        return self.model(self.img)

    def criterion(self):
        self.per_loss = self._criterion(self.pred, self.gt)
        return torch.mean(self.per_loss)


@STRATEGY.register_obj
def classic_stragtegy(model, optimizer, **kwargs):
    return ClassicStragtegy(model, optimizer, **kwargs)
