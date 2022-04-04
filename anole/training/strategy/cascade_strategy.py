from typing import List

import torch
from .base_strategy import BaseStrategy
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..builder import STRATEGY

__all__ = ['cascade_stragtegy']


class CascadeStragtegy(BaseStrategy):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        assert self.pred is list and len(self.pred) > 0
        total_loss = 0
        accumulated_ill = torch.ones_like(self.pred[0], device=self.pred[0].device)
        for single_loss in self.pred:
            accumulated_ill *= single_loss
            per_loss = self._criterion(accumulated_ill, self.gt)
            total_loss = total_loss + torch.mean(per_loss) / len(self.pred)
        return total_loss


@STRATEGY.register_obj
def cascade_stragtegy(**kwargs):
    return CascadeStragtegy(**kwargs)
