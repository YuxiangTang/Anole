"""
Angular Loss = arccos(p', p) * 180 / pi
"""

import torch
import torch.nn as nn
import math

from ..builder import LOSS

__all__ = ['angular_loss']


class AngularLoss(torch.nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()
        self.threshold = 0.999999

    def forward(self, pred, target):
        return self.loss(pred, target)

    def loss(self, pred, target):
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)

        arccos_num = torch.sum(pred * target, dim=1)
        arccos_num = torch.clamp(arccos_num, -self.threshold, self.threshold)
        angle = torch.acos(arccos_num) * (180 / math.pi)
        return angle


@LOSS.register_obj
def angular_loss(**kwargs):
    return AngularLoss()
