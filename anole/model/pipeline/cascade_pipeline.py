import torch.nn as nn

from ..builder import PIPELINE
from .base_pipeline import BasePipeline
from ...raw_processing import white_balance_nonlinear_torch

__all__ = ['cascade_pipeline']


class CascadePipeline(nn.Module):
    def __init__(self, backbone, neck, head, cascade_num=3, **kwargs):
        super(CascadePipeline, self).__init__()
        self.base_learner = []
        self.cascade_num = cascade_num
        for _ in range(cascade_num):
            self.base_learner.append(BasePipeline(backbone=backbone, neck=neck, head=head, **kwargs))

    def forward(self, x, **kwargs):
        estimate = []
        for i, model in enumerate(self.base_learner):
            tmp_estimate = model(x)
            tmp_estimate = nn.functional.normalize(tmp_estimate, dim=1)
            estimate.append(tmp_estimate)
            if i == self.cascade_num - 1:
                break
            x = white_balance_nonlinear_torch(x, tmp_estimate)
        return estimate


@PIPELINE.register_obj
def cascade_pipeline(backbone, neck, head, params, **kwargs):
    return CascadePipeline(backbone=backbone,
                           neck=neck,
                           head=head,
                           cascade_num=params.cascade_num,
                           **kwargs)
