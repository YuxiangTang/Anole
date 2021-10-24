import torch.nn as nn

from ..builder import BACKBONE, NECK, HEAD, PIPELINE
from anole.utils import build_from_cfg

__all__ = ['base_pipeline']


class BasePipeline(nn.Module):
    def __init__(self, backbone, neck, head):
        super(BasePipeline, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[0][:12])
        self.neck = neck
        self.head = head
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


@PIPELINE.register_obj
def base_pipeline(backbone, neck, head, **kwargs):
    _backbone = build_from_cfg(backbone.name, backbone.params, BACKBONE)
    _neck = build_from_cfg(neck.name, neck.params, NECK)
    _head = build_from_cfg(head.name, head.params, HEAD)
    return BasePipeline(backbone=_backbone, neck=_neck, head=_head)
