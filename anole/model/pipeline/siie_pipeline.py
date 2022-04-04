import torch.nn as nn
import torch

from anole.model.builder import BACKBONE, NECK, HEAD, PIPELINE
from anole.utils import build_from_cfg
from anole.model.component.rgb2uvHist import RGB2UVHist
from anole.raw_processing.color_correction import mat_inverse_torch, mat_norm_torch, color_correction_torch

__all__ = ['siie_pipeline']


class SIIEPipeline(nn.Module):
    def __init__(self, backbone, neck, head, **kwargs):
        super(SIIEPipeline, self).__init__()

        self.rgb2uvhist = RGB2UVHist()
        self.mapping_backbone = build_from_cfg(backbone.name, backbone.params,
                                               BACKBONE)
        self.mapping_neck = build_from_cfg(neck.name, neck.params, NECK)
        head.params["pred_mode"] = "matrix"
        self.mapping_head = build_from_cfg(head.name, head.params, HEAD)

        self.estimation_backbone = build_from_cfg(backbone.name,
                                                  backbone.params, BACKBONE)
        self.estimation_neck = build_from_cfg(neck.name, neck.params, NECK)
        head.params["pred_mode"] = "illumination"
        self.estimation_head = build_from_cfg(head.name, head.params, HEAD)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def ill_convert(self, ill, ccm):
        ccm_inv = mat_inverse_torch(ccm)
        return torch.matmul(
            ill.unsqueeze(dim=1).unsqueeze(dim=1),
            ccm_inv.transpose(3, 2)
        ).squeeze(dim=1).squeeze(dim=1)

    def forward(self, x, **kwargs):
        uv_hist1 = self.rgb2uvhist(x)
        feature = self.mapping_backbone(uv_hist1)
        feature = self.mapping_neck(feature, **kwargs)
        mat = self.mapping_head(feature, **kwargs)
        mat = mat_norm_torch(mat)
        x_c = color_correction_torch(x, mat)
        uv_hist2 = self.rgb2uvhist(x_c)
        feature = self.estimation_backbone(uv_hist2)
        feature = self.estimation_neck(feature, **kwargs)
        ill = self.estimation_head(feature, **kwargs)
        ill = self.ill_convert(ill, mat)
        return ill


@PIPELINE.register_obj
def siie_pipeline(backbone, neck, head, **kwargs):
    return SIIEPipeline(backbone=backbone, neck=neck, head=head, **kwargs)
