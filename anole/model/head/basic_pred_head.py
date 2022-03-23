"""
Ref 
"""
import torch
import torch.nn as nn

from ..builder import HEAD

__all__ = ['basic_pred_head']


class BasicPredHead(nn.Module):
    def __init__(self, input_channels, kernel_size, pred_mode):
        super(BasicPredHead, self).__init__()
        self.pred_mode = pred_mode
        if pred_mode == "illumination":
            output_channels = 3
        elif pred_mode == "matrix":
            output_channels = 9
        else:
            raise Exception("Not proper predict mode.")
        self.fc = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1),
        )

    def forward(self, x, **kwargs):
        x = self.fc(x)
        if self.pred_mode == "illumination":
            x = x.view(-1, 3)
        elif self.pred_mode == "matrix":
            x = x.view(-1, 1, 3, 3)
        return x


@HEAD.register_obj
def basic_pred_head(**kwargs):
    return BasicPredHead(**kwargs)
