"""
Ref https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/squeezenet.py
"""
import torch
import torch.nn as nn

from ..builder import HEAD

__all__ = ['confidence_pooling_head']


class ConfidencePoolingHead(nn.Module):
    def __init__(self, input_channels):
        super(ConfidencePoolingHead, self).__init__()
        output_channels = 4
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(input_channels, output_channels, kernel_size=1,
                      stride=1),
        )

    def confidence_pooling(self, x):
        rgb = nn.functional.normalize(x[:, :3, :, :], dim=1)
        confidence = x[:, 3:4, :, :]
        return torch.sum(rgb * confidence, dim=(2, 3))

    def forward(self, x, **kwargs):
        x = self.fc(x)
        return nn.functional.normalize(self.confidence_pooling(x), dim=1)


@HEAD.register_obj
def confidence_pooling_head(**kwargs):
    return ConfidencePoolingHead(**kwargs)
