"""
Ref https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/squeezenet.py
"""
import torch
import torch.nn as nn

from ..builder import HEAD

__all__ = ['average_pooling_head']


class AveragePoolingHead(nn.Module):
    def __init__(self, input_channels):
        super(AveragePoolingHead, self).__init__()
        output_channels = 3
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(input_channels, output_channels, kernel_size=1,
                      stride=1),
        )

    def average_pooling(self, x):
        return torch.mean(x, dim=(2, 3))

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.normalize(self.average_pooling(x), dim=1)


@HEAD.register_obj
def average_pooling_head(input_channels, output_channels):
    return AveragePoolingHead(input_channels, output_channels)
