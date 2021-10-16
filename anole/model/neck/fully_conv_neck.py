"""
Ref https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/squeezenet.py
"""
import torch.nn as nn

from ..builder import NECK

__all__ = ['fully_conv_neck']

class FullyConvNeck(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FullyConvNeck, self).__init__()
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(input_channels, output_channels, kernel_size=6, stride=1, padding=3),
                  nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.fc(x)
    
@NECK.register_obj
def fully_conv_neck(input_channels, output_channels):
    return FullyConvNeck(input_channels, output_channels)