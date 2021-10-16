"""
Ref https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/squeezenet.py
"""
import torch
import torch.nn as nn

from ..builder import HEAD

__all__ = ['average_pooling_head']

class AveragePoolingHead(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels=3):
        super(AveragePoolingHead, self).__init__()
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(input_channels, hidden_channels, kernel_size=6, stride=1, padding=3),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5),
                  nn.Conv2d(hidden_channels, output_channels, kernel_size=1, stride=1)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return nn.functional.normalize(torch.mean(x,dim=(2,3)), dim=1)
    
@HEAD.register_obj
def average_pooling_head(input_channels, output_channels):
    return AveragePoolingHead(input_channels, output_channels)