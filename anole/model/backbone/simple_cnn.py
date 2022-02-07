"""
Ref 
"""
import torch.nn as nn

from ..builder import BACKBONE
from anole.utils import load_state_dict

__all__ = ['ThreeLayerCNN', 'three_layer_cnn']


class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super(three_layer_cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2, bias=True),  # 29 * 29
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=True),  # 14 * 14
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=True),  # 13 * 13
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x


@BACKBONE.register_obj
def three_layer_cnn(**kwargs):
    model = ThreeLayerCNN(**kwargs)
    return model
