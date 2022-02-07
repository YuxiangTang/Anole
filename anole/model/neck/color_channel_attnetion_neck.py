import torch.nn as nn

from ..builder import NECK


class ColorChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ColorChannelAttention, self).__init__()
        
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=True, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, bias=True, padding=0)
        )
        self.relu = nn.ReLU(inplace=True) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        feat = self.conv(kwargs["color_feature"])
        gamma = self.sigmoid(feat)
        x = x * gamma
        return self.relu(x)
    
    
@NECK.register_obj
def color_channel_att_neck(**kwargs):
    return ColorChannelAttention(**kwargs)
