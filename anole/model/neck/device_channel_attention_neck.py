import torch
import torch.nn as nn

from ..builder import NECK


class DeviceChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DeviceChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=6, stride=1, padding=3)
        self.device_fc1 = nn.Conv2d(12, int(in_planes * out_planes), kernel_size=1, bias=False, padding=0)
        self.device_fc2 = nn.Conv2d(12, int(out_planes * out_planes), kernel_size=1, bias=False, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, **kwargs):
        device_id = kwargs["device_id"]
        feat = self.avg_pool(x)
        bn, c, _, _ = feat.shape
        device_id = torch.unsqueeze(torch.unsqueeze(device_id, -1), -1)
        
        conv_param1 = self.device_fc1(device_id).view((bn, c, -1))
        feat = feat.view(bn, 1, c)
        feat = torch.matmul(feat, conv_param1)
        feat = self.leaky_relu(feat)
        
        bn, _, c2 = feat.shape
        conv_param2 = self.device_fc2(device_id).view((bn, c2, c2))
        feat = torch.matmul(feat, conv_param2) 
        gamma = self.sigmoid(feat.view((bn, c2, 1, 1)))
        
        out = self.maxpool(x)
        out = self.conv1(out)
        out = out * gamma
        return self.relu(out)


@NECK.register_obj
def device_channel_att_neck(**kwargs):
    return DeviceChannelAttention(**kwargs)
