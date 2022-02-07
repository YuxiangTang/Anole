'''
Ref https://github.com/AishuaiYao/PyTorch/blob/master/MobileNet/mobileNet.py
@Author  :   {AishuaiYao}
@License :   (C) Copyright 2013-2017, {None}
@Contact :   {aishuaiyao@163.com}
'''

import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONE
from anole.utils import load_state_dict

__all__ = [
    'MobileNetV1',
    'mobilenetv1',
    'MobileNetV2',
    'mobilenetv2',
    'MobileNetV3',
    'mobilenetv3_large',
    'mobilenetv3_small',
]

model_urls = {
    'mobilenetv1':
    '',
    'mobilenetv2':
    'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1',
    'mobilenetv3_large':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-1cd25616.pth',
    'mobilenetv3_small':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-55df8e1f.pth'
}


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=inchannel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=inchannel), nn.BatchNorm2d(inchannel),
            nn.ReLU6(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=outchannel,
                      kernel_size=1,
                      stride=1), nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'MobileNetV1'
        self.cfg = [
            64, (2, 128), 128, (2, 256), 256, (2, 512), 512, 512, 512, 512,
            512, (2, 1024), (2, 1024)
        ]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layers = self.make_layer(32)
        self.pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, 5)

    def make_layer(self, inchannel):
        layers = []
        for param in self.cfg:
            stride = 1 if isinstance(param, int) else param[0]
            outchannel = param if isinstance(param, int) else param[1]
            layers.append(Block(inchannel, outchannel, stride))
            inchannel = outchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class InvertResidual(nn.Module):
    def __init__(self, inchannel, outchannel, expansion_scale, stride):
        nn.Module.__init__(self)
        self.expand_channel = expansion_scale * inchannel
        self.stride = stride
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=self.expand_channel,
                      kernel_size=1,
                      stride=1), nn.BatchNorm2d(self.expand_channel),
            nn.ReLU6(inplace=True))
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=self.expand_channel,
                      out_channels=self.expand_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=self.expand_channel),
            nn.BatchNorm2d(self.expand_channel), nn.ReLU6(inplace=True))
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=self.expand_channel,
                      out_channels=outchannel,
                      kernel_size=1,
                      stride=1), nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=outchannel,
                      kernel_size=1), nn.BatchNorm2d(outchannel))

    def forward(self, x):
        out = self.expansion(x)
        out = self.depthwise(out)
        out = self.projection(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'MobileNetV2'
        # t, c, n, s [expansion_scale,out_channel,repeated times,stride]
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.bottlenecks = self.make_layer(32)
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=320,
                      out_channels=1280,
                      kernel_size=1,
                      stride=1), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True),
            nn.AvgPool2d(kernel_size=7),
            nn.Conv2d(in_channels=1280, out_channels=5, kernel_size=1))

    def make_layer(self, inchannel):
        bottlenecks = []
        for t, c, n, s in self.cfgs:
            for i in range(n):
                bottlenecks.append(
                    InvertResidual(inchannel=inchannel,
                                   outchannel=c,
                                   expansion_scale=t,
                                   stride=s))
                inchannel = c
        return nn.Sequential(*bottlenecks)

    def forward(self, x):
        out = self.head(x)
        out = self.bottlenecks(out)
        out = self.tail(out)
        return out.view(x.size(0), -1)


def _make_divisible(v, divisor=8, min_value=None):
    """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, self.inplace) / 6
        return out


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * (F.relu6(x + 3, self.inplace) / 6)
        return out


class SEBlock(nn.Module):
    def __init__(self, expand_size, divide=4):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(expand_size,
                      expand_size // divide), nn.ReLU(inplace=True),
            nn.Linear(expand_size // divide, expand_size), HardSigmoid())

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class SEInvertResidual(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, expand_size, SE, NL,
                 stride):
        super(SEInvertResidual, self).__init__()
        self.SE = SE
        self.NL = NL
        self.connect = stride == 1 and inchannel == outchannel
        padding = (kernel_size - 1) // 2
        if self.NL == 'RE':
            activation = nn.ReLU
        else:
            activation = HardSigmoid

        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=expand_size,
                      kernel_size=1,
                      stride=1), nn.BatchNorm2d(expand_size),
            activation(inplace=True))

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=expand_size,
                      out_channels=expand_size,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=expand_size),
            nn.BatchNorm2d(expand_size),
        )
        if SE:
            self.se = SEBlock(expand_size)

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=expand_size,
                      out_channels=outchannel,
                      kernel_size=1,
                      stride=1), nn.BatchNorm2d(outchannel),
            activation(inplace=True))

    def forward(self, x):
        out = self.expansion(x)
        out = self.depthwise(out)
        if self.SE:
            out = self.se(out)
        out = self.pointwise(out)
        if self.connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, net_mode='large', multiplier=1.0):
        super(MobileNetV3, self).__init__()
        self.model_name = 'MobileNetV3'
        if net_mode == 'large':
            self.cfgs = [
                # [in, out,k, s, act,  SE,   exp]
                [16, 16, 3, 1, "RE", False, 16],
                [16, 24, 3, 2, "RE", False, 64],
                [24, 24, 3, 1, "RE", False, 72],
                [24, 40, 5, 2, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120],
                [40, 40, 5, 1, "RE", True, 120],
                [40, 80, 3, 2, "HS", False, 240],
                [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 112, 3, 1, "HS", True, 480],
                [112, 112, 3, 1, "HS", True, 672],
                [112, 160, 5, 2, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960]
            ]
            self.head = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=16,
                          kernel_size=3,
                          stride=2,
                          padding=1), nn.BatchNorm2d(16),
                HardSwish(inplace=True))
            self.bottlenecks = self.make_layer()
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels=160,
                          out_channels=960,
                          kernel_size=1,
                          stride=1), nn.BatchNorm2d(960),
                HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7),
                HardSwish(inplace=True))
            self.conv_nbn = nn.Sequential(
                nn.Conv2d(in_channels=960,
                          out_channels=1280,
                          kernel_size=1,
                          stride=1), HardSwish(inplace=True),
                nn.Conv2d(in_channels=1280, out_channels=5, kernel_size=1))
        else:
            self.cfgs = [[16, 16, 3, 2, "RE", True, 16],
                         [16, 24, 3, 2, "RE", False, 72],
                         [24, 24, 3, 1, "RE", False, 88],
                         [24, 40, 5, 2, "RE", True, 96],
                         [40, 40, 5, 1, "RE", True, 240],
                         [40, 40, 5, 1, "RE", True, 240],
                         [40, 48, 5, 1, "HS", True, 120],
                         [48, 48, 5, 1, "HS", True, 144],
                         [48, 96, 5, 2, "HS", True, 288],
                         [96, 96, 5, 1, "HS", True, 576],
                         [96, 96, 5, 1, "HS", True, 576]]
            self.head = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=16,
                          kernel_size=3,
                          stride=2,
                          padding=1), nn.BatchNorm2d(16),
                HardSwish(inplace=True))
            self.bottlenecks = self.make_layer()
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels=96,
                          out_channels=576,
                          kernel_size=1,
                          stride=1), nn.BatchNorm2d(576), SEBlock(576),
                HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7))
            self.conv_nbn = nn.Sequential(
                nn.Conv2d(in_channels=576,
                          out_channels=1280,
                          kernel_size=1,
                          stride=1), HardSwish(inplace=True),
                nn.Conv2d(in_channels=1280,
                          out_channels=5,
                          kernel_size=1,
                          stride=1))

    def make_layer(self):
        bottlenecks = []
        for inchannel, outchannel, kernel_size, stride, activation, se, expand_size in self.cfgs:
            bottlenecks.append(
                SEInvertResidual(inchannel, outchannel, kernel_size,
                                 expand_size, se, activation, stride))
        return nn.Sequential(*bottlenecks)

    def forward(self, x):
        out = self.head(x)
        out = self.bottlenecks(out)
        out = self.tail(out)
        out = self.conv_nbn(out)
        out = out.view(x.size(0), -1)
        return out


@BACKBONE.register_obj
def mobilenetv1(pretrained=False, model_root=None, **kwargs):
    model = MobileNetV1(**kwargs)
    if pretrained:
        load_state_dict(model, model_urls['mobilenetv1'], model_root)
    return model


@BACKBONE.register_obj
def mobilenetv2(pretrained=False, model_root=None, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        load_state_dict(model, model_urls['mobilenetv2'], model_root)
    return model


@BACKBONE.register_obj
def mobilenetv3_large(pretrained=False, model_root=None, **kwargs):
    model = MobileNetV3(net_mode='large', **kwargs)
    if pretrained:
        load_state_dict(model, model_urls['mobilenetv3_large'], model_root)
    return model


@BACKBONE.register_obj
def mobilenetv3_small(pretrained=False, model_root=None, **kwargs):
    model = MobileNetV3(net_mode='small', **kwargs)
    if pretrained:
        load_state_dict(model, model_urls['mobilenetv3_small'], model_root)
    return model
