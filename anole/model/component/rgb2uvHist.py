"""
This file support the Function Frgb2uv() that convert 
the rgb image to uv parameterized  RGB-uv histogram.

Reference: "Sensor-Independent Illumination Estimation for DNN Models"
url: http://cvil.eecs.yorku.ca/projects/public_html/siie/index.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import COMPONENT


class RGB2UVHist(nn.Module):
    """
    Convert the rgb image to uv parameterized RGB-uv histogram.

    :param image_size: Resize the image to the same size to make
                       the number of pixels consistent.
    :param hist_num: the number of histogram.

    Return: A Tensor with size 3 * hist_num * hist_num.
    """
    def __init__(self, image_size: int = 150, hist_num: int = 61):
        super(RGB2UVHist, self).__init__()
        self.hist_num = hist_num
        self.eps = 6.4 / (hist_num - 1)
        self.size = image_size
        self.A = self.core_params_init()
        self.Asize = torch.nn.Parameter(torch.tensor(self.A.shape[2]),
                                        requires_grad=False)
        self.sigma_u = nn.Parameter(torch.ones(3).view((1, 3, 1, 1)).float())
        self.sigma_v = nn.Parameter(torch.ones(3).view((1, 3, 1, 1)).float())
        self.C = nn.Parameter(torch.ones(3).view((1, 3, 1, 1)).float())
        self.lower = torch.tensor(1e-9)

    def core_params_init(self):
        A_init = torch.unsqueeze(torch.arange(-3.2, 3.21, self.eps), dim=0)
        A_init = torch.repeat_interleave(A_init, self.size * self.size, dim=0)
        A_init = A_init.view((1, 1, -1, self.hist_num))
        return torch.nn.Parameter(A_init, requires_grad=False)

    def rgb2uvhist_param(self, img):
        bn, c, _, _ = img.shape
        img = F.interpolate(img,
                            size=(self.size, self.size),
                            mode="bilinear",
                            align_corners=False)
        img_channelwise = img.view((bn, 3, -1))
        Iy = torch.norm(img_channelwise, dim=1,
                        keepdim=True).reshape(bn, 1, 1, -1)
        father_idx = torch.tensor([0, 1, 2])
        son1_idx = torch.tensor([1, 0, 0])
        son2_idx = torch.tensor([2, 2, 1])
        father = img_channelwise[:, father_idx, :]
        son1 = img_channelwise[:, son1_idx, :]
        son2 = img_channelwise[:, son2_idx, :]
        Iu = torch.log(father / son1).view((bn, 3, -1, 1))
        Iv = torch.log(father / son2).view((bn, 3, -1, 1))
        diff_u = torch.pow(torch.abs(Iu - self.A), 2)
        diff_v = torch.pow(torch.abs(Iv - self.A), 2)
        diff_u = torch.exp(-diff_u / (torch.pow(self.sigma_u, 2) + self.lower))
        diff_v = torch.exp(-diff_v / (torch.pow(self.sigma_v, 2) + self.lower))
        tu = diff_u.permute(0, 1, 3, 2)
        tv = diff_v.permute(0, 1, 2, 3)
        hist = torch.matmul(Iy * tu, tv)
        hist_norm = torch.sqrt(hist * self.C)
        return hist_norm

    def forward(self, x, **kwargs):
        x = self.rgb2uvhist_param(x)
        return x


@COMPONENT.register_obj
def rgb2uv_hist(**kwargs):
    return RGB2UVHist(**kwargs)


if __name__ == '__main__':
    t = RGB2UVHist()
    print(t.A.shape)
