import torch.nn as nn

from ..builder import NECK

__all__ = ['identity_neck']


class IdentityNeck(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityNeck, self).__init__()

    def forward(self, x, **kwargs):
        return x


@NECK.register_obj
def identity_neck(**kwargs):
    return IdentityNeck(**kwargs)
