import torch
from torch import nn

from .nets.UNet_3D import UNet_3D


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels=3):
        super(UNet3D, self).__init__()
        self.net = UNet_3D(in_channels, out_channels)

    def forward(self, x):
        return self.net(x)


class AmpUNet3D(UNet3D):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpUNet3D, self).forward(*args)
