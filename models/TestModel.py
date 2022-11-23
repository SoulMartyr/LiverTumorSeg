import torch
from torch import nn

from .nets.UNet3D import UNet3D


class TestModel(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels=3):
        super(TestModel, self).__init__()
        self.net = UNet3D(in_channels, out_channels)

    def forward(self, x):
        return self.net(x)


class AmpTestModel(TestModel):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpTestModel, self).forward(*args)
