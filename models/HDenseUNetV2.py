import torch
import torch.nn as nn

from .nets.DenseUNet2D import DenseUNet2D
from .nets.DenseUNet3D import DenseUNet3D


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ration=0.25):
        super(SqueezeExcite, self).__init__()
        se_channels = int(in_channels * se_ration)
        self.conv1 = nn.Conv3d(in_channels, se_channels, 1)
        self.activate1 = nn.SiLU()
        self.conv2 = nn.Conv3d(se_channels, in_channels, 1)
        self.activate2 = nn.Sigmoid()

    def forward(self, x):
        avg_x = x.mean((2, 3, 4), keepdim=True)
        scale = self.conv1(avg_x)
        scale = self.activate1(scale)
        scale = self.conv2(scale)
        scale = self.activate2(scale)
        return scale * x


class HDenseUNetV2(nn.Module):
    def __init__(self, out_channels: int = 3):
        super(HDenseUNetV2, self).__init__()
        self.dense_unet2d = DenseUNet2D(1, out_channels - 1)
        self.dense_unet3d = DenseUNet3D(1, out_channels - 1)
        self.softmax = nn.Softmax(dim=2)
        self.conv = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(64)
        self.projection1 = nn.Conv3d(64, 2, kernel_size=1, padding="same")
        self.projection2 = nn.Conv3d(4, out_channels, kernel_size=1, padding="same")

    def forward(self, x):
        input2d = x.squeeze(1).permute(1, 0, 2, 3)

        feature2d, classifer2d = self.dense_unet2d(input2d)
        feature3d_liver = feature2d.permute(1, 0, 2, 3).unsqueeze(0)
        classifer3d_liver = classifer2d.permute(1, 0, 2, 3).unsqueeze(0)

        roi_input_3d = self.softmax(classifer3d_liver)[:, 1] * x

        feature3d, classifer3d = self.dense_unet3d(roi_input_3d)

        final = 1.5 * feature3d + 0.5 * feature3d_liver
        final = self.se(final)
        final_conv = self.conv(final)
        # final_conv = self.dropout(final_conv)
        final_bn = self.bn(final_conv)
        final_relu = self.relu(final_bn)

        classifer3d_tumor = self.projection1(final_relu)
        out = self.projection2(torch.cat([classifer3d_liver, classifer3d_tumor], dim=1))
        return out


class AmpHDenseUNetV2(HDenseUNetV2):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpHDenseUNetV2, self).forward(*args)


if __name__ == "__main__":
    a = torch.randn([1, 1, 16, 256, 256])
    model = HDenseUNetV2(out_channels=3)
    res = model(a)
    print(res.shape)
