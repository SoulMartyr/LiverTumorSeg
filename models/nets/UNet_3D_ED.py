import torch
import torch.nn as nn


class InitConv(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16, dropout_rate: int = 0.0):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        out = self.conv(x)
        if self.dropout_rate > 0:
            out = self.dropout(out)

        return out


class EnBlockGN(nn.Module):
    def __init__(self, in_channels: int):
        super(EnBlockGN, self).__init__()

        self.bn1 = nn.GroupNorm(8, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = nn.GroupNorm(8, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.bn1(x)
        out1 = self.relu1(out1)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = self.relu2(out2)
        out2 = self.conv2(out2)
        out = out2 + x

        return out


class EnDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)

        return out


class Unet3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super(Unet3DEncoder, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels)
        self.EnBlock1 = EnBlockGN(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels * 2)

        self.EnBlock2_1 = EnBlockGN(in_channels=base_channels * 2)
        self.EnBlock2_2 = EnBlockGN(in_channels=base_channels * 2)
        self.EnDown2 = EnDown(in_channels=base_channels * 2, out_channels=base_channels * 4)

        self.EnBlock3_1 = EnBlockGN(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlockGN(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels * 4, out_channels=base_channels * 8)

        self.EnBlock4_1 = EnBlockGN(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlockGN(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlockGN(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlockGN(in_channels=base_channels * 8)

    def forward(self, x):
        x = self.InitConv(x)  # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 16, 16, 16)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)
        output = self.EnBlock4_4(x4_3)  # (1, 128, 16, 16, 16)

        return x1_1, x2_1, x3_1, output


class DeBlockBN(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 1):
        super(DeBlockBN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels // ratio, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels // ratio)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels // ratio, in_channels // ratio, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels // ratio)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class DeUpCat(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DeUpCat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class Unet3DDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_dim: int, patch_dim: int, embed_dim: int):
        super(Unet3DDecoder, self).__init__()

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim

        self.Softmax = nn.Softmax(dim=1)

        self.EnBlock1 = DeBlockBN(in_channels=in_channels, ratio=4)
        self.EnBlock2 = DeBlockBN(in_channels=in_channels // 4)

        self.DeUp4 = DeUpCat(in_channels=in_channels // 4, out_channels=in_channels // 8)
        self.DeBlock4 = DeBlockBN(in_channels=in_channels // 8)

        self.DeUp3 = DeUpCat(in_channels=in_channels // 8, out_channels=in_channels // 16)
        self.DeBlock3 = DeBlockBN(in_channels=in_channels // 16)

        self.DeUp2 = DeUpCat(in_channels=in_channels // 16, out_channels=in_channels // 32)
        self.DeBlock2 = DeBlockBN(in_channels=in_channels // 32)

        self.EndConv = nn.Conv3d(in_channels // 32, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4, trans_out):
        x = trans_out
        x = x.view(x4.size(0), x4.size(2), x4.size(3), x4.size(4), self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.EnBlock1(x)
        x = self.EnBlock2(x)

        y4 = self.DeUp4(x, x3)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)

        y3 = self.DeUp3(y4, x2)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        y2 = self.DeUp2(y3, x1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.EndConv(y2)  # (1, 4, 128, 128, 128)
        # out = self.Softmax(y)
        return y


if __name__ == "__main__":
    a = torch.randn([5, 1, 16, 128, 128])
    model = Unet3DEncoder(in_channels=1)
    d = model(a)
