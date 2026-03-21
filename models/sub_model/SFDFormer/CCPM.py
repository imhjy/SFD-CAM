import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class PLM(nn.Module):

    def __init__(self, poolingsize, channels):
        super().__init__()
        self.plm = nn.Sequential(
            nn.MaxPool3d(kernel_size=[poolingsize, 1, 1]),
            nn.Conv3d(channels, channels, 5, padding=2), # 深度可分离卷积
            # nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.Conv3d(channels, channels, 7, stride=1, padding=9, dilation=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.plm(x)


class skip(nn.Module):

    def __init__(self, channels, cube_height):
        super().__init__()
        self.double2dconv = DoubleConv(channels * 2, channels)
        # self.compress_pool =nn.MaxPool3d(kernel_size=[cube_height, 1, 1])
        self.skip_conv = nn.Conv3d(channels, channels, kernel_size=[cube_height, 1, 1], stride=[cube_height, 1, 1])

    def forward(self, x0, x):
        x1 = self.skip_conv(x0)
        x = torch.cat([x1, x], dim=1)
        x = torch.squeeze(x, 2)
        return self.double2dconv(x)


class InConv3d(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.in_conv(x)


class OutConv2d(nn.Module):
    def __init__(self, channels, n_class):
        super(OutConv2d, self).__init__()
        self.conv = nn.Conv2d(channels, n_class, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CCPM(nn.Module):  # Large core fast projection module
    def __init__(self, in_channels=1, channels=64, block_size=[160, 100, 100], **kwargs):
        super(CCPM, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.input3d_1 = InConv3d(in_channels, channels)
        self.input3d_2 = InConv3d(channels, channels)

        self.PLM_U1 = PLM(8, channels)
        self.PLM_U2 = PLM(10, channels)
        self.PLM_U3 = PLM(4, channels)

        self.PLM_D1 = PLM(8, channels)
        self.PLM_D2 = PLM(5, channels)
        self.PLM_D3 = PLM(8, channels)
        self.input2d = skip(channels, block_size[0])  # 把原始三维数据直接卷积成二维，和池化的数据拼接后进行卷积得到值

    def forward(self, x0):
        x1 = self.input3d_1(x0)  # (1,64,160,100,100)
        x2 = self.input3d_2(x1)  # (1,64,160,100,100)

        x1_1 = self.PLM_U1(x1)  # (1,64,20, 100,100)
        x2_1 = self.PLM_D1(x2)  # (1,64,20, 100,100)
        x1_2 = self.PLM_U2(torch.cat([x1_1, x2_1], dim=2))  # (1,64,4, 100,100)
        x2_2 = self.PLM_D2(x2_1)  # (1,64,4, 100,100)
        x1_3 = self.PLM_U3(x1_2)  # (1,64,1, 100,100)
        x2_3 = self.PLM_D3(torch.cat([x1_2, x2_2], dim=2))  # (1,64,1, 100,100)

        x = self.input2d(x1 + x2, x1_3 + x2_3)  # (1,64,100,100)
        return x