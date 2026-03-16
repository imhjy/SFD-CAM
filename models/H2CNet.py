import torch.nn as nn
import torch


class UndirectAVGpool(nn.Module):
    # ks表示几个压在一起
    def __init__(self, ks):
        super().__init__()
        self.maxpool = nn.AvgPool3d(kernel_size=[ks, 1, 1])

    def forward(self, x):
        x = self.maxpool(x)
        return x


class DoubleConv2D(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(channels, channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(channels // 2, channels // 2, kernel_size=2, stride=2)  # cyr6e# channels ?

        self.conv = DoubleConv2D(channels * 2, channels)
        self.channel_attention = ChannelAttention(512)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        attention_value = self.channel_attention(x)
        out = x.mul(attention_value)

        return self.conv(out)


class UNet(nn.Module):
    def __init__(self, in_channels, channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.inc = DoubleConv2D(in_channels, channels)
        self.down1 = Down(channels)
        self.down2 = Down(channels)
        self.down3 = Down(channels)
        self.down4 = Down(channels)
        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        self.outc = nn.Sequential(
            nn.Conv2d(channels, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.PReLU(n_classes),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.PReLU(n_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feature = self.up4(x, x1)
        logits = self.outc(feature)
        return logits


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class H2CNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(H2CNet, self).__init__()
        self.FPM1 = UndirectAVGpool(1)  # 128->8
        # self.FPM2 = directMAXpool(4)#8->2
        self.conv1d = DoubleConv2D(180, 256)
        # self.conv2d = DoubleConv2D(5,64)
        self.FPM3 = UndirectAVGpool(8)  # 2->1
        # self.FPM4 = directMAXpool(8)#2->1
        self.SegNet2D = UNet(256, 256, n_classes)
        # self.SegNet2D_2 = UNet(64, 64,n_classes)
        self.channel_attention = ChannelAttention(256)

        # self.SegNet2D = create_RepLKNet31B(num_classes=n_classes)

    def forward(self, x):
        # print("x",x.shape)
        # x1 = self.FPM1(x)
        # x1,x2 = x.split(1,dim=1)
        x1 = self.FPM1(x)
        # print(x1.shape)
        # print("x1",x1.shape)
        # x2 = self.FPM2(x)
        # print("x2",x2.shape)
        x3 = self.FPM3(x)
        # print(x3.shape)
        # print("x3",x3.shape)
        # x2 = self.FPM3(x1)
        # print("x4",x4.shape)
        x4 = torch.cat([x1, x3], dim=2)
        # print(x.shape)
        # print(x3.shape)
        x3 = torch.squeeze(x4, 1)

        x3 = self.conv1d(x3)
        attention_value = self.channel_attention(x3)
        out = x3.mul(attention_value)
        logits = self.SegNet2D(out)
        # x = self.conv2d(x)
        # x = self.SegNet2D_2(out)
        # logits = torch.unsqueeze(x, 2)
        return {
            'out': logits
        }


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import re
    import copy

    model = H2CNet(in_channels=1, n_classes=5).to('cuda')
    t1 = torch.randn((4, 1, 160, 128, 128)).to('cuda')
    out = model(t1)
    print(model)
    print(out['out'].shape)
    # while True:
    #     pass
    model = copy.deepcopy(model)
    macs, params = get_model_complexity_info(model, (1, 160, 128, 128), as_strings=True, print_per_layer_stat=True)
    print('GMAC: ', macs, 'params: ', params)  # GMAC:  22.58 GMac params:  37.22 M   GMAC =（乘法累加运算次数）/（10⁹）
    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('GFlops: {} {}Flops'.format(flops, flops_unit))  # GFlops: 45.16 GFlops
