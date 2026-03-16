import torch
import torch.nn as nn


class InConv3d(nn.Module):

    def __init__(self, in_channels, channels):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.in_conv(x)


class PLM(nn.Module):

    def __init__(self, poolingsize, channels):
        super().__init__()
        self.plm = nn.Sequential(
            nn.MaxPool3d(kernel_size=[poolingsize, 1, 1]),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.plm(x)


class OutConv3d(nn.Module):

    def __init__(self, channels, n_class):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, n_class, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.out_conv(x)


class IPN(nn.Module):
    def __init__(self, in_channels, channels, n_classes):
        super(IPN, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.input = InConv3d(in_channels, channels)
        self.PLM1 = PLM(5, channels)
        self.PLM2 = PLM(4, channels)
        self.PLM3 = PLM(4, channels)
        self.PLM4 = PLM(2, channels)
        self.output = OutConv3d(channels, n_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.PLM1(x)
        x = self.PLM2(x)
        x = self.PLM3(x)
        feature = self.PLM4(x)
        feature = torch.squeeze(feature, 2)
        logits = self.output(feature)
        return {
            'out': logits,
            'feature': feature
        }

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import re

    model = IPN(in_channels=2, channels=64,n_classes=5)
    t1 = torch.randn((1, 2, 160, 100, 100))
    out = model(t1)
    print(out)
    macs, params = get_model_complexity_info(model, (2, 160, 100, 100), as_strings=True, print_per_layer_stat=True)
    print('GMAC: ', macs, 'params: ', params)  # GMAC:  71.59 GMac params:  2.25 M   GMAC =（乘法累加运算次数）/（10⁹）
    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('GFlops: {} {}Flops'.format(flops, flops_unit))  # GFlops: 143.18 GFlops