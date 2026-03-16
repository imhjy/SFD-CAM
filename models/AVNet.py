"""
https://github.com/dleninja/mf-av-net
AV-Net: deep learning for fully automated artery-vein classification in Optical Coherence Tomography Angiography
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """DenseNet中的卷积子块"""

    def __init__(self, in_channels, growth_rate, name="conv_block"):
        super().__init__()
        self.name = name
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    """多个ConvBlock组成的Dense Block"""

    def __init__(self, in_channels, num_layers, growth_rate, name="dense_block"):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            layers.append(ConvBlock(channels, growth_rate, name=f"{name}_block{i + 1}"))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


class TransitionBlock(nn.Module):
    """DenseNet中的过渡层 (用于降采样 + 压缩通道数)"""

    def __init__(self, in_channels, reduction=0.5, name="transition_block"):
        super().__init__()
        out_channels = int(in_channels * reduction)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.relu(self.bn(x))
        x2 = self.conv(x1)
        x_pool = self.pool(x2)
        return x_pool, x2  # pool结果, conv结果


class DecoderBlock(nn.Module):
    """解码器模块：拼接 + 两个3x3卷积"""

    def __init__(self, in_channels, skip_channels, out_channels, name="decoder_block"):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AVNet(nn.Module):
    """AV-Net主结构"""

    def __init__(self, block_layers=[6, 12, 24, 16], in_channels=3,out_channels=5):
        super().__init__()

        # ===== Encoder Stem =====
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===== Dense Blocks =====
        # conv2
        self.block1 = DenseBlock(64, block_layers[0], growth_rate=32, name="conv2")
        self.trans1 = TransitionBlock(self.block1.out_channels, reduction=0.5, name="pool2")

        # conv3
        self.block2 = DenseBlock(self.trans1.out_channels, block_layers[1], growth_rate=32, name="conv3")
        self.trans2 = TransitionBlock(self.block2.out_channels, reduction=0.5, name="pool3")

        # conv4
        self.block3 = DenseBlock(self.trans2.out_channels, block_layers[2], growth_rate=32, name="conv4")
        self.trans3 = TransitionBlock(self.block3.out_channels, reduction=0.5, name="pool4")

        # conv5
        self.block4 = DenseBlock(self.trans3.out_channels, block_layers[3], growth_rate=32, name="conv5")
        self.bn_final = nn.BatchNorm2d(self.block4.out_channels)
        self.relu_final = nn.ReLU(inplace=True)

        # ===== Decoder =====
        self.decode1 = DecoderBlock(self.block4.out_channels, self.trans3.out_channels, 256, name="decode1")
        self.decode2 = DecoderBlock(256, self.trans2.out_channels, 128, name="decode2")
        self.decode3 = DecoderBlock(128, self.trans1.out_channels, 64, name="decode3")
        self.decode4 = DecoderBlock(64, 64, 32, name="decode4")

        # ===== Final layers =====
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(16, out_channels, kernel_size=1)


    def forward(self, x):
        # ----- Encoder -----
        x0 = self.relu1(self.bn1(self.conv1(x)))  # conv1
        x_pool = self.pool1(x0)

        x2 = self.block1(x_pool)
        x2_pool, skip1 = self.trans1(x2)

        x3 = self.block2(x2_pool)
        x3_pool, skip2 = self.trans2(x3)

        x4 = self.block3(x3_pool)
        x4_pool, skip3 = self.trans3(x4)

        x5 = self.block4(x4_pool)
        x5 = self.relu_final(self.bn_final(x5))

        # ----- Decoder -----
        x = self.decode1(x5, skip3)
        x = self.decode2(x, skip2)
        x = self.decode3(x, skip1)
        x = self.decode4(x, x0)

        # ----- Output -----
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_final(x)
        x = self.output_conv(x)
        return {
            'out': x
        }


if __name__ == "__main__":
    model = AVNet(block_layers=[6, 12, 24, 16], in_channels=1,out_channels=5)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Output shape:", y['out'].shape)
