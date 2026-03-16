import torch
import torch.nn as nn
import torch.nn.functional as F


class MNet(nn.Module):
    """
    Huazhu Fu, Jun Cheng, Yanwu Xu, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation", IEEE Transactions on Medical Imaging (TMI), vol. 37, no. 7, pp. 1597–1605, 2018.
    """
    def __init__(self, in_channel=3, n_classes=3):
        super(MNet, self).__init__()

        # Downsampling layers
        self.avg_pool2 = nn.AvgPool2d(2)
        self.avg_pool3 = nn.AvgPool2d(2)
        self.avg_pool4 = nn.AvgPool2d(2)

        # Block 1
        self.block1_conv1 = nn.Conv2d(in_channel, 32, 3, padding='same')
        self.block1_conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2)

        # Block 2
        self.block2_input1 = nn.Conv2d(in_channel, 64, 3, padding='same')
        self.block2_conv1 = nn.Conv2d(64 + 32, 64, 3, padding='same')
        self.block2_conv2 = nn.Conv2d(64, 64, 3, padding='same')
        self.pool2 = nn.MaxPool2d(2)

        # Block 3
        self.block3_input1 = nn.Conv2d(in_channel, 128, 3, padding='same')
        self.block3_conv1 = nn.Conv2d(128 + 64, 128, 3, padding='same')
        self.block3_conv2 = nn.Conv2d(128, 128, 3, padding='same')
        self.pool3 = nn.MaxPool2d(2)

        # Block 4
        self.block4_input1 = nn.Conv2d(in_channel, 256, 3, padding='same')
        self.block4_conv1 = nn.Conv2d(256 + 128, 256, 3, padding='same')
        self.block4_conv2 = nn.Conv2d(256, 256, 3, padding='same')
        self.pool4 = nn.MaxPool2d(2)

        # Block 5 (Bottleneck)
        self.block5_conv1 = nn.Conv2d(256, 512, 3, padding='same')
        self.block5_conv2 = nn.Conv2d(512, 512, 3, padding='same')

        # Block 6 (Decoder)
        self.block6_dconv = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.block6_conv1 = nn.Conv2d(512, 256, 3, padding='same')
        self.block6_conv2 = nn.Conv2d(256, 256, 3, padding='same')

        # Block 7 (Decoder)
        self.block7_dconv = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.block7_conv1 = nn.Conv2d(256, 128, 3, padding='same')
        self.block7_conv2 = nn.Conv2d(128, 128, 3, padding='same')

        # Block 8 (Decoder)
        self.block8_dconv = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.block8_conv1 = nn.Conv2d(128, 64, 3, padding='same')
        self.block8_conv2 = nn.Conv2d(64, 64, 3, padding='same')

        # Block 9 (Decoder)
        self.block9_dconv = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.block9_conv1 = nn.Conv2d(64, 32, 3, padding='same')
        self.block9_conv2 = nn.Conv2d(32, 32, 3, padding='same')

        # Side outputs
        self.side6_conv = nn.Conv2d(256, n_classes, 1)
        self.side7_conv = nn.Conv2d(128, n_classes, 1)
        self.side8_conv = nn.Conv2d(64, n_classes, 1)
        self.side9_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Downscaled images
        scale_img_2 = self.avg_pool2(x)
        scale_img_3 = self.avg_pool3(scale_img_2)
        scale_img_4 = self.avg_pool4(scale_img_3)

        # Encoder path
        # Block 1
        conv1 = F.relu(self.block1_conv1(x))
        conv1 = F.relu(self.block1_conv2(conv1))
        pool1 = self.pool1(conv1)

        # Block 2
        input2 = F.relu(self.block2_input1(scale_img_2))
        input2 = torch.cat([input2, pool1], dim=1)
        conv2 = F.relu(self.block2_conv1(input2))
        conv2 = F.relu(self.block2_conv2(conv2))
        pool2 = self.pool2(conv2)

        # Block 3
        input3 = F.relu(self.block3_input1(scale_img_3))
        input3 = torch.cat([input3, pool2], dim=1)
        conv3 = F.relu(self.block3_conv1(input3))
        conv3 = F.relu(self.block3_conv2(conv3))
        pool3 = self.pool3(conv3)

        # Block 4
        input4 = F.relu(self.block4_input1(scale_img_4))
        input4 = torch.cat([input4, pool3], dim=1)
        conv4 = F.relu(self.block4_conv1(input4))
        conv4 = F.relu(self.block4_conv2(conv4))
        pool4 = self.pool4(conv4)

        # Block 5 (Bottleneck)
        conv5 = F.relu(self.block5_conv1(pool4))
        conv5 = F.relu(self.block5_conv2(conv5))

        # Decoder path
        # Block 6
        up6 = self.block6_dconv(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = F.relu(self.block6_conv1(up6))
        conv6 = F.relu(self.block6_conv2(conv6))

        # Block 7
        up7 = self.block7_dconv(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = F.relu(self.block7_conv1(up7))
        conv7 = F.relu(self.block7_conv2(conv7))

        # Block 8
        up8 = self.block8_dconv(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = F.relu(self.block8_conv1(up8))
        conv8 = F.relu(self.block8_conv2(conv8))

        # Block 9
        up9 = self.block9_dconv(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = F.relu(self.block9_conv1(up9))
        conv9 = F.relu(self.block9_conv2(conv9))

        # Side outputs
        side6 = F.interpolate(conv6, scale_factor=8, mode='bilinear', align_corners=False)
        side7 = F.interpolate(conv7, scale_factor=4, mode='bilinear', align_corners=False)
        side8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=False)

        # out6 = torch.sigmoid(self.side6_conv(side6))
        # out7 = torch.sigmoid(self.side7_conv(side7))
        # out8 = torch.sigmoid(self.side8_conv(side8))
        # out9 = torch.sigmoid(self.side9_conv(conv9))

        out6 = self.side6_conv(side6)
        out7 = self.side7_conv(side7)
        out8 = self.side8_conv(side8)
        out9 = self.side9_conv(conv9)

        # Average output
        out10 = (out6 + out7 + out8 + out9) / 4.0

        return {
            'out': out10
        }


# 使用示例
if __name__ == "__main__":
    size_set = 800
    model = MNet(in_channel=1, n_classes=5)

    # 创建测试输入 (batch_size, channels, height, width)
    x = torch.randn(4, 1, 128, 128)

    # 前向传播
    outputs = model(x)

    # 输出形状检查
    for i, out in enumerate(outputs):
        print(f"Output key {out} shape: {outputs[out].shape}")
