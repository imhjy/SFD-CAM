import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        model = []
        out_channels = int(in_channels / 2)
        model += [ConvLeakyRelu2d(in_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, 4, activation='Sigmod', kernel_size=3, padding=1, stride=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ConvLeakyRelu2d(nn.Module):
    """
    Conv2d + Norm(optional) + Activation(optional)
    支持 BatchNorm / GroupNorm / 无归一化
    """

    def __init__(self, in_channels, out_channels,
                 norm='Group', activation='ReLU',
                 kernel_size=3, padding=1, stride=1,
                 dilation=1, num_groups=32):
        """
        Args:
            norm: 归一化类型，可选 ['Batch', 'Group', None]
            activation: 激活类型，可选 ['LReLU', 'ReLU', 'Sigmoid', 'Tanh', None]
            num_groups: GroupNorm 的分组数（默认 8，可根据 out_channels 调整）
        """
        super(ConvLeakyRelu2d, self).__init__()

        model = []
        # --- 卷积层 ---
        model += [nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            groups=out_channels,
                            bias=(norm is None))]  # 有Norm时可去掉bias

        # --- 归一化层 ---
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        elif norm == 'Group':
            # 确保分组数不大于通道数
            g = min(num_groups, out_channels)
            model += [nn.GroupNorm(g, out_channels)]

        # --- 激活层 ---
        if activation == 'LReLU':
            model += [nn.LeakyReLU(inplace=True)]
        elif activation == 'ReLU':
            model += [nn.ReLU(inplace=True)]
        elif activation == 'Sigmoid':
            model += [nn.Sigmoid()]
        elif activation == 'Tanh':
            model += [nn.Tanh()]
        # 可选 None（无激活）

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class SplitModule(nn.Module):
    def __init__(self, in_channels, mid_channels=None, activation='relu', norm='Group', num_groups=32):
        """
        Args:
            in_channels: 输入通道数（必须能被4整除）
            mid_channels: 中间层通道数（默认 = in_channels // 4）
            activation: 激活函数类型 ('relu', 'leakyrelu', 'gelu', 'silu'等)
            norm: 归一化类型，可选 ['Batch', 'Group', 'Instance', 'Layer', None]
            num_groups: GroupNorm 的组数（当 norm='Group' 时使用）
        """
        super(SplitModule, self).__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4"
        c = in_channels // 4
        mid_channels = mid_channels or c

        # ---- 激活函数选择 ----
        act_layer = {
            'relu': nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.1, inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True)
        }.get(activation.lower(), nn.ReLU(inplace=True))

        # ---- 内部辅助函数 ----
        def make_norm_layer(channels):
            """根据 norm 类型创建对应的归一化层"""
            if norm == 'Batch':
                return nn.BatchNorm2d(channels)
            elif norm == 'Group':
                g = min(num_groups, channels)
                return nn.GroupNorm(g, channels)
            elif norm == 'Instance':
                return nn.InstanceNorm2d(channels, affine=True)
            elif norm == 'Layer':
                # 对 CNN 来说 LayerNorm 需要指定 normalized_shape
                return nn.LayerNorm([channels, 1, 1])
            else:
                return nn.Identity()  # 无归一化

        def make_branch():
            """构建一个方向的分支结构：Conv -> Norm -> Act -> Conv -> Norm -> Act"""
            layers = [
                nn.Conv2d(c, mid_channels, kernel_size=3, stride=1, padding=1, groups=c, bias=(norm is None)),
                make_norm_layer(mid_channels),
                act_layer,
                nn.Conv2d(mid_channels, c, kernel_size=3, stride=1, padding=1, groups=c, bias=(norm is None)),
                make_norm_layer(c),
                act_layer
            ]
            return nn.Sequential(*layers)

        # ---- 四个方向分支 ----
        self.branch_left = make_branch()
        self.branch_right = make_branch()
        self.branch_up = make_branch()
        self.branch_down = make_branch()

    def forward(self, x):
        # ---- 通道划分 ----
        x_left, x_right, x_up, x_down = torch.chunk(x, 4, dim=1)

        # ---- 各方向独立处理 ----
        out_left = self.branch_left(x_left)
        out_right = self.branch_right(x_right)
        out_up = self.branch_up(x_up)
        out_down = self.branch_down(x_down)

        # ---- 返回结果 ----
        return out_left, out_right, out_up, out_down


class DSCM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(DSCM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = SplitModule(self.out_channels)
        self.irnn2 = SplitModule(self.out_channels)
        self.conv_in = ConvLeakyRelu2d(in_channels, in_channels, activation='ReLU')
        self.conv2 = ConvLeakyRelu2d(in_channels, in_channels,  kernel_size=3, padding=1, stride=1)
        self.conv3 = ConvLeakyRelu2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.attention = attention
        self.attention_layer = Attention(in_channels)
        self.conv_out = ConvLeakyRelu2d(in_channels, in_channels, activation='ReLU', kernel_size=3, padding=1,
                                        stride=1)

    def forward(self, x):
        weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            # print('top_up device:', top_up.device, 'weight device:', weight.device)
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        mask = self.conv_out(out)
        return mask
