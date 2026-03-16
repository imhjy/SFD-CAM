from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

__all__ = ["AttentionUnet"]


class ConvBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[int] | int = 3,
            strides: int = 1,
            dropout=0.0,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionLayer(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            submodule: nn.Module,
            up_kernel_size=3,
            strides=2,
            dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class AttentionUnet(nn.Module):
    """
    Attention UNet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: 输入图像的空间维度数, 二维图像设为2
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        up_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Sequence[int] | int = 3,
            up_kernel_size: Sequence[int] | int = 3,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout

        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
            kernel_size=self.kernel_size,
        )
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                            kernel_size=self.kernel_size,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
                kernel_size=self.kernel_size,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        x_m: torch.Tensor = self.model(x)
        return {'out': x_m}


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import re

    model = AttentionUnet(spatial_dims=2, in_channels=1, out_channels=2, channels=[64, 128, 256, 512],
                          strides=[2, 2, 2])
    t1 = torch.randn((2, 1, 480, 480))
    print(model(t1)['out'].shape)
    macs, params = get_model_complexity_info(model, (1, 480, 480), as_strings=True, print_per_layer_stat=True)
    print('GMAC: ', macs, 'params: ', params)  # flops:  152.26 GMac params:  7.87 M   GMAC =（乘法累加运算次数）/（10⁹）
    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('GFlops: {} {}Flops'.format(flops, flops_unit))  # GFlops: 304.52 GFlops
