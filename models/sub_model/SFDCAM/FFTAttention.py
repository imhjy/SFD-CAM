import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm_F(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_F, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FSAS(nn.Module):
    def __init__(self, dim=64, bias=False, patch_size=8):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm_F(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = patch_size

    def forward(self, x):
        hidden = self.to_hidden(x)  # (4,384,32,32)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)  # (4,128,32,32) (4,128,32,32) (4,128,32,32)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)  # (4,128,4,4,8,8)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)  # (4,128,4,4,8,8)
        q_fft = torch.fft.rfft2(q_patch.float())  # (4,128,4,4,8,5)
        k_fft = torch.fft.rfft2(k_patch.float())  # (4,128,4,4,8,5)

        out = q_fft * k_fft  # (4,128,4,4,8,5)
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))  # (4,128,4,4,8,8)
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


class FFTAttention(nn.Module):
    def __init__(self, in_dim=64, dim=64, patch_size=8):
        super(FFTAttention, self).__init__()

        self.conv_first = nn.Conv2d(in_dim, dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(dim, in_dim, 3, 1, 1)
        self.norm1 = LayerNorm_F(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm_F(dim, LayerNorm_type='WithBias')
        self.att1 = FSAS(dim, patch_size=patch_size)
        self.att2 = FSAS(dim, patch_size=patch_size)

    def forward(self, x):
        x1 = self.conv_first(x)

        x2 = self.att1(self.norm1(x1))

        x3 = self.att2(self.norm2(x1 + x2))

        x4 = self.conv_last(x3 + x2 + x1)
        return x4 + x