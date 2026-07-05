import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
from torch import einsum


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads=4, hidden_dim=16, w_size = 4, h_size = 4):
        # hidden_dim // heads, pixel_w // w_size, pixel_h //  h_size
        super().__init__()
        self.scale = (hidden_dim * w_size * h_size / heads)**-0.5
        self.w_size = w_size
        self.h_size = h_size
        self.heads = heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, w, h = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (d c) (x w) (y h) -> b d (c w h) (x y)",d=self.heads, w=self.w_size, h=self.h_size), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b d (x y) (c w h) -> b (d c) (x w) (y h)", w=self.w_size, h=self.h_size, x=h // self.h_size)

        return self.to_out(out)


class Block_attention(nn.Module):
    def __init__(self, dim, wd_size = 96, hd_size = 96, heads=4, hidden_dim=16, w_size = 8, h_size = 8):
        # hidden_dim // heads, pixel_w // w_size, pixel_h //  h_size
        super().__init__()

        self.wd_size = wd_size
        self.hd_size = hd_size
        self.scale = (hidden_dim * w_size * h_size / heads) ** -0.5
        self.w_size = w_size
        self.h_size = h_size
        self.heads = heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b0, c0, w0, h0 = x.shape
        qkv = self.to_qkv(x)


        if w0 % self.wd_size == 0:
            w_pad = 0
        else:
            w_pad = self.wd_size - w0 % self.wd_size

        if h0 % self.hd_size == 0:
            h_pad = 0
        else:
            h_pad = self.hd_size -  h0 % self.hd_size

        pad = (0, h_pad, 0, w_pad, 0, 0)
        qkv = F.pad(qkv, pad, mode='constant', value=0)
        b1, c1, w1, h1 = qkv.shape


        qkv = rearrange(qkv, 'b c (x w) (y h) -> (b x y) c w h', w = self.wd_size, h = self.hd_size)
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b (d c) (x w) (y h) -> b d (c w h) (x y)", d=self.heads, w=self.w_size,
                                h=self.h_size), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b d (x y) (c w h) -> b (d c) (x w) (y h)", w=self.w_size, h=self.h_size,
                        x=self.hd_size // self.h_size)

        out = rearrange(out, '(b x y) c w h -> b c (x w) (y h)', x = w1 // self.wd_size, y = h1 // self.hd_size)
        out = out[:,:,0:w0, 0:h0]

        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Basic_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=True)

        self.attention = PreNorm(dim, Block_attention(dim))
        self.res = ResBlock(dim)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.attention(x1)
        x1 = self.conv2(x1)
        x = x + x1

        x = self.res(x)

        return x


class AttentionResidualCNN(nn.Module):
    def __init__(self, LayerNo):
        super().__init__()
        self.LayerNo = LayerNo
        channels = 16

        self.inin_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1, stride=1, bias=True)

        layers = []
        for i in range(self.LayerNo):
            layers.append(Basic_Block(channels))
        self.fcs = nn.ModuleList(layers)

        self.fina_conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x = self.inin_conv(x)

        for i in range(self.LayerNo):
            x = self.fcs[i](x)

        x = self.fina_conv(x)

        return x


class rec_model(nn.Module):
    def __init__(self):
        super(rec_model, self).__init__()

        self.model = AttentionResidualCNN(LayerNo=5)

    def forward(self, x):
        x = self.model(x)
        return x


