""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()


        self.up = HaarUpsampling(in_channels)
        self.conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)
        self.conv2 = DoublesubConv(out_channels//4, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            HaarDownsampling(in_channels, pad_mode='replicate'),
            DoublesubConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)



class DoublesubConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        self.double_conv = nn.Sequential(
            SubbandSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            SubbandSeparableConv(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.double_conv(x)



class SubbandSeparableConv(nn.Module):
    """
    对 Haar 下采样得到的 4 个子带分别做不同卷积，再拼接回来。

    Input:
        x: [B, 4*C_in, H, W]

    Output:
        out: [B, 4*C_out, H, W]
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_ll = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_lh = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_hl = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_hh = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != 4 * self.in_channels:
            raise ValueError(
                f"Expected input channels = {4 * self.in_channels}, but got {c}"
            )

        # [B, 4C, H, W] -> [B, C, 4, H, W]
        x = x.view(b, self.in_channels, 4, h, w)

        ll = x[:, :, 0, :, :]
        lh = x[:, :, 1, :, :]
        hl = x[:, :, 2, :, :]
        hh = x[:, :, 3, :, :]

        ll = self.conv_ll(ll)
        lh = self.conv_lh(lh)
        hl = self.conv_hl(hl)
        hh = self.conv_hh(hh)

        out = torch.cat([ll, lh, hl, hh], dim=1)
        return out


class HaarDownsampling(nn.Module):
    def __init__(self, in_channels, pad_mode='replicate'):
        super().__init__()
        self.in_channels = in_channels
        self.pad_mode = pad_mode

        ll = torch.tensor([[1., 1.],
                           [1., 1.]]) / 2.0
        lh = torch.tensor([[1., 1.],
                           [-1., -1.]]) / 2.0
        hl = torch.tensor([[1., -1.],
                           [1., -1.]]) / 2.0
        hh = torch.tensor([[1., -1.],
                           [-1., 1.]]) / 2.0

        filters = torch.stack([ll, lh, hl, hh], dim=0)   # [4, 2, 2]
        filters = filters.unsqueeze(1)                   # [4, 1, 2, 2]
        filters = filters.repeat(in_channels, 1, 1, 1)  # [4*C, 1, 2, 2]

        self.register_buffer('filters', filters)

    def forward(self, x):
        h, w = x.shape[-2:]

        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)

        filters = self.filters.to(device=x.device, dtype=x.dtype)
        out = F.conv2d(x, filters, stride=2, groups=self.in_channels)
        return out


class HaarUpsampling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        if in_channels % 4 != 0:
            raise ValueError(f"in_channels must be divisible by 4, but got {in_channels}")
        self.in_channels = in_channels
        self.out_channels = in_channels // 4

    def forward(self, x1, x2):
        """
        x1: [B, 4C, H, W]
        x2: reference tensor providing target spatial size
        return: [B, C, H_target, W_target]
        """
        b, c, h, w = x1.shape
        if c != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, but got {c}")

        c_out = self.out_channels

        # 关键：按每个原始通道的 4 个子带重排
        x1 = x1.view(b, c_out, 4, h, w)

        ll = x1[:, :, 0, :, :]
        lh = x1[:, :, 1, :, :]
        hl = x1[:, :, 2, :, :]
        hh = x1[:, :, 3, :, :]

        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll + lh - hl - hh) * 0.5
        x10 = (ll - lh + hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        out = torch.empty(
            (b, c_out, h * 2, w * 2),
            device=x1.device,
            dtype=x1.dtype
        )

        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11

        target_h, target_w = x2.shape[-2:]
        out = out[:, :, :target_h, :target_w]

        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        n_channels, n_classes = 1, 1
        bilinear = True
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear



        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 64))
        self.down2 = (Down(64, 256))
        self.down3 = (Down(256, 1024))
        self.down4 = (Down(1024, 4096))
        self.up1 = (Up(4096, 1024, 2048))
        self.up2 = (Up(1024, 256, 512))
        self.up3 = (Up(256, 64, 128))
        self.up4 = (Up(64, 16, 32))

        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class rec_model(nn.Module):
    def __init__(self):
        super(rec_model, self).__init__()

        self.model = UNet()

    def forward(self, x):
        x = self.model(x)
        return x




if __name__ == "__main__":
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = rec_model().to(device)

    # ===== 1. 统计参数量 =====
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # ===== 2. 构造输入测试 forward =====
    # 你可以改成你实际输入尺寸，比如 1x1x256x256
    x = torch.randn(1, 1, 128, 128).to(device)

    try:
        with torch.no_grad():
            y = model(x)

        print("Forward pass: SUCCESS")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {y.shape}")

    except Exception as e:
        print("Forward pass: FAILED")
        print(e)