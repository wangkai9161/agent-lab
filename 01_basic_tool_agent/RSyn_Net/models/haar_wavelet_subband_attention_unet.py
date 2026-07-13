""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedECA(nn.Module):
    """Lightweight channel attention reused across wavelet subbands."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.activation(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SubbandAttention(nn.Module):
    """Subband-aware reweighting followed by a shared ECA refinement."""

    def __init__(self, in_channels, eca_kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.subband_gate = nn.Sequential(
            nn.Linear(4, 4, bias=True),
            nn.Sigmoid()
        )
        self.shared_eca = SharedECA(in_channels, kernel_size=eca_kernel_size)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != 4 * self.in_channels:
            raise ValueError(
                f"Expected input channels = {4 * self.in_channels}, but got {c}"
            )

        x = x.reshape(b, self.in_channels, 4, h, w)

        # Learn a lightweight global preference over LL/LH/HL/HH.
        subband_desc = x.mean(dim=(1, 3, 4))
        subband_weight = self.subband_gate(subband_desc).view(b, 1, 4, 1, 1)
        x = x * subband_weight

        refined_subbands = []
        for idx in range(4):
            refined_subbands.append(self.shared_eca(x[:, :, idx, :, :]))

        return torch.cat(refined_subbands, dim=1)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.down_conv = nn.Sequential(
            HaarDownsampling(in_channels, pad_mode='replicate'),
            DoublesubConv(in_channels, out_channels, mid_channels=mid_channels)
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
            SubbandAttention(in_channels),
            SubbandSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=16, num_channels=mid_channels * 4),
            nn.ReLU(inplace=True),
            SubbandAttention(mid_channels),
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
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 512, mid_channels=256))
        self.up1 = (Up(512, 512, 640))
        self.up2 = (Up(512, 256, 384))
        self.up3 = (Up(256, 128, 192))
        self.up4 = (Up(128, 64, 96))

        self.outc = (OutConv(64, n_classes))

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
    x = torch.randn(1, 1, 128, 129).to(device)

    try:
        with torch.no_grad():
            y = model(x)

        print("Forward pass: SUCCESS")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {y.shape}")

    except Exception as e:
        print("Forward pass: FAILED")
        print(e)

    # ===== 3. verify HaarDownsampling <-> HaarUpsampling invertibility =====
    print("\n=== Haar invertibility check ===")
    test_shapes = [(128, 128), (127, 127), (129, 131), (255, 257)]
    in_channels = 3
    haar_down = HaarDownsampling(in_channels, pad_mode='replicate').to(device)
    haar_up = HaarUpsampling(in_channels * 4).to(device)

    with torch.no_grad():
        for h, w in test_shapes:
            x_test = torch.randn(1, in_channels, h, w, device=device)
            y_sub = haar_down(x_test)
            x_rec = haar_up(y_sub, x_test)

            max_abs_err = (x_rec - x_test).abs().max().item()
            mse = F.mse_loss(x_rec, x_test).item()
            is_close = torch.allclose(x_rec, x_test, atol=1e-6, rtol=1e-6)

            print(
                f"shape=({h},{w}) | down={tuple(y_sub.shape)} | rec={tuple(x_rec.shape)} "
                f"| max_abs_err={max_abs_err:.3e} | mse={mse:.3e} | allclose={is_close}"
            )
