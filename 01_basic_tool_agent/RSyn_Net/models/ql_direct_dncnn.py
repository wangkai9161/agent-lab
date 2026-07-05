import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

class QLConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding=1, stride=1, bias=True):
        super(QLConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 * x2 + x3

        return x


class QLDirectDnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super(QLDirectDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(QLConv(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(QLConv(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(QLConv(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.backbone = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.backbone(x)
        return out

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class rec_model(nn.Module):
    def __init__(self):
        super(rec_model, self).__init__()

        self.model = QLDirectDnCNN(depth=10, n_channels=32)

    def forward(self, x):
        x = self.model(x)
        return x
