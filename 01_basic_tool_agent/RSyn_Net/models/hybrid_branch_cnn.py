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

class HNBlock(nn.Module):
    def __init__(self, dim):
        super(HNBlock, self).__init__()

        self.Qconv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.Qconv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.Qconv3 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=True)

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=True)

        self.conv4 = nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, stride=1, bias=True)

        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        x1 = self.Qconv1(x)
        x2 = self.Qconv2(x)
        # x2 = self.relu(x2)
        x2 = self.Qconv3(x2)

        x3 = self.conv1(x)
        x4 = self.conv2(x)
        # x4 = self.relu(x4)
        x4 = self.conv3(x4)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv4(x5)

        x5 = self.relu(x5)

        x5 = self.conv5(x5) + x

        return x5





class HybridBranchCNN(nn.Module):

    def __init__(self):
        super(HybridBranchCNN, self).__init__()
        channels = 16
        self.LayerNo = 10
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, padding=2, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, bias=True),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, bias=True),
        )

        self.conv_fin = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1, bias=True)

        layers = []
        for i in range(self.LayerNo):
            layers.append(HNBlock(channels))
        self.fcs = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv_init(x)

        for i in range(5):
            x = self.fcs[i](x)

        x = self.conv_mid(x) + x

        for i in range(5, self.LayerNo):
            x = self.fcs[i](x)

        x = self.conv_fin(x)

        return x


class rec_model(nn.Module):
    def __init__(self):
        super(rec_model, self).__init__()

        self.model = HybridBranchCNN()

    def forward(self, x):
        x = self.model(x)
        return x
