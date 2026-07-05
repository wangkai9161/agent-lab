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



class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = QLConv(dim, dim, kernel_size=3, padding=1, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = QLConv(dim, dim, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))



class QLResidualBlockCNN(nn.Module):

    def __init__(self, LayerNo):
        super(QLResidualBlockCNN, self).__init__()
        channels = 32
        self.LayerNo = LayerNo

        self.inin_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1, stride=1, bias=True)

        layers = []
        for i in range(self.LayerNo):
            layers.append(ResBlock(channels))
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

        self.model = QLResidualBlockCNN(LayerNo=10)

    def forward(self, x):
        x = self.model(x)
        return x
