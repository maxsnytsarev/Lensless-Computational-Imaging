import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, in_channel, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channel, in_channel, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y += x
        return self.relu2(y)


class BigResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.layers = nn.ModuleList()
        for j in range(4):
            self.layers.append(ResBlock(in_channel))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DRUNet(nn.Module):
    def __init__(self, input_channels, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, channels[0], kernel_size=3, stride=1, padding=1
        )
        self.res1 = BigResBlock(channels[0])

        self.conv2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=2, padding=1
        )
        self.res2 = BigResBlock(channels[1])

        self.conv3 = nn.Conv2d(
            channels[1], channels[2], kernel_size=3, stride=2, padding=1
        )
        self.res3 = BigResBlock(channels[2])

        self.conv4 = nn.Conv2d(
            channels[2], channels[3], kernel_size=3, stride=2, padding=1
        )
        self.res4 = BigResBlock(channels[3])

        self.conv5 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.res5 = BigResBlock(channels[2])

        self.conv6 = nn.ConvTranspose2d(
            channels[2],
            channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.res6 = BigResBlock(channels[1])

        self.conv7 = nn.ConvTranspose2d(
            channels[1],
            channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.res7 = BigResBlock(channels[0])

        self.conv8 = nn.Conv2d(
            channels[0], input_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.res1(x0)

        x0 = self.conv2(x0)
        x1 = self.res2(x0)

        x2 = self.conv3(x1)
        x3 = self.res3(x2)

        x3 = self.conv4(x3)
        y = self.res4(x3)

        y += x3
        y = self.conv5(y)
        y = self.res5(y)

        y += x2
        y = self.conv6(y)
        y = self.res6(y)

        y += x1
        y = self.conv7(y)
        y = self.res7(y)

        y = self.conv8(y)
        return y
