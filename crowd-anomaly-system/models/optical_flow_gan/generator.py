"""U-Net style generator for flow reconstruction."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_bn: bool = True):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FlowGenerator(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 2):
        super().__init__()
        self.e1 = ConvBlock(in_channels, 64, down=True, use_bn=False)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.d1 = ConvBlock(512, 256, down=False)
        self.d2 = ConvBlock(512, 128, down=False)
        self.d3 = ConvBlock(256, 64, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.bottleneck(e4)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e3], dim=1))
        d3 = self.d3(torch.cat([d2, e2], dim=1))
        return self.final(torch.cat([d3, e1], dim=1))
