"""Minimal CSRNet implementation for density map estimation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    def __init__(self, load_weights: bool = False):
        super().__init__()
        frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(frontend_feat)
        self.backend = make_layers(backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._initialize_frontend(mod)

    def _initialize_frontend(self, vgg):
        fsd = self.frontend.state_dict()
        vsd = vgg.features.state_dict()
        for k in fsd.keys():
            fsd[k] = vsd[k]
        self.frontend.load_state_dict(fsd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


def make_layers(cfg, in_channels: int = 3, dilation: bool = False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)
