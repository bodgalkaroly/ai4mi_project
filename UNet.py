"""U-Net (Ronneberger et al., 2015), with batch normalisation and padded convolutions."""

import torch
import torch.nn as nn
from torch import Tensor

from ENet import random_weights_init


def double_conv(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(out_dim),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(out_dim),
                         nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        K: int = kwargs["kernels"] if "kernels" in kwargs else 32  # Width of the first level
        widths: list[int] = [K, K * 2, K * 4, K * 8, K * 16]  # 4 poolings: inputs must be multiples of 16
        self.widths = widths

        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList(double_conv(i, o) for i, o in zip([in_dim] + widths[:-1], widths))

        self.ups = nn.ModuleList(nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
                                 for w in reversed(widths[:-1]))
        self.decoders = nn.ModuleList(double_conv(w * 2, w) for w in reversed(widths[:-1]))

        self.final = nn.Conv2d(K, out_dim, kernel_size=1)

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def encode(self, input: Tensor) -> list[Tensor]:
        features: list[Tensor] = [self.encoders[0](input)]
        for encoder in self.encoders[1:]:
            features.append(encoder(self.pool(features[-1])))
        return features

    def forward(self, input: Tensor) -> Tensor:
        skips = self.encode(input)
        x = skips.pop()

        for up, decoder in zip(self.ups, self.decoders):
            x = decoder(torch.cat((skips.pop(), up(x)), dim=1))

        return self.final(x)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)
