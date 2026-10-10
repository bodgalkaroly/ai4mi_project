"""UNet++ (Zhou et al., 2018), without deep supervision: the U-Net of UNet.py
with its skip connections replaced by nested, dense paths of convolution
blocks, which bring the encoder features closer to the decoder ones before
fusing them."""

import torch
import torch.nn as nn
from torch import Tensor

from ENet import random_weights_init
from UNet import double_conv

class UNetPlusPlus(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        K: int = kwargs["kernels"] if "kernels" in kwargs else 32  # Width of the first level
        widths: list[int] = [K, K * 2, K * 4, K * 8, K * 16]  # Same levels as UNet.py
        self.depth: int = len(widths) - 1

        self.pool = nn.MaxPool2d(2)
        self.nodes = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, w in enumerate(widths):
            n: int = len(widths) - i  # Nodes on level i
            self.nodes.append(nn.ModuleList([double_conv(in_dim if i == 0 else widths[i - 1], w)]
                                            + [double_conv((j + 1) * w, w) for j in range(1, n)]))
            self.ups.append(nn.ModuleList(nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
                                          for _ in range(1, n)))

        self.final = nn.Conv2d(K, out_dim, kernel_size=1)

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input: Tensor) -> Tensor:
        X: list[list[Tensor]] = [[self.nodes[0][0](input)]]
        for i in range(1, self.depth + 1):
            X.append([self.nodes[i][0](self.pool(X[i - 1][0]))])

        for j in range(1, self.depth + 1):
            for i in range(self.depth + 1 - j):
                up = self.ups[i][j - 1](X[i + 1][j - 1])
                X[i].append(self.nodes[i][j](torch.cat((*X[i], up), dim=1)))

        return self.final(X[0][-1])

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)
