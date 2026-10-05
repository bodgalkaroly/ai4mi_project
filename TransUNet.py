"""Compact 2D CNN/ViT segmentation model; inspired by, not a replica of, TransUNet."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    layers = []
    for incoming in (in_channels, out_channels):
        layers.extend([nn.Conv2d(incoming, out_channels, 3, padding=1, bias=False),
                       nn.GroupNorm(8, out_channels), nn.GELU()])
    return nn.Sequential(*layers)


class CompactTransUNet(nn.Module):
    def __init__(self, nin=1, nout=5, **kwargs):
        super().__init__()
        self.encoder1 = conv_block(nin, 32)
        self.encoder2 = conv_block(32, 64)
        self.encoder3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        # One token per position of the feature map at 1/8 input resolution.
        self.position = nn.Parameter(torch.zeros(1, 128, 32, 32))
        block = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            block, num_layers=4, norm=nn.LayerNorm(128), enable_nested_tensor=False)
        self.decoder3 = conv_block(256, 128)
        self.decoder2 = conv_block(192, 64)
        self.decoder1 = conv_block(96, 32)
        self.head = nn.Conv2d(32, nout, 1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        nn.init.trunc_normal_(self.position, std=0.02)

    @staticmethod
    def upsample_join(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat((x, skip), dim=1)

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 8:
            raise ValueError('Expected [batch, channels, height, width] with height/width >= 8')
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        features = self.pool(e3)
        b, c, h, w = features.shape
        position = F.interpolate(self.position, size=(h, w), mode='bilinear', align_corners=False)
        tokens = (features + position).flatten(2).transpose(1, 2)
        x = self.transformer(tokens).transpose(1, 2).reshape(b, c, h, w)
        x = self.decoder3(self.upsample_join(x, e3))
        x = self.decoder2(self.upsample_join(x, e2))
        x = self.decoder1(self.upsample_join(x, e1))
        return self.head(x)
