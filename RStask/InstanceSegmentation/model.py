import torch
import torch.nn as nn
from RStask.InstanceSegmentation.swin import swin
from RStask.InstanceSegmentation.uper import UPerHead

Activation=torch.nn.ReLU
class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)





class SwinUPer(torch.nn.Module):
    def __init__(self, classes: int = 16):
        super(SwinUPer, self).__init__()
        # encoder
        self.encoder = swin(embed_dim=96,depths=[2, 2, 6, 2],num_heads=[3, 6, 12, 24],
                            window_size=7,ape=False,drop_path_rate=0.3,patch_norm=True)
        # decoder
        self.decoder = UPerHead(
            in_channels = self.encoder.out_channels[1:],
            channels = self.encoder.out_channels[2],
            in_index = (0, 1, 2, 3),dropout_ratio = 0.1,
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        )

        self.semseghead = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(self.encoder.out_channels[2], classes, kernel_size=1)
        )

        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.semseghead)
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(*features)
        output = self.semseghead(output)
        return output









