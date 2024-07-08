from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "A_LRN": [64, "L", "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "C",
        "M",
        512,
        512,
        "C",
        "M",
        512,
        512,
        "C",
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes=2,
        dropout: float = 0.5,
        init_weights: bool = True,
    ):
        super(VGG, self).__init__()
        self.features = features
        self.avepool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avepool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# def make_layers(cfg: List[Union[str, int]]):
#     layers = []
#     in_channels = 3

#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'C':
#             conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#             layers += [conv2d, nn.ReLU(True)]
#         elif v == 'L':
#             layers += [nn.LocalResponseNorm(5, k=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             layers += [conv2d, nn.ReLU(True)]
#             in_channels = v

#     return nn.Sequential(*layers)


# def vgg_a(num_classes: int=1000, dropout: float=0.5) -> VGG:
#     features = make_layers(cfgs['A'])
#     return VGG(features, num_classes, dropout)


# def vgg_a_lrn(num_classes: int=1000, dropout: float=0.5) -> VGG:
#     features = make_layers(cfgs['A_LRN'])
#     return VGG(features, num_classes, dropout)


# def vgg_b(num_classes: int=1000, dropout: float=0.5) -> VGG:
#     features = make_layers(cfgs['B'])
#     return VGG(features, num_classes, dropout)


# def vgg_c(num_classes: int=1000, dropout: float=0.5) -> VGG:
#     features = make_layers(cfgs['C'])
#     return VGG(features, num_classes, dropout)


# def vgg_d(num_classes: int=1000, dropout: float=0.5) -> VGG:
#     features = make_layers(cfgs['D'])
#     return VGG(features, num_classes, dropout)


# def vgg_e(num_classes: int=1000, dropout: float=0.5) -> VGG:
#     features = make_layers(cfgs['E'])
#     return VGG(features, num_classes, dropout)


# def get_vgg_model(model_name: str='vgg_a', num_classes:int =1000,
#                   dropout: float=0.5) -> VGG:
#     if model_name == 'vgg_a':
#         return vgg_a(num_classes, dropout)
#     elif model_name == 'vgg_a_lrn':
#         return vgg_a_lrn(num_classes, dropout)
#     elif model_name == 'vgg_b':
#         return vgg_b(num_classes, dropout)
#     elif model_name == 'vgg_c':
#         return vgg_c(num_classes, dropout)
#     elif model_name == 'vgg_d':
#         return vgg_d(num_classes, dropout)
#     elif model_name == 'vgg_e':
#         return vgg_e(num_classes, dropout)
