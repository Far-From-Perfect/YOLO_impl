import torch
import torch.nn as nn


config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class _CNNBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs))
        self.add_module('bn1', nn.BatchNorm2d(out_channels))
        self.add_module('leaky1', nn.LeakyReLU(0.1))
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn_act = bn_act

    def forward(self, x):
        if self.bn_act:
            return super(_CNNBlock, self).forward(x)
        else:
            conv = self.get_submodule('conv1')
            return conv(x)


class _ResidualBlock(nn.Sequential):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        for repeat in range(num_repeats):
            layer1 = _CNNBlock(channels, channels // 2, kernel_size=1)
            layer2 = _CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
            self.add_module(f'ResidualBlockLayer1_{repeat+1}', layer1)
            self.add_module(f'ResidualBlockLayer2_{repeat+1}', layer2)

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        if self.use_residual:
            x = x + super(_ResidualBlock, self).forward(x)
        else:
            x = super(_ResidualBlock, self).forward(x)

        return x


class _ScalePrediction(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(_ScalePrediction, self).__init__()
        self.add_module(f'CNNBlock1', _CNNBlock(in_channels, in_channels * 2, kernel_size=3, padding=1))
        self.add_module(f'CNNBlock2', _CNNBlock(in_channels * 2, (num_classes + 5) * 3, bn_act=False, kernel_size=1))
        self.num_classes = num_classes

    def forward(self, x):
        return super(_ScalePrediction, self).forward(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)


class YOLO_v3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(YOLO_v3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_layers()
        # print(self.layers)

    def forward(self, x):
        outputs = []
        route_connection = []
        for layer in self.layers:
            if isinstance(layer, _ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, _ResidualBlock) and layer.num_repeats == 8:
                route_connection.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connection[-1]], dim=1)
                route_connection.pop()

        return outputs

    def _create_layers(self):
        in_channels = self.in_channels
        layers = nn.Sequential()

        for idx, elem in enumerate(config):
            if isinstance(elem, tuple):
                out_channels, kernel_size, stride = elem
                layers.add_module(
                    f'CNNBlock_{idx+1}',
                    _CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels

            elif isinstance(elem, list):
                layers.add_module(
                    f'ResidualBlock_{idx+1}',
                    _ResidualBlock(in_channels, num_repeats=elem[1])
                )

            elif isinstance(elem, str):
                if elem == 'S':
                    layers.add_module(
                        f'ResidualBlock_{idx+1}',
                        _ResidualBlock(in_channels, use_residual=False, num_repeats=1)
                    )
                    layers.add_module(
                        f'CNNBlock_{idx+1}',
                        _CNNBlock(in_channels, in_channels // 2, kernel_size=1)
                    )
                    layers.add_module(
                        f'Scale_Prediction_{idx+1}',
                        _ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    )
                    in_channels = in_channels // 2

                elif elem == 'U':
                    layers.add_module(f'Upsample_{idx+1}', nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers
