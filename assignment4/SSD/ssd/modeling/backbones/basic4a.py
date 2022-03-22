from collections import OrderedDict

import torch
from typing import Tuple, List
from torch import nn

class BasicBlock(nn.Sequential):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, out_stride: int = 2, out_padding=1):
        super(BasicBlock, self).__init__(OrderedDict([
            ('relu0', nn.ReLU()),
            ('conv1', nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=out_stride, padding=out_padding)),
            ('relu2', nn.ReLU())
        ]))


class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.features = nn.Sequential(OrderedDict([
            ('features1', nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1), nn.ReLU()
            )),
            ('features2', BasicBlock(output_channels[0], 128, output_channels[1])),
            ('features3', BasicBlock(output_channels[1], 256, output_channels[2])),
            ('features4', BasicBlock(output_channels[2], 128, output_channels[3])),
            ('features5', BasicBlock(output_channels[3], 128, output_channels[4])),
            ('features6', BasicBlock(output_channels[4], 128, output_channels[5], out_stride=1, out_padding=0)),
        ]))

    def forward(self, x) -> Tuple[torch.Tensor, ...]:
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features: List[torch.Tensor] = []
        for feat_layer in self.features:
            x = feat_layer(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

