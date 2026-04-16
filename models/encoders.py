"""
3D Encoder models for multi-sequence MRI.
DenseNet3D (G1) and ResNet3D (G2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride):
    """Downsample for basic block."""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        x.size(0), planes - out.size(1), out.size(2),
        out.size(3), out.size(4), dtype=out.dtype, device=out.device)
    out = torch.cat([out.data, zero_pads], dim=1)
    return out


# ==============================================================================
# DenseNet3D (G1)
# ==============================================================================

class _DenseBlock(nn.Module):
    """Dense block for DenseNet3D."""

    def __init__(self, in_planes, growth_rate, bn_size=4):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], 1)


class _DenseLayer(nn.Module):
    """Single dense layer."""
    def __init__(self, num_input_features, growth_rate, bn_size=4):
        super().__init__()
        self.dense_block = _DenseBlock(num_input_features, growth_rate, bn_size)

    def forward(self, x):
        return self.dense_block(x)


class DenseNet3D(nn.Module):
    """DenseNet3D encoder."""

    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        num_classes=2,
        in_channels=1
    ):
        super().__init__()

        self.features = nn.Sequential()

        # Initial convolution
        self.features.add_module('conv0', nn.Conv3d(
            in_channels, num_init_features,
            kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm3d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(
                num_features, num_layers, growth_rate, bn_size)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = self._make_transition(num_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        self.num_features = num_features

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_dense_block(self, in_planes, num_layers, growth_rate, bn_size):
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_planes + i * growth_rate, growth_rate, bn_size))
        return nn.Sequential(*layers)

    def _make_transition(self, num_features):
        return nn.Sequential(
            nn.BatchNorm3d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_features, num_features // 2, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2))

    def forward(self, x):
        features = self.features(x)
        return features


# ==============================================================================
# ResNet3D (G2) - Based on CoPAS-main
# ==============================================================================

class BasicBlock3D(nn.Module):
    """Basic block for ResNet3D."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """Bottleneck block for ResNet3D."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """ResNet3D encoder - from CoPAS-main."""

    def __init__(
        self,
        block,
        layers,
        in_channels=1,
        num_classes=2
    ):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            in_channels, 64,
            kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.num_features = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def densenet3d_121(in_channels=1):
    """DenseNet121-3D."""
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        in_channels=in_channels
    )


def resnet3d_18(in_channels=1):
    """ResNet18-3D."""
    return ResNet3D(
        BasicBlock3D, [2, 2, 2, 2],
        in_channels=in_channels
    )


def resnet3d_50(in_channels=1):
    """ResNet50-3D."""
    return ResNet3D(
        Bottleneck3D, [3, 4, 6, 3],
        in_channels=in_channels
    )


def resnet3d_101(in_channels=1):
    """ResNet101-3D."""
    return ResNet3D(
        Bottleneck3D, [3, 4, 23, 3],
        in_channels=in_channels
    )


def resnet3d_152(in_channels=1):
    """ResNet152-3D."""
    return ResNet3D(
        Bottleneck3D, [3, 8, 36, 3],
        in_channels=in_channels
    )


def get_encoder(name, in_channels=1):
    """Get encoder by name."""
    name = name.lower()
    if "densenet121" in name or "densenet" in name:
        return densenet3d_121(in_channels)
    elif "resnet152" in name:
        return resnet3d_152(in_channels)
    elif "resnet101" in name:
        return resnet3d_101(in_channels)
    elif "resnet50" in name:
        return resnet3d_50(in_channels)
    elif "resnet18" in name or "resnet3d" in name:
        return resnet3d_18(in_channels)
    else:
        raise ValueError("Unknown encoder: %s" % name)


class MultiSeqEncoder3D(nn.Module):
    """Multi-sequence 3D encoder with per-sequence encoders."""

    def __init__(
        self,
        num_sequences,
        encoder_name="resnet3d_18",
        in_channels=1,
        pretrained=False
    ):
        super().__init__()

        self.num_sequences = num_sequences
        self.encoder_name = encoder_name

        # Create per-sequence encoders
        self.encoders = nn.ModuleList([
            get_encoder(encoder_name, in_channels)
            for _ in range(num_sequences)
        ])

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 32, 64, 64)
            out = self.encoders[0](dummy)
            if out.dim() > 2:
                out = F.adaptive_avg_pool3d(out, 1)
            self.feature_dim = out.view(1, -1).shape[1]

    def forward(self, x):
        """Forward pass.

        Args:
            x: [B, num_seq, C, D, H, W]

        Returns:
            features: [B, num_seq, feature_dim]
        """
        B, num_seq = x.shape[:2]

        # Process each sequence
        features = []
        for i in range(num_seq):
            seq_input = x[:, i]  # [B, C, D, H, W]
            feat = self.encoders[i](seq_input)  # [B, C', D', H', W']
            # Global pooling
            feat = F.adaptive_avg_pool3d(feat, 1)
            feat = feat.view(B, -1)  # [B, feature_dim]
            features.append(feat)

        features = torch.stack(features, dim=1)  # [B, num_seq, feature_dim]

        return features