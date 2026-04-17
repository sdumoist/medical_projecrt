"""
3D Encoder models for multi-sequence shoulder MRI.
DenseNet3D (G1) and ResNet3D (G2).

Key change: encoders output spatial feature maps, NOT pooled vectors.
- forward()  -> [B, C, D', H', W']  (full 5-D feature map)
- forward_slice() -> [B, D', C]     (pool H'W', keep depth for CoPlaneAttention)
- forward_pool()  -> [B, C]         (global pool, for CrossModalAttention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=3, dilation=dilation, stride=stride,
        padding=dilation, bias=False)


# ==============================================================================
# DenseNet3D (G1)
# ==============================================================================

class _DenseBlock(nn.Module):
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
    def __init__(self, num_input_features, growth_rate, bn_size=4):
        super().__init__()
        self.dense_block = _DenseBlock(num_input_features, growth_rate, bn_size)

    def forward(self, x):
        return self.dense_block(x)


class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, in_channels=1):
        super().__init__()
        self.features = nn.Sequential()

        self.features.add_module('conv0', nn.Conv3d(
            in_channels, num_init_features,
            kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm3d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_features, num_layers, growth_rate, bn_size)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = self._make_transition(num_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        self.num_features = num_features

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
        """Return full feature map: [B, C, D', H', W']."""
        return self.features(x)

    def forward_slice(self, x):
        """Return per-slice features: [B, D', C]."""
        feat = self.forward(x)                      # [B, C, D', H', W']
        feat = F.adaptive_avg_pool3d(feat, (feat.size(2), 1, 1))  # [B, C, D', 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)          # [B, C, D']
        return feat.permute(0, 2, 1)                 # [B, D', C]

    def forward_pool(self, x):
        """Return globally pooled feature: [B, C]."""
        feat = self.forward(x)                       # [B, C, D', H', W']
        feat = F.adaptive_avg_pool3d(feat, 1)        # [B, C, 1, 1, 1]
        return feat.view(feat.size(0), -1)           # [B, C]


# ==============================================================================
# ResNet3D (G2) - Based on CoPAS-main
# ==============================================================================

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck3D(nn.Module):
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

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                               stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
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
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Return full feature map: [B, C, D', H', W']."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_slice(self, x):
        """Return per-slice features: [B, D', C]."""
        feat = self.forward(x)                       # [B, C, D', H', W']
        feat = F.adaptive_avg_pool3d(feat, (feat.size(2), 1, 1))  # [B, C, D', 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)           # [B, C, D']
        return feat.permute(0, 2, 1)                  # [B, D', C]

    def forward_pool(self, x):
        """Return globally pooled feature: [B, C]."""
        feat = self.forward(x)                        # [B, C, D', H', W']
        feat = F.adaptive_avg_pool3d(feat, 1)         # [B, C, 1, 1, 1]
        return feat.view(feat.size(0), -1)            # [B, C]


# ==============================================================================
# Factory functions
# ==============================================================================

def densenet3d_121(in_channels=1):
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16),
                      num_init_features=64, in_channels=in_channels)

def densenet3d_169(in_channels=1):
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 32, 32),
                      num_init_features=64, in_channels=in_channels)

def densenet3d_201(in_channels=1):
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 48, 32),
                      num_init_features=64, in_channels=in_channels)

def resnet3d_18(in_channels=1):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels=in_channels)

def resnet3d_50(in_channels=1):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels=in_channels)

def resnet3d_101(in_channels=1):
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3], in_channels=in_channels)

def resnet3d_152(in_channels=1):
    return ResNet3D(Bottleneck3D, [3, 8, 36, 3], in_channels=in_channels)


def get_encoder(name, in_channels=1):
    """Get encoder by name. Returns an encoder with forward/forward_slice/forward_pool."""
    name = name.lower()
    if "densenet201" in name:
        return densenet3d_201(in_channels)
    elif "densenet169" in name:
        return densenet3d_169(in_channels)
    elif "densenet121" in name or "densenet" in name:
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
