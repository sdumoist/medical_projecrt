"""
3D Encoder models for multi-sequence shoulder MRI.
DenseNet3D (G1), ResNet3D (G2), and SwinTransformer3D (G3).

Key change: encoders output spatial feature maps, NOT pooled vectors.
- forward()  -> [B, C, D', H', W']  (full 5-D feature map)
- forward_slice() -> [B, D', C]     (pool H'W', keep depth for CoPlaneAttention)
- forward_pool()  -> [B, C]         (global pool, for CrossModalAttention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul


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
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
# Swin Transformer 3D (G3)
# ==============================================================================

def window_partition_3d(x, window_size):
    """Partition tensor into non-overlapping 3D windows.
    Args:
        x: [B, D, H, W, C]
        window_size: (wd, wh, ww)
    Returns:
        windows: [num_windows*B, wd, wh, ww, C]
    """
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd, wh, ww, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """Reverse window_partition_3d.
    Args:
        windows: [num_windows*B, wd, wh, ww, C]
        window_size: (wd, wh, ww)
        D, H, W: original spatial dims
    Returns:
        x: [B, D, H, W, C]
    """
    wd, wh, ww = window_size
    B = int(windows.shape[0] / (D // wd * H // wh * W // ww))
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """Window-based multi-head self-attention for 3D."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wd, wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # [3, wd, wh, ww]
        coords_flat = torch.flatten(coords, 1)  # [3, wd*wh*ww]
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        """
        Args:
            x: [num_windows*B, N, C]  where N = wd*wh*ww
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """A single Swin Transformer block for 3D data."""

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, D, H, W):
        """
        Args:
            x: [B, D*H*W, C]
            D, H, W: spatial dims
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Pad to multiples of window_size
        wd, wh, ww = self.window_size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        # Cyclic shift
        sd, sh, sw = self.shift_size
        if sd > 0 or sh > 0 or sw > 0:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))

        # Partition windows
        x_windows = window_partition_3d(x, self.window_size)  # [nW*B, wd, wh, ww, C]
        x_windows = x_windows.view(-1, reduce(mul, self.window_size), C)  # [nW*B, N, C]

        # Window attention
        attn_windows = self.attn(x_windows)  # [nW*B, N, C]

        # Merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        x = window_reverse_3d(attn_windows, self.window_size, Dp, Hp, Wp)  # [B, Dp, Hp, Wp, C]

        # Reverse cyclic shift
        if sd > 0 or sh > 0 or sw > 0:
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(1, 2, 3))

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.view(B, D * H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding using convolution."""

    def __init__(self, patch_size=(2, 4, 4), in_channels=1, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            x: [B, D'*H'*W', embed_dim], (D', H', W')
        """
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # [B, D'*H'*W', embed_dim]
        x = self.norm(x)
        return x, D, H, W


class PatchMerging3D(nn.Module):
    """3D Patch merging for downsampling (2x in H,W only; keep D unchanged)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, D, H, W):
        """
        Args:
            x: [B, D*H*W, C]
        Returns:
            x: [B, D*H/2*W/2, 2*C], (D, H//2, W//2)
        """
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)

        # Pad if H or W is odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, D, H/2, W/2, 4*C]

        H, W = H // 2, W // 2
        x = x.view(B, D * H * W, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, D, H, W


class SwinTransformerStage3D(nn.Module):
    """A single Swin Transformer stage with multiple blocks."""

    def __init__(self, dim, depth, num_heads, window_size=(2, 7, 7),
                 mlp_ratio=4.0, drop=0.0, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = tuple(s // 2 for s in window_size) if (i % 2 == 1) else (0, 0, 0)
            self.blocks.append(SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=shift, mlp_ratio=mlp_ratio, drop=drop))
        self.downsample = downsample

    def forward(self, x, D, H, W):
        for blk in self.blocks:
            x = blk(x, D, H, W)
        if self.downsample is not None:
            x, D, H, W = self.downsample(x, D, H, W)
        return x, D, H, W


class SwinTransformer3D(nn.Module):
    """Swin Transformer 3D encoder with same interface as DenseNet3D/ResNet3D.

    Args:
        in_channels: input channels (1 for grayscale MRI)
        embed_dim: base embedding dimension
        depths: number of blocks per stage
        num_heads: attention heads per stage
        window_size: 3D window size (wd, wh, ww)
        patch_size: initial patch embedding size
        mlp_ratio: MLP expansion ratio
        drop_rate: dropout rate
    """

    def __init__(self, in_channels=1, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=(2, 7, 7),
                 patch_size=(2, 4, 4), mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()
        self.num_stages = len(depths)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        # Build stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            dim = embed_dim * (2 ** i)
            downsample = PatchMerging3D(dim) if i < self.num_stages - 1 else None
            stage = SwinTransformerStage3D(
                dim=dim, depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                drop=drop_rate, downsample=downsample)
            self.stages.append(stage)

        self.num_features = embed_dim * (2 ** (self.num_stages - 1))
        self.norm = nn.LayerNorm(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Extract features, return [B, D', H', W', C] before final pool."""
        x, D, H, W = self.patch_embed(x)  # [B, D'*H'*W', C]
        for stage in self.stages:
            x, D, H, W = stage(x, D, H, W)
        x = self.norm(x)  # [B, D'*H'*W', C]
        x = x.view(-1, D, H, W, self.num_features)
        return x  # [B, D', H', W', C]

    def forward(self, x):
        """Return full feature map: [B, C, D', H', W']."""
        feat = self.forward_features(x)  # [B, D', H', W', C]
        return feat.permute(0, 4, 1, 2, 3)  # [B, C, D', H', W']

    def forward_slice(self, x):
        """Return per-slice features: [B, D', C]."""
        feat = self.forward_features(x)  # [B, D', H', W', C]
        feat = feat.mean(dim=(2, 3))  # [B, D', C]  (pool H', W')
        return feat

    def forward_pool(self, x):
        """Return globally pooled feature: [B, C]."""
        feat = self.forward_features(x)  # [B, D', H', W', C]
        feat = feat.mean(dim=(1, 2, 3))  # [B, C]
        return feat


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


def swin3d_tiny(in_channels=1):
    return SwinTransformer3D(
        in_channels=in_channels, embed_dim=96, depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24), window_size=(2, 7, 7), patch_size=(2, 4, 4))

def swin3d_small(in_channels=1):
    return SwinTransformer3D(
        in_channels=in_channels, embed_dim=96, depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24), window_size=(2, 7, 7), patch_size=(2, 4, 4))

def swin3d_base(in_channels=1):
    return SwinTransformer3D(
        in_channels=in_channels, embed_dim=128, depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32), window_size=(2, 7, 7), patch_size=(2, 4, 4))


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
    elif "swin3d_base" in name or "swin_base" in name:
        return swin3d_base(in_channels)
    elif "swin3d_small" in name or "swin_small" in name:
        return swin3d_small(in_channels)
    elif "swin3d_tiny" in name or "swin_tiny" in name or "swin3d" in name or "swin" in name:
        return swin3d_tiny(in_channels)
    else:
        raise ValueError("Unknown encoder: %s" % name)
