"""
CoPAS-style fusion for shoulder MRI.

CoPlaneAttention:
    Input:  main_f [B, D, C], co_f1 [B, D, C], co_f2 [B, D, C]
    Output: [B, C]
    Main sequence attends to two co-plane sequences along the depth (slice) axis,
    then aggregates into a single global vector.

CrossModalAttention:
    Input:  pdw_f [B, C], aux_f [B, C]
    Output: [B, C]
    Fuses a PD-weighted feature with an auxiliary modality (T1/T2) feature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class CoPlaneAttention(nn.Module):
    """Co-Plane Attention across MRI planes (slice-level).

    Given a main sequence and two co-plane sequences (all as per-slice features),
    performs cross-attention where main queries attend to co-plane keys/values,
    then pools the attended slices into a global vector.

    Args:
        embed_dim: feature channel dimension C
        num_heads: number of attention heads (default 4)
    """

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        # Q from main, K/V from co-plane
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.scale = math.sqrt(self.head_dim)

        self.apply(_init_weights)

    def forward(self, main_f, co_f1, co_f2):
        """
        Args:
            main_f: [B, D_m, C] - main sequence per-slice features
            co_f1:  [B, D_1, C] - co-plane sequence 1 per-slice features
            co_f2:  [B, D_2, C] - co-plane sequence 2 per-slice features

        Returns:
            out: [B, C] - globally pooled attended feature
        """
        B, D_m, C = main_f.shape

        # Concatenate co-plane KV along slice dim
        co_f = torch.cat([co_f1, co_f2], dim=1)  # [B, D_1+D_2, C]
        D_co = co_f.size(1)

        # Project Q, K, V
        q = self.q_proj(main_f)   # [B, D_m, C]
        k = self.k_proj(co_f)     # [B, D_co, C]
        v = self.v_proj(co_f)     # [B, D_co, C]

        # Multi-head reshape: [B, num_heads, D, head_dim]
        q = q.view(B, D_m, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, D_co, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, D_co, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: [B, num_heads, D_m, D_co]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: [B, num_heads, D_m, head_dim]
        attended = torch.matmul(attn, v)

        # Merge heads: [B, D_m, C]
        attended = attended.permute(0, 2, 1, 3).contiguous().view(B, D_m, C)
        attended = self.out_proj(attended)

        # Residual + norm
        fused = self.norm(attended + main_f)  # [B, D_m, C]

        # Pool over slices -> [B, C]
        out = fused.mean(dim=1)
        return out


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention: fuse PD feature with auxiliary modality (T1/T2).

    Takes two globally pooled vectors and produces a fused vector using
    learned attention weights.

    Args:
        feature_dim: channel dimension C
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        self.gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Sigmoid()
        )

        self.apply(_init_weights)

    def forward(self, pdw_f, aux_f):
        """
        Args:
            pdw_f: [B, C] - PD-weighted sequence feature (globally pooled)
            aux_f: [B, C] - auxiliary modality feature (T1/T2, globally pooled)

        Returns:
            fused: [B, C]
        """
        cat_f = torch.cat([pdw_f, aux_f], dim=1)  # [B, 2C]
        g = self.gate(cat_f)                        # [B, C], values in (0, 1)

        # Gated fusion: blend PD and auxiliary
        fused = g * pdw_f + (1 - g) * aux_f         # [B, C]
        return fused
