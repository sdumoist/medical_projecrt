"""
CoPAS-style fusion module for multi-sequence MRI.
Based on CoPAS-main: Co_Plane_Att + Cross_Modal_Att.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ini_weights(module_list):
    """Initialize weights."""
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CoPlaneAttention(nn.Module):
    """Co-Plane Attention: attention across MRI planes.

    For multi-sequence input, we treat each sequence as a 'plane' for attention.
    Main sequence attends to other sequences.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Q, K, V for main sequence
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv = nn.Linear(embed_dim, embed_dim, bias=False)

        # Q, K, V for co sequences (average over other sequences)
        self.co_mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.co_mk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.co_mv = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm = nn.LayerNorm(embed_dim)

        ini_weights([self.mq, self.mk, self.mv, self.co_mq, self.co_mk, self.co_mv])

    def forward(self, main_f, co_f):
        """Forward pass.

        Args:
            main_f: [B, embed_dim] - main sequence features
            co_f: [B, embed_dim] - co-sequence features (e.g., from other MRI planes)

        Returns:
            fused: [B, embed_dim]
        """
        B = main_f.shape[0]

        # Expand for batch processing
        res = main_f

        # Q from main
        q = self.mq(main_f)  # [B, embed_dim]
        q = q.unsqueeze(1)  # [B, 1, embed_dim]

        # K, V from co
        k = self.co_mk(co_f).unsqueeze(1)  # [B, 1, embed_dim]
        v = self.co_mv(co_f).unsqueeze(1)  # [B, 1, embed_dim]

        # Attention
        att = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.embed_dim)
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, v)  # [B, 1, embed_dim]
        out = out.squeeze(1)  # [B, embed_dim]

        # Residual + norm
        f = self.norm(0.5 * out + res)

        return f


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention: combine features from different modalities."""

    def __init__(self, feature_channel):
        super().__init__()
        self.feature_channel = feature_channel

        # Transform matrix: concatenate then project
        self.transform_matrix = nn.Linear(2 * feature_channel, feature_channel)
        self.norm = nn.BatchNorm1d(feature_channel)

        ini_weights([self.transform_matrix, self.norm])

    def forward(self, pdw_f, aux_f):
        """Forward pass.

        Args:
            pdw_f: [B, feature_channel] - primary sequence features
            aux_f: [B, feature_channel] - auxiliary sequence features (e.g., T1/T2)

        Returns:
            fused: [B, feature_channel]
        """
        # Add
        add_f = pdw_f + aux_f

        # Concatenate
        cat_f = torch.cat([pdw_f, aux_f], dim=1)  # [B, 2*feature_channel]

        # Transform
        att_f = self.transform_matrix(cat_f)
        att_f = self.norm(att_f)
        att_f = F.relu(att_f)

        # Softmax weights
        att_f = F.softmax(att_f, dim=-1)

        # Weighted sum
        f = add_f * att_f

        return f


class MultiSeqFusion(nn.Module):
    """Multi-sequence fusion using CoPAS-style attention."""

    def __init__(
        self,
        num_sequences,
        feature_dim,
        use_co_att=True,
        use_cross_modal=True,
        dropout=0.3
    ):
        super().__init__()

        self.num_sequences = num_sequences
        self.use_co_att = use_co_att
        self.use_cross_modal = use_cross_modal

        # Co-plane attention for each sequence
        if use_co_att:
            self.co_attentions = nn.ModuleList([
                CoPlaneAttention(feature_dim)
                for _ in range(num_sequences)
            ])

        # Cross-modal attention (between sequences)
        if use_cross_modal:
            self.cross_attentions = nn.ModuleList([
                CrossModalAttention(feature_dim)
                for _ in range(num_sequences)
            ])

        # Fusion FC
        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, features):
        """Fusion.

        Args:
            features: [B, num_seq, feature_dim]

        Returns:
            fused: [B, feature_dim // 2]
        """
        B, num_seq, feature_dim = features.shape

        if num_seq == 1:
            # Single sequence no fusion needed
            fused = features.squeeze(1)
        elif self.use_co_att and num_seq > 1:
            # Use first sequence as main, average of rest as co
            main_f = features[:, 0]  # [B, feature_dim]
            co_f = features[:, 1:].mean(dim=1)  # [B, feature_dim]

            fused = self.co_attentions[0](main_f, co_f)
        else:
            # Simple average (fallback)
            fused = features.mean(dim=1)

        # Final FC
        fused = self.fusion_fc(fused)

        return fused


class SimpleFusion(nn.Module):
    """Simple concatenation + FC fusion (fallback)."""

    def __init__(self, num_sequences, feature_dim, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(feature_dim * num_sequences, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, features):
        """Concatenate then FC.

        Args:
            features: [B, num_seq, feature_dim]

        Returns:
            fused: [B, hidden_dim]
        """
        B, num_seq, feature_dim = features.shape
        fused = features.view(B, -1)  # [B, num_seq * feature_dim]
        fused = self.fc(fused)
        return fused


# Aliases for backward compatibility
CoPASFusion = MultiSeqFusion
AttentionFusion = SimpleFusion
ConcatenateFusion = SimpleFusion