"""
Localizer branch for key-slice / ROI guided training.
Uses nnUNet masks to guide attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KeySliceExtractor(nn.Module):
    """Extract key slice from 3D volume based on mask."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Key slice attention
        self.key_slice_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, features, key_slice_indices):
        """Extract features at key slice positions.

        Args:
            features: [B, num_seq, D, feature_dim] - per-slice features
            key_slice_indices: [B, num_seq] - key slice indices

        Returns:
            key_features: [B, num_seq, feature_dim]
        """
        B, num_seq, D, feature_dim = features.shape
        key_features = []

        for b in range(B):
            seq_features = []
            for s in range(num_seq):
                idx = key_slice_indices[b, s]
                if idx >= 0 and idx < D:
                    seq_features.append(features[b, s, idx])
                else:
                    seq_features.append(features[b, s].mean(0))
            key_features.append(torch.stack(seq_features))

        key_features = torch.stack(key_features)
        return key_features


class ROIExtractor(nn.Module):
    """Extract ROI from 3D volume based on bounding box."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # ROI FC
        self.roi_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )

    def forward(self, global_features, roi_features):
        """Combine global and ROI features.

        Args:
            global_features: [B, embed_dim]
            roi_features: [B, embed_dim]

        Returns:
            fused: [B, embed_dim]
        """
        # Concatenate global + ROI
        fused = torch.cat([global_features, roi_features], dim=1)
        fused = self.roi_fc(fused)
        return fused


class LocalizerBranch(nn.Module):
    """Localizer branch using key-slice / ROI from nnUNet masks."""

    def __init__(
        self,
        feature_dim,
        use_key_slice=True,
        use_roi=True
    ):
        super().__init__()

        self.use_key_slice = use_key_slice
        self.use_roi = use_roi

        # Key slice branch
        if use_key_slice:
            self.key_slice_extractor = KeySliceExtractor(feature_dim)

        # ROI branch
        if use_roi:
            self.roi_extractor = ROIExtractor(feature_dim)

            # Local feature encoder
            self.local_encoder = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

    def forward(self, features, key_slices=None, roi_boxes=None):
        """Forward with localizer guidance.

        Args:
            features: [B, num_seq, feature_dim] - encoder features
            key_slices: [B, num_seq] - key slice indices from mask
            roi_boxes: [B, 6] - bounding boxes (d_min, h_min, w_min, d_max, h_max, w_max)

        Returns:
            local_features: [B, feature_dim]
        """
        B, num_seq, feature_dim = features.shape

        if self.use_key_slice and key_slices is not None:
            # Extract at key slices
            key_features = self.key_slice_extractor(features, key_slices)
            # Global pooling over sequences
            local_features = key_features.mean(dim=1)
        else:
            # Simple average
            local_features = features.mean(dim=1)

        if self.use_roi and roi_boxes is not None:
            # ROI features would come from separate ROI encoder
            # For now, just use global features
            pass

        # Encode local features
        local_features = self.local_encoder(local_features)

        return local_features


def create_localizer(config):
    """Create localizer from config."""
    return LocalizerBranch(
        feature_dim=config.get("feature_dim", 512),
        use_key_slice=config.get("use_key_slice", True),
        use_roi=config.get("use_roi", True)
    )