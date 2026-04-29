"""
2D Mask grounding head for shoulder MRI.

MaskHead2D takes disease-aware local tokens [B, 7, C] and produces per-disease
2D segmentation probability maps [B, 7, H, W] on the key-slice plane.

Architecture:
    local_tokens [B, 7, C]
        → per-disease MLP → spatial seed [B, 7, S*S]     (S = 7, i.e. 7×7 grid)
        → reshape  → [B, 7, S, S]
        → bilinear upsample → [B, 7, H, W]               (H = W = 56 default)
        → sigmoid → probability map

Loss:
    L_mask = BCE(pred_mask, gt_mask_2d, weight=valid_mask)

where gt_mask_2d [B, 7, H, W] is the ground-truth 2D binary mask on the key-slice,
and valid_mask [B, 7] indicates which diseases have a gt mask.

Note: MaskHead2D is independent of ROIBoxHead2D and can be used in parallel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead2D(nn.Module):
    """Per-disease 2D mask prediction head.

    Args:
        feat_dim:    C (local token feature dimension)
        seed_size:   S (spatial grid size before upsample; default 7 → 7×7 = 49 pixels)
        output_size: (H, W) final mask resolution (default 56×56)
        dropout:     dropout rate in MLP
    """

    def __init__(self, feat_dim, seed_size=7, output_size=(56, 56), dropout=0.1):
        super().__init__()
        self.seed_size = seed_size
        self.output_size = output_size
        num_diseases = 7

        # Shared MLP: C → C → S*S (separate per disease via grouped structure)
        # We use a single MLP applied to each token independently
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, seed_size * seed_size),
        )

    def forward(self, local_tokens):
        """
        Args:
            local_tokens: [B, 7, C]

        Returns:
            mask_logits:  [B, 7, H, W] before sigmoid
            mask_probs:   [B, 7, H, W] after sigmoid
        """
        B, N, C = local_tokens.shape
        S = self.seed_size
        H, W = self.output_size

        # [B, 7, S*S]
        seed = self.mlp(local_tokens)

        # [B, 7, S, S]
        seed = seed.view(B, N, S, S)

        # Upsample to output_size: treat diseases as channels for F.interpolate
        # [B, 7, S, S] -> [B, 7, H, W]
        mask_logits = F.interpolate(seed, size=(H, W), mode="bilinear",
                                    align_corners=False)

        mask_probs = torch.sigmoid(mask_logits)
        return mask_logits, mask_probs


def compute_mask_loss(mask_logits, gt_mask_2d, valid_mask):
    """Compute 2D mask BCE loss.

    Args:
        mask_logits: [B, 7, H, W] raw logits from MaskHead2D
        gt_mask_2d:  [B, 7, H, W] ground-truth binary masks (float 0/1)
        valid_mask:  [B, 7] float, 1.0 where disease has gt mask

    Returns:
        loss:      scalar Tensor
        loss_dict: dict with 'mask_bce', 'n_valid'
    """
    B, N, H, W = mask_logits.shape

    # Resize gt to match mask_logits resolution if needed
    if gt_mask_2d.shape[-2:] != (H, W):
        gt_mask_2d = F.interpolate(
            gt_mask_2d.float(), size=(H, W), mode="nearest")

    # Per-pixel BCE: [B, 7, H, W]
    bce = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_2d.float(), reduction="none")

    # Average over spatial dims: [B, 7]
    bce_per_disease = bce.mean(dim=(-2, -1))

    # Mask out diseases without gt: weighted sum
    valid = valid_mask.float()  # [B, 7]
    n_valid = valid.sum().clamp(min=1)
    loss = (bce_per_disease * valid).sum() / n_valid

    return loss, {"mask_bce": loss.item(), "n_valid": int(n_valid.item())}
