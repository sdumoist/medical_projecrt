"""
ROI grounding heads for shoulder MRI.

ROIBoxHead2D:
    Input:  local_tokens [B, 7, C]
    Output: roi_box_2d   [B, 7, 4]   (x1, y1, x2, y2) normalized [0,1]
            roi_box_conf [B, 7]       confidence logit (before sigmoid)

Box parameterization uses center form (cx, cy, w, h) → sigmoid → convert to corner form,
which is more stable than directly regressing corners.
"""
import torch
import torch.nn as nn


class ROIBoxHead2D(nn.Module):
    """Predict 2D bounding boxes on the key-slice plane from local_tokens.

    Architecture:
        Linear(C, C) → GELU → Dropout → Linear(C, 5)
            output[..., :4]  = raw box params (cx, cy, w, h) → sigmoid
            output[..., 4]   = raw confidence logit

    Box conversion (center form → corner form):
        cx, cy, w, h = sigmoid(raw_box)   # all in [0, 1]
        x1 = cx - w/2,  y1 = cy - h/2
        x2 = cx + w/2,  y2 = cy + h/2
        clamp to [0, 1]
    """

    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 5),   # 4 box params + 1 confidence
        )

    def forward(self, local_tokens):
        """
        Args:
            local_tokens: [B, 7, C]
        Returns:
            roi_box_2d:  [B, 7, 4]  corner form (x1, y1, x2, y2), values in [0, 1]
            roi_box_conf:[B, 7]     raw confidence logit (apply sigmoid for prob)
        """
        raw = self.head(local_tokens)         # [B, 7, 5]
        raw_box  = raw[..., :4]               # [B, 7, 4]
        raw_conf = raw[..., 4]                # [B, 7]

        # Convert center form to corner form
        cx_cy_wh = torch.sigmoid(raw_box)     # [B, 7, 4] all in (0, 1)
        cx = cx_cy_wh[..., 0]
        cy = cx_cy_wh[..., 1]
        w  = cx_cy_wh[..., 2]
        h  = cx_cy_wh[..., 3]

        x1 = (cx - w / 2).clamp(0.0, 1.0)
        y1 = (cy - h / 2).clamp(0.0, 1.0)
        x2 = (cx + w / 2).clamp(0.0, 1.0)
        y2 = (cy + h / 2).clamp(0.0, 1.0)

        roi_box_2d = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, 7, 4]
        return roi_box_2d, raw_conf


def compute_roi_loss(roi_box_2d, roi_box_conf, roi_box_gt, roi_box_valid):
    """Compute ROI regression + confidence loss.

    Args:
        roi_box_2d:    [B, 7, 4]  predicted corner boxes (normalized [0,1])
        roi_box_conf:  [B, 7]     raw confidence logits
        roi_box_gt:    [B, 7, 4]  ground-truth corner boxes
        roi_box_valid: [B, 7]     1.0 where gt box exists, 0.0 otherwise

    Returns:
        loss:     scalar
        loss_dict: {"box_reg": float, "box_conf": float}
    """
    import torch.nn.functional as F

    # Regression loss: SmoothL1 only on valid positions
    valid_mask = roi_box_valid.bool()  # [B, 7]

    if valid_mask.any():
        pred_valid = roi_box_2d[valid_mask]    # [N, 4]
        gt_valid   = roi_box_gt.to(roi_box_2d.device)[valid_mask]   # [N, 4]
        box_reg_loss = F.smooth_l1_loss(pred_valid, gt_valid, beta=0.1)
    else:
        box_reg_loss = roi_box_2d.sum() * 0.0  # differentiable zero

    # Confidence loss: BCE on all positions
    conf_target = roi_box_valid.to(roi_box_conf.device)   # [B, 7]
    box_conf_loss = F.binary_cross_entropy_with_logits(roi_box_conf, conf_target)

    loss = box_reg_loss + 0.5 * box_conf_loss
    return loss, {
        "box_reg": box_reg_loss.item(),
        "box_conf": box_conf_loss.item(),
    }
