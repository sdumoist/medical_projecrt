"""
Loss functions with mask support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy loss with mask support."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: [B, num_diseases]
            targets: [B, num_diseases]
            mask: [B, num_diseases] - 1 for valid, 0 for masked
        """
        # Compute BCE (targets must be float)
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if mask is not None:
            # Apply mask
            loss = loss * mask
            # Count valid samples
            valid_count = mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = torch.tensor(0.0, device=logits.device)
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()

        return loss


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with mask support for multi-class."""

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_diseases, num_classes]
            targets: [B, num_diseases]
            mask: [B, num_diseases] - 1 for valid, 0 for masked
        """
        B, num_diseases, num_classes = logits.shape

        # Reshape for cross entropy
        logits = logits.view(-1, num_classes)
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)

        # Reshape back
        loss = loss.view(B, num_diseases)

        if mask is not None:
            # Apply mask
            loss = loss * mask
            valid_count = mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = torch.tensor(0.0, device=logits.device)
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()

        return loss


class FocalLoss(nn.Module):
    """Focal loss for binary classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()