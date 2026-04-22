"""
Prediction heads for shoulder MRI multi-disease classification.

Supports:
- BranchHead: per-branch (sag/cor/axi) binary classification
- FinalHead:  final fusion head combining all branch features
"""
import torch
import torch.nn as nn


class BranchHead(nn.Module):
    """Classification head for a single branch.

    Args:
        input_dim: feature dimension from fusion output
        num_diseases: number of diseases this branch predicts
        dropout: dropout rate
        num_classes: 2 for binary (output [B, num_diseases]),
                     3 for ternary (output [B, num_diseases, num_classes])
    """

    def __init__(self, input_dim, num_diseases, dropout=0.3, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        out_dim = num_diseases if num_classes == 2 else num_diseases * num_classes
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, out_dim)
        )
        self.num_diseases = num_diseases

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            binary:  logits [B, num_diseases]
            ternary: logits [B, num_diseases, num_classes]
        """
        out = self.fc(x)
        if self.num_classes > 2:
            B = out.shape[0]
            out = out.view(B, self.num_diseases, self.num_classes)
        return out


class FinalHead(nn.Module):
    """Final head that combines all branch features for joint prediction.

    Concatenates branch features and predicts all 7 diseases.

    Args:
        branch_dim: feature dimension per branch
        num_branches: number of branches (3: sag, cor, axi)
        num_diseases: total number of diseases (7)
        dropout: dropout rate
        num_classes: 2 for binary, 3 for ternary
    """

    def __init__(self, branch_dim, num_branches, num_diseases, dropout=0.3,
                 num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_diseases = num_diseases
        total_dim = branch_dim * num_branches
        out_dim = num_diseases if num_classes == 2 else num_diseases * num_classes
        self.fc = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, out_dim)
        )

    def forward(self, branch_feats):
        """
        Args:
            branch_feats: list of [B, branch_dim] tensors, one per branch

        Returns:
            binary:  logits [B, num_diseases]
            ternary: logits [B, num_diseases, num_classes]
        """
        x = torch.cat(branch_feats, dim=1)  # [B, total_dim]
        out = self.fc(x)
        if self.num_classes > 2:
            B = out.shape[0]
            out = out.view(B, self.num_diseases, self.num_classes)
        return out
