"""
Prediction heads for shoulder MRI multi-disease classification.

Supports:
- BranchHead: per-branch (sag/cor/axi) binary classification
- FinalHead:  final fusion head combining all branch features
"""
import torch
import torch.nn as nn


class BranchHead(nn.Module):
    """Binary classification head for a single branch.

    Args:
        input_dim: feature dimension from fusion output
        num_diseases: number of diseases this branch predicts
        dropout: dropout rate
    """

    def __init__(self, input_dim, num_diseases, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_diseases)
        )

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B, num_diseases]
        """
        return self.fc(x)


class FinalHead(nn.Module):
    """Final head that combines all branch features for joint prediction.

    Concatenates branch features and predicts all 7 diseases.

    Args:
        branch_dim: feature dimension per branch
        num_branches: number of branches (3: sag, cor, axi)
        num_diseases: total number of diseases (7)
        dropout: dropout rate
    """

    def __init__(self, branch_dim, num_branches, num_diseases, dropout=0.3):
        super().__init__()
        total_dim = branch_dim * num_branches
        self.fc = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, num_diseases)
        )

    def forward(self, branch_feats):
        """
        Args:
            branch_feats: list of [B, branch_dim] tensors, one per branch

        Returns:
            logits: [B, num_diseases]
        """
        x = torch.cat(branch_feats, dim=1)  # [B, total_dim]
        return self.fc(x)
