"""
Prediction heads for multi-disease classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinaryHead(nn.Module):
    """Binary classification head for each disease."""

    def __init__(self, input_dim: int, num_diseases: int, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.num_diseases = num_diseases

        # Shared FC
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Per-disease output
        self.output = nn.Linear(input_dim // 2, num_diseases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, input_dim]

        Returns:
            logits: [B, num_diseases]
        """
        x = self.fc(x)
        logits = self.output(x)
        return logits


class TernaryHead(nn.Module):
    """Ternary classification head (negative, uncertain, positive)."""

    def __init__(self, input_dim: int, num_diseases: int, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.num_diseases = num_diseases

        # Shared FC
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Per-disease output (3 classes)
        self.output = nn.Linear(input_dim // 2, num_diseases * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, input_dim]

        Returns:
            logits: [B, num_diseases * 3]
        """
        x = self.fc(x)
        logits = self.output(x)
        # Reshape to [B, num_diseases, 3]
        logits = logits.view(-1, self.num_diseases, 3)
        return logits


class MultiTaskHead(nn.Module):
    """Multi-task head with separate output per disease."""

    def __init__(
        self,
        input_dim: int,
        num_diseases: int,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_diseases = num_diseases
        self.num_classes = num_classes

        hidden_dim = hidden_dim or input_dim // 2

        # Per-disease heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_diseases)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, input_dim]

        Returns:
            logits: [B, num_diseases, num_classes] or [B, num_diseases] for binary
        """
        B = x.shape[0]

        outputs = []
        for head in self.heads:
            out = head(x)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # [B, num_diseases, num_classes]

        if self.num_classes == 2:
            # Return [B, num_diseases] for binary (logits for class 1)
            outputs = outputs[:, :, 1]

        return outputs