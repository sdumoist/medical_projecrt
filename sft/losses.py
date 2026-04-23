"""
SFT loss functions.

Stage 1: Pure language modeling loss (CE on output tokens).
Stage 2: LM loss only (auxiliary heads for joint loss are a future extension).
"""
import torch
import torch.nn as nn


class SFTLoss(nn.Module):
    """SFT training loss.

    Currently only language modeling cross-entropy.
    Stage 2 auxiliary heads (label_loss, keyslice_loss) are a planned extension
    that requires differentiable auxiliary heads -- not implemented yet.
    """

    def __init__(self):
        super().__init__()

    def forward(self, lm_loss):
        """Compute total SFT loss.

        Args:
            lm_loss: language modeling loss from LLM forward (scalar tensor)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss components for logging
        """
        loss_dict = {
            "lm_loss": lm_loss.item(),
            "total_loss": lm_loss.item(),
        }
        return lm_loss, loss_dict
