"""
Grounding heads for disease-specific key-slice prediction and local token extraction.

Architecture:
    3 branch-specific SliceHeads (cor/axi/sag), each predicting key_slice
    for its assigned diseases from the corresponding branch's slice features.

    SoftAttentionLocalTokenPooler uses slice_logits as attention weights
    to extract per-disease local tokens via differentiable weighted pooling.

Branch routing (from utils.constants.DISEASE_BRANCH_MAP):
    cor_slice_head: SST, LHBT, IGHL, GHOA  (indices 0,3,4,6) on cor_pd_slice
    axi_slice_head: IST, SSC               (indices 1,2)     on axi_pd_slice
    sag_slice_head: RIPI                    (index 5)         on sag_pd_slice
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import (
    NUM_DISEASES, DISEASE_BRANCH_MAP, DISEASE_TO_BRANCH,
)


class BranchSliceHead(nn.Module):
    """Predict key-slice logits for a subset of diseases from one branch.

    Input:  slice_features [B, D', C]
    Output: slice_logits   [B, N_d, D']  (N_d = number of diseases for this branch)
    """

    def __init__(self, feat_dim, num_diseases_in_branch):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_diseases_in_branch)

    def forward(self, slice_features):
        """
        Args:
            slice_features: [B, D', C]
        Returns:
            logits: [B, N_d, D']
        """
        # [B, D', C] -> [B, D', N_d]
        logits = self.fc(slice_features)
        # -> [B, N_d, D']
        return logits.permute(0, 2, 1)


class DiseaseSpecificSliceHeads(nn.Module):
    """Three branch-specific SliceHeads assembled into a unified interface.

    Returns unified slice_logits [B, 7, D'] in canonical disease order.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        # Create one head per branch
        self.cor_slice_head = BranchSliceHead(
            feat_dim, len(DISEASE_BRANCH_MAP["cor"]))  # 4 diseases
        self.axi_slice_head = BranchSliceHead(
            feat_dim, len(DISEASE_BRANCH_MAP["axi"]))  # 2 diseases
        self.sag_slice_head = BranchSliceHead(
            feat_dim, len(DISEASE_BRANCH_MAP["sag"]))  # 1 disease

        # Store branch -> disease indices mapping as buffers for easy access
        # (not parameters, just index tensors)
        self.register_buffer(
            "cor_indices",
            torch.tensor(DISEASE_BRANCH_MAP["cor"], dtype=torch.long))
        self.register_buffer(
            "axi_indices",
            torch.tensor(DISEASE_BRANCH_MAP["axi"], dtype=torch.long))
        self.register_buffer(
            "sag_indices",
            torch.tensor(DISEASE_BRANCH_MAP["sag"], dtype=torch.long))

    def forward(self, sag_pd_slice, cor_pd_slice, axi_pd_slice):
        """
        Args:
            sag_pd_slice: [B, D', C] sagittal PD per-slice features
            cor_pd_slice: [B, D', C] coronal PD per-slice features
            axi_pd_slice: [B, D', C] axial PD per-slice features

        Returns:
            slice_logits: [B, 7, D'] unified logits in canonical disease order
        """
        B = cor_pd_slice.shape[0]
        D = cor_pd_slice.shape[1]

        # Each head predicts for its assigned diseases
        cor_logits = self.cor_slice_head(cor_pd_slice)  # [B, 4, D']
        axi_logits = self.axi_slice_head(axi_pd_slice)  # [B, 2, D']
        sag_logits = self.sag_slice_head(sag_pd_slice)  # [B, 1, D']

        # Assemble into canonical [B, 7, D'] order
        # Use dtype from actual head output (handles AMP bf16/fp32 mismatch)
        slice_logits = cor_logits.new_zeros(B, NUM_DISEASES, D)
        slice_logits[:, self.cor_indices] = cor_logits
        slice_logits[:, self.axi_indices] = axi_logits
        slice_logits[:, self.sag_indices] = sag_logits

        return slice_logits


class SoftAttentionLocalTokenPooler(nn.Module):
    """Extract per-disease local tokens via soft attention over slice dimension.

    For each disease d:
        1. Get slice_logits[d]: [B, D']
        2. attn_weights = softmax(logits / temperature, dim=-1): [B, D']
        3. local_token = weighted sum over slice features: [B, C]

    The attention weights are derived from the SliceHead logits, making the
    extraction fully differentiable. The SliceHead learns to focus on the
    diagnostically relevant slice for each disease.

    Output: local_tokens [B, 7, C] (one per disease in canonical order)
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, slice_logits, sag_pd_slice, cor_pd_slice, axi_pd_slice):
        """
        Args:
            slice_logits: [B, 7, D'] from DiseaseSpecificSliceHeads
            sag_pd_slice: [B, D', C]
            cor_pd_slice: [B, D', C]
            axi_pd_slice: [B, D', C]

        Returns:
            local_tokens: [B, 7, C]
        """
        B, _, D = slice_logits.shape
        C = cor_pd_slice.shape[-1]

        # Softmax over slice dimension for each disease
        attn_weights = F.softmax(
            slice_logits / self.temperature, dim=-1)  # [B, 7, D']

        # Build per-disease local tokens using the correct branch features
        local_tokens = cor_pd_slice.new_zeros(B, NUM_DISEASES, C)

        # Branch feature lookup
        branch_feats = {
            "cor": cor_pd_slice,  # [B, D', C]
            "axi": axi_pd_slice,
            "sag": sag_pd_slice,
        }

        for d_idx in range(NUM_DISEASES):
            branch_name = DISEASE_TO_BRANCH[d_idx]
            feats = branch_feats[branch_name]        # [B, D', C]
            w = attn_weights[:, d_idx, :]             # [B, D']
            # Weighted sum: [B, D'] x [B, D', C] -> [B, C]
            local_tokens[:, d_idx] = torch.einsum("bd,bdc->bc", w, feats)

        return local_tokens
