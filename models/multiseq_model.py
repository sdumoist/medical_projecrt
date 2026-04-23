"""
ShoulderCoPASModel: CoPAS-style 3-branch model for shoulder MRI.

Architecture:
    5 sequences -> 5 independent 3D encoders
    3 PD-anchored branches:
        sagittal branch: sagittal_PD (main) + coronal_PD, axial_PD (co-plane) + sagittal_T1WI (cross-modal)
        coronal  branch: coronal_PD  (main) + sagittal_PD, axial_PD (co-plane) + coronal_T2WI  (cross-modal)
        axial    branch: axial_PD    (main) + sagittal_PD, coronal_PD (co-plane), no auxiliary
    Each branch:
        1. CoPlaneAttention(main_slice, co1_slice, co2_slice) -> [B, C]
        2. CrossModalAttention(copas_out, aux_pool) -> [B, C]  (sag/cor only)
        3. BranchHead -> branch logits
    Final:
        FinalHead(cat(sag_feat, cor_feat, axi_feat)) -> final logits [B, 7]

    Optional grounded localizer (when use_localizer=True):
        Disease-specific SliceHeads predict key-slice logits from the
        corresponding anchor branch's slice features:
            cor_slice_head: SST/LHBT/IGHL/GHOA on cor_pd_slice
            axi_slice_head: IST/SSC on axi_pd_slice
            sag_slice_head: RIPI on sag_pd_slice
        SoftAttentionLocalTokenPooler extracts 7 disease-aware local tokens.

Loss:
    alpha * (sag_loss + cor_loss + axi_loss) + final_loss
    + localizer_alpha * localizer_loss   (when enabled)

Sequence index convention (matching config data.sequences order):
    0: axial_PD
    1: coronal_PD
    2: coronal_T2WI
    3: sagittal_PD
    4: sagittal_T1WI
"""
import torch
import torch.nn as nn

from models.encoders import get_encoder
from models.fusion_copas import CoPlaneAttention, CrossModalAttention
from models.heads import BranchHead, FinalHead
from utils.constants import NUM_DISEASES

# Sequence name -> index (must match config data.sequences order)
SEQ_INDEX = {
    'axial_PD': 0,
    'coronal_PD': 1,
    'coronal_T2WI': 2,
    'sagittal_PD': 3,
    'sagittal_T1WI': 4,
}


class ShoulderCoPASModel(nn.Module):
    """CoPAS 3-branch model for shoulder MRI classification.

    Args:
        encoder_name: encoder type, e.g. "resnet3d_18", "densenet121"
        num_diseases: number of target diseases (default 7)
        pretrained: whether to use pretrained weights (placeholder)
        dropout: dropout rate
        num_heads: number of attention heads in CoPlaneAttention
        branch_alpha: weight for auxiliary branch losses
        use_localizer: whether to enable key-slice prediction head
        num_classes: 2 for binary, 3 for ternary
    """

    def __init__(
        self,
        encoder_name="resnet3d_18",
        num_diseases=7,
        pretrained=False,
        dropout=0.3,
        num_heads=4,
        branch_alpha=0.3,
        use_localizer=False,
        num_classes=2,
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.branch_alpha = branch_alpha
        self.use_localizer = use_localizer
        self.num_classes = num_classes

        # --- 5 independent encoders (one per sequence) ---
        self.encoders = nn.ModuleDict()
        for seq_name in SEQ_INDEX:
            self.encoders[seq_name] = get_encoder(encoder_name, in_channels=1)

        # Infer feature dim from encoder
        feat_dim = self.encoders['axial_PD'].num_features

        # --- Sagittal branch ---
        # main=sagittal_PD, co-plane=coronal_PD+axial_PD, aux=sagittal_T1WI
        self.sag_copas = CoPlaneAttention(feat_dim, num_heads=num_heads)
        self.sag_cross = CrossModalAttention(feat_dim)

        # --- Coronal branch ---
        # main=coronal_PD, co-plane=sagittal_PD+axial_PD, aux=coronal_T2WI
        self.cor_copas = CoPlaneAttention(feat_dim, num_heads=num_heads)
        self.cor_cross = CrossModalAttention(feat_dim)

        # --- Axial branch ---
        # main=axial_PD, co-plane=sagittal_PD+coronal_PD, no aux
        self.axi_copas = CoPlaneAttention(feat_dim, num_heads=num_heads)
        # no cross-modal for axial

        # --- Heads ---
        self.sag_head = BranchHead(feat_dim, num_diseases, dropout, num_classes)
        self.cor_head = BranchHead(feat_dim, num_diseases, dropout, num_classes)
        self.axi_head = BranchHead(feat_dim, num_diseases, dropout, num_classes)
        self.final_head = FinalHead(feat_dim, num_branches=3,
                                     num_diseases=num_diseases, dropout=dropout,
                                     num_classes=num_classes)

        # --- Grounded localizer: disease-specific key-slice heads ---
        if use_localizer:
            from models.grounding_heads import (
                DiseaseSpecificSliceHeads, SoftAttentionLocalTokenPooler)
            self.grounding_heads = DiseaseSpecificSliceHeads(feat_dim)
            self.local_token_pooler = SoftAttentionLocalTokenPooler(
                temperature=1.0)

    def _encode_slice(self, x, seq_name):
        """Encode a single sequence to per-slice features [B, D', C]."""
        return self.encoders[seq_name].forward_slice(x)

    def _encode_pool(self, x, seq_name):
        """Encode a single sequence to global feature [B, C]."""
        return self.encoders[seq_name].forward_pool(x)

    def forward(self, x, **kwargs):
        """
        Args:
            x: [B, 5, C, D, H, W] where dim=1 follows SEQ_INDEX order
            **kwargs: optional localizer inputs (key_slices, localizer_mask)

        Returns:
            dict with keys:
                'final_logits':  [B, num_diseases]
                'sag_logits':    [B, num_diseases]
                'cor_logits':    [B, num_diseases]
                'axi_logits':    [B, num_diseases]
                'sag_feat':      [B, feat_dim]
                'cor_feat':      [B, feat_dim]
                'axi_feat':      [B, feat_dim]
                'sag_pd_slice':  [B, D', C]  (always returned)
                'cor_pd_slice':  [B, D', C]  (always returned)
                'axi_pd_slice':  [B, D', C]  (always returned)
                'slice_logits':  [B, 7, D']  (only when use_localizer)
                'local_tokens':  [B, 7, C]   (only when use_localizer)
        """
        # Extract individual sequences: x[:, i] -> [B, C, D, H, W]
        axi_pd = x[:, SEQ_INDEX['axial_PD']]
        cor_pd = x[:, SEQ_INDEX['coronal_PD']]
        cor_t2 = x[:, SEQ_INDEX['coronal_T2WI']]
        sag_pd = x[:, SEQ_INDEX['sagittal_PD']]
        sag_t1 = x[:, SEQ_INDEX['sagittal_T1WI']]

        # === Encode PD sequences to slice features (shared across branches) ===
        sag_pd_slice = self._encode_slice(sag_pd, 'sagittal_PD')    # [B, D', C]
        cor_pd_slice = self._encode_slice(cor_pd, 'coronal_PD')     # [B, D', C]
        axi_pd_slice = self._encode_slice(axi_pd, 'axial_PD')      # [B, D', C]

        # === Sagittal branch ===
        sag_t1_pool = self._encode_pool(sag_t1, 'sagittal_T1WI')   # [B, C]
        sag_copas_out = self.sag_copas(sag_pd_slice, cor_pd_slice, axi_pd_slice)  # [B, C]
        sag_feat = self.sag_cross(sag_copas_out, sag_t1_pool)                      # [B, C]

        # === Coronal branch ===
        cor_t2_pool = self._encode_pool(cor_t2, 'coronal_T2WI')     # [B, C]
        cor_copas_out = self.cor_copas(cor_pd_slice, sag_pd_slice, axi_pd_slice)  # [B, C]
        cor_feat = self.cor_cross(cor_copas_out, cor_t2_pool)                      # [B, C]

        # === Axial branch (no cross-modal) ===
        axi_feat = self.axi_copas(axi_pd_slice, sag_pd_slice, cor_pd_slice)  # [B, C]

        # === Heads ===
        sag_logits = self.sag_head(sag_feat)
        cor_logits = self.cor_head(cor_feat)
        axi_logits = self.axi_head(axi_feat)
        final_logits = self.final_head([sag_feat, cor_feat, axi_feat])

        result = {
            'final_logits': final_logits,
            'sag_logits': sag_logits,
            'cor_logits': cor_logits,
            'axi_logits': axi_logits,
            'sag_feat': sag_feat,
            'cor_feat': cor_feat,
            'axi_feat': axi_feat,
            # Always expose slice-level features for downstream consumers
            'sag_pd_slice': sag_pd_slice,
            'cor_pd_slice': cor_pd_slice,
            'axi_pd_slice': axi_pd_slice,
        }

        # === Grounded localizer: disease-specific key-slice + local tokens ===
        if self.use_localizer:
            slice_logits = self.grounding_heads(
                sag_pd_slice, cor_pd_slice, axi_pd_slice)  # [B, 7, D']
            local_tokens = self.local_token_pooler(
                slice_logits, sag_pd_slice, cor_pd_slice, axi_pd_slice)  # [B, 7, C]
            result['slice_logits'] = slice_logits
            result['local_tokens'] = local_tokens

        return result


def create_model(encoder="resnet3d_18", num_diseases=7, pretrained=False,
                 dropout=0.3, num_heads=4, branch_alpha=0.3,
                 use_localizer=False, num_classes=2, **kwargs):
    """Create ShoulderCoPASModel from config."""
    return ShoulderCoPASModel(
        encoder_name=encoder,
        num_diseases=num_diseases,
        pretrained=pretrained,
        dropout=dropout,
        num_heads=num_heads,
        branch_alpha=branch_alpha,
        use_localizer=use_localizer,
        num_classes=num_classes,
    )
