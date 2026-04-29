"""
SFT model definitions for shoulder MRI + Qwen2.5.

Extracted from sft/train_sft.py so that train_sft.py, eval scripts,
GRPO scripts and any future consumers share the same model definition.

Contents:
    VisualProjector         -- project MRI-CV features to LLM embedding space
    ShoulderSFTModel        -- MRI-CV + Projector + Qwen2.5 end-to-end model
    freeze_module           -- freeze all params in a module
    unfreeze_module         -- unfreeze all params in a module
    apply_freeze_strategy   -- Stage 1 / Stage 2 freeze policy
"""
import torch
import torch.nn as nn


# ── Visual Projector ──────────────────────────────────────────────────────

class VisualProjector(nn.Module):
    """Project visual features [B, N, C] to LLM embedding space [B, N, H].

    Handles both global branch tokens (3) and local key-slice tokens (7).
    The same shared MLP is applied to all tokens independently.

    Learnable slot embeddings are added to the projected tokens to encode
    the identity of each visual slot:
        0: SAG_GLOBAL, 1: COR_GLOBAL, 2: AXI_GLOBAL,
        3: SST_LOCAL, 4: IST_LOCAL, 5: SSC_LOCAL, 6: LHBT_LOCAL,
        7: IGHL_LOCAL, 8: RIPI_LOCAL, 9: GHOA_LOCAL

    ROI-extended layout (num_slots=17):
        10-16: SST_ROI, IST_ROI, SSC_ROI, LHBT_ROI, IGHL_ROI, RIPI_ROI, GHOA_ROI
    """

    def __init__(self, cv_dim, llm_dim, num_slots=10):
        super().__init__()
        self.num_slots = num_slots
        self.proj = nn.Sequential(
            nn.Linear(cv_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        # Learnable slot identity embeddings [N, H]
        self.slot_embed = nn.Embedding(num_slots, llm_dim)

    def forward(self, visual_tokens):
        """
        Args:
            visual_tokens: [B, N, C] where N = num_visual_tokens (3, 10 or 17)
        Returns:
            projected: [B, N, H] in LLM dim, with slot identity added
        """
        B, N, _ = visual_tokens.shape
        projected = self.proj(visual_tokens)  # [B, N, H]
        # Add slot identity embeddings (broadcast over batch)
        slot_ids = torch.arange(N, device=visual_tokens.device)
        projected = projected + self.slot_embed(slot_ids).unsqueeze(0)
        return projected


# ── SFT Model ─────────────────────────────────────────────────────────────

class ShoulderSFTModel(nn.Module):
    """Full SFT model: MRI-CV → Projector → Qwen2.5-7B.

    Token layout (num_visual_tokens controls which are used):
        10 tokens:  [sag, cor, axi] + [SST, IST, SSC, LHBT, IGHL, RIPI, GHOA]
        17 tokens:  above + [SST_roi, IST_roi, SSC_roi, LHBT_roi, IGHL_roi, RIPI_roi, GHOA_roi]

    When use_roi_tokens=True and MRI-CV has roi_box_2d, the 7 ROI summary
    tokens are projected from roi_box_2d (4-dim) + roi_box_conf (1-dim)
    via a small linear layer before being concatenated.
    """

    def __init__(self, mri_cv, projector, llm, num_visual_tokens=10,
                 use_roi_tokens=False, cv_dim=None, llm_dim=None):
        super().__init__()
        self.mri_cv = mri_cv
        self.projector = projector
        self.llm = llm
        self.num_visual_tokens = num_visual_tokens
        self.mri_cv_frozen = True
        self.use_roi_tokens = use_roi_tokens

        # Small linear to lift (box 4 + conf 1) → cv_dim for ROI tokens
        if use_roi_tokens and cv_dim is not None:
            self.roi_token_proj = nn.Linear(5, cv_dim)
        else:
            self.roi_token_proj = None

    def _extract_visual_tokens(self, images):
        """Extract visual tokens from MRI-CV.

        Returns:
            visual_feats: [B, N, C] where N = num_visual_tokens
            cv_out: full MRI-CV output dict
        """
        ctx = torch.no_grad() if self.mri_cv_frozen else torch.enable_grad()
        with ctx:
            cv_out = self.mri_cv(images)

        # 3 global branch tokens
        global_tokens = torch.stack([
            cv_out["sag_feat"],
            cv_out["cor_feat"],
            cv_out["axi_feat"],
        ], dim=1)  # [B, 3, C]

        # If localizer is enabled, append 7 disease-aware local tokens
        if "local_tokens" in cv_out:
            local_tokens = cv_out["local_tokens"]  # [B, 7, C]
            visual_feats = torch.cat([global_tokens, local_tokens], dim=1)  # [B, 10, C]
        else:
            visual_feats = global_tokens  # [B, 3, C]

        # If ROI tokens requested and roi_box_2d exists
        if self.use_roi_tokens and self.roi_token_proj is not None \
                and "roi_box_2d" in cv_out:
            roi_box  = cv_out["roi_box_2d"]   # [B, 7, 4]
            roi_conf = cv_out["roi_box_conf"].unsqueeze(-1)  # [B, 7, 1]
            roi_feat_in = torch.cat([roi_box, roi_conf], dim=-1)   # [B, 7, 5]
            roi_tokens  = self.roi_token_proj(roi_feat_in.to(visual_feats.dtype))  # [B, 7, C]
            visual_feats = torch.cat([visual_feats, roi_tokens], dim=1)  # [B, 17, C]

        return visual_feats, cv_out

    def _merge_visual_tokens(self, text_embeds, visual_tokens):
        """Replace first N positions with projected visual embeddings."""
        merged = text_embeds.clone()
        N = self.num_visual_tokens
        merged[:, :N, :] = visual_tokens
        return merged

    def _get_embed_fn(self):
        """Get embed_tokens function, handling PeftModel wrapping."""
        llm = self.llm
        if hasattr(llm, "base_model"):
            llm = llm.base_model
        if hasattr(llm, "model"):
            inner = llm.model
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens
            if hasattr(inner, "model") and hasattr(inner.model, "embed_tokens"):
                return inner.model.embed_tokens
        return llm.get_input_embeddings()

    def forward(self, images, input_ids, attention_mask, labels=None):
        visual_feats, cv_out = self._extract_visual_tokens(images)
        visual_tokens = self.projector(visual_feats)  # [B, N, H]

        embed_fn = self._get_embed_fn()
        text_embeds = embed_fn(input_ids)  # [B, L, H]
        inputs_embeds = self._merge_visual_tokens(
            text_embeds, visual_tokens.to(text_embeds.dtype))

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs, cv_out

    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask, **gen_kwargs):
        visual_feats, _ = self._extract_visual_tokens(images)
        visual_tokens = self.projector(visual_feats)

        embed_fn = self._get_embed_fn()
        text_embeds = embed_fn(input_ids)
        inputs_embeds = self._merge_visual_tokens(
            text_embeds, visual_tokens.to(text_embeds.dtype))

        return self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )


# ── Freeze utilities ──────────────────────────────────────────────────────

def freeze_module(module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def apply_freeze_strategy(model, stage, config):
    """Apply freezing strategy based on training stage.

    Stage 1: Freeze MRI-CV entirely. Train projector + LoRA.
    Stage 2: Unfreeze MRI-CV last stage. Train projector + LoRA + last stage.
    """
    # Always freeze MRI-CV first
    freeze_module(model.mri_cv)
    model.mri_cv_frozen = True

    if stage == 2:
        unfreeze_layers = config.get("mri_cv", {}).get(
            "unfreeze_layers", "last_stage")

        for seq_name, encoder in model.mri_cv.encoders.items():
            if unfreeze_layers == "last_stage":
                for attr in ["layer4", "layer3"]:
                    if hasattr(encoder, attr):
                        unfreeze_module(getattr(encoder, attr))
                        break
                else:
                    if hasattr(encoder, "stages"):
                        unfreeze_module(encoder.stages[-1])

        model.mri_cv_frozen = False

    # Projector is always trainable
    unfreeze_module(model.projector)

    # ROI token projector (if exists) is always trainable
    if hasattr(model, "roi_token_proj") and model.roi_token_proj is not None:
        unfreeze_module(model.roi_token_proj)

    # Count trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters: total=%d (%.1fM), trainable=%d (%.1fM), frozen=%d" % (
        total, total / 1e6, trainable, trainable / 1e6, total - trainable))
