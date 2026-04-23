#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT training script for shoulder MRI + Qwen2.5-7B.

Supports:
  - Stage 1: Frozen MRI-CV, train VisualProjector + Qwen LoRA
  - Stage 2: Partially unfrozen MRI-CV, same loss (LM only)

Usage:
    # Stage 1 (single GPU smoke test)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python sft/train_sft.py \
        --config configs/sft_stage1_frozen.yaml

    # Stage 1 (multi-GPU with DeepSpeed)
    torchrun --nproc_per_node=8 sft/train_sft.py \
        --config configs/sft_stage1_frozen.yaml
"""
from __future__ import print_function

import os
import sys
import json
import yaml
import argparse
import warnings
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import create_model
from sft.dataset import SFTDataset, sft_collate_fn
from sft.losses import SFTLoss
from sft.eval_utils import evaluate_sample, aggregate_metrics, try_parse_json


# ── Visual Projector ─────────────────────────────────────────────────────

class VisualProjector(nn.Module):
    """Project visual features [B, N, C] to LLM embedding space [B, N, H].

    Handles both global branch tokens (3) and local key-slice tokens (7).
    The same shared MLP is applied to all tokens independently.
    """

    def __init__(self, cv_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cv_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_tokens):
        """
        Args:
            visual_tokens: [B, N, C] where N = num_visual_tokens (3 or 10)
        Returns:
            projected: [B, N, H] in LLM dim
        """
        return self.proj(visual_tokens)


# ── SFT Model ────────────────────────────────────────────────────────────

class ShoulderSFTModel(nn.Module):
    """Full SFT model: MRI-CV → Projector → Qwen2.5-7B.

    When MRI-CV has use_localizer=True, produces 10 visual tokens:
        [sag_feat, cor_feat, axi_feat] + [SST, IST, SSC, LHBT, IGHL, RIPI, GHOA]
    Otherwise falls back to 3 global tokens only.
    """

    def __init__(self, mri_cv, projector, llm, num_visual_tokens=10):
        super().__init__()
        self.mri_cv = mri_cv
        self.projector = projector
        self.llm = llm
        self.num_visual_tokens = num_visual_tokens
        self.mri_cv_frozen = True

    def _extract_visual_tokens(self, images):
        """Extract visual tokens from MRI-CV.

        Args:
            images: [B, 5, 1, Z, H, W]
        Returns:
            visual_feats: [B, N, C] where N = num_visual_tokens
            cv_out: full MRI-CV output dict (for slice_logits access in Stage 2)
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

        return visual_feats, cv_out

    def _merge_visual_tokens(self, text_embeds, visual_tokens):
        """Replace first N visual token positions with projected embeddings.

        Args:
            text_embeds: [B, L, H] from llm.embed_tokens(input_ids)
            visual_tokens: [B, N_vis, H] from projector

        Returns:
            merged_embeds: [B, L, H] with visual tokens injected
        """
        merged = text_embeds.clone()
        N = self.num_visual_tokens
        merged[:, :N, :] = visual_tokens
        return merged

    def _get_embed_fn(self):
        """Get the embed_tokens function, handling PeftModel wrapping."""
        # PeftModel -> base_model -> model -> model -> embed_tokens
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
        """
        Args:
            images: [B, 5, 1, Z, H, W]
            input_ids: [B, L] (first N positions are visual placeholders)
            attention_mask: [B, L]
            labels: [B, L] (-100 for non-output positions)

        Returns:
            outputs: LLM outputs with .loss and .logits
            cv_out: MRI-CV output dict (contains slice_logits for Stage 2 loss)
        """
        # 1. Extract and project visual features
        visual_feats, cv_out = self._extract_visual_tokens(images)
        visual_tokens = self.projector(visual_feats)  # [B, N, H]

        # 2. Get text embeddings
        embed_fn = self._get_embed_fn()

        text_embeds = embed_fn(input_ids)  # [B, L, H]

        # 3. Merge visual tokens into text embeddings
        inputs_embeds = self._merge_visual_tokens(
            text_embeds, visual_tokens.to(text_embeds.dtype))

        # 4. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs, cv_out

    @torch.no_grad()
    def generate(self, images, input_ids, attention_mask, **gen_kwargs):
        """Generate text autoregressively.

        Args:
            images: [B, 5, 1, Z, H, W]
            input_ids: [B, L] prompt tokens (with visual placeholders)
            attention_mask: [B, L]
            **gen_kwargs: passed to llm.generate()

        Returns:
            generated_ids: [B, L_out]
        """
        visual_feats, _ = self._extract_visual_tokens(images)
        visual_tokens = self.projector(visual_feats)

        embed_fn = self._get_embed_fn()

        text_embeds = embed_fn(input_ids)
        inputs_embeds = self._merge_visual_tokens(
            text_embeds, visual_tokens.to(text_embeds.dtype))

        # Pass input_ids alongside inputs_embeds so the returned sequence
        # includes the prompt token IDs (inputs_embeds is used for the
        # first forward pass, input_ids seeds the output sequence).
        return self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )


# ── Freeze utilities ─────────────────────────────────────────────────────

def freeze_module(module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def apply_freeze_strategy(model, stage, config):
    """Apply freezing strategy based on stage.

    Stage 1: Freeze MRI-CV entirely. Train projector + LoRA.
    Stage 2: Unfreeze MRI-CV last stage. Train projector + LoRA.
    """
    # Always freeze MRI-CV first
    freeze_module(model.mri_cv)
    model.mri_cv_frozen = True

    if stage == 2:
        # Unfreeze last stage of each encoder
        unfreeze_layers = config.get("mri_cv", {}).get(
            "unfreeze_layers", "last_stage")

        for seq_name, encoder in model.mri_cv.encoders.items():
            if unfreeze_layers == "last_stage":
                # Try common last-stage attribute names
                for attr in ["layer4", "layer3"]:
                    if hasattr(encoder, attr):
                        unfreeze_module(getattr(encoder, attr))
                        break
                else:
                    # For Swin-style: unfreeze last stage
                    if hasattr(encoder, "stages"):
                        unfreeze_module(encoder.stages[-1])

        model.mri_cv_frozen = False

    # Projector is always trainable
    unfreeze_module(model.projector)

    # Count trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters: total=%d (%.1fM), trainable=%d (%.1fM), frozen=%d" % (
        total, total / 1e6, trainable, trainable / 1e6, total - trainable))


# ── Config loading ───────────────────────────────────────────────────────

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch,
                max_grad_norm=1.0, grad_accum_steps=1, keyslice_alpha=0.0):
    """Train for one epoch with gradient accumulation.

    Args:
        keyslice_alpha: weight for L_keyslice loss (Stage 2). 0 = disabled.
    """
    model.train()
    # Keep frozen MRI-CV in eval mode to avoid BatchNorm issues with bs=1
    if model.mri_cv_frozen:
        model.mri_cv.eval()
    else:
        # Stage 2: MRI-CV partially unfrozen, but still set eval for
        # frozen sub-modules to keep BatchNorm stable
        model.mri_cv.eval()
        # Then set only unfrozen sub-modules back to train
        for name, module in model.mri_cv.named_modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                module.train()
    total_loss = 0.0
    num_steps = 0
    accum_loss = 0.0

    try:
        from tqdm import tqdm
        iter_ = tqdm(loader, desc="Train epoch %d" % (epoch + 1), leave=False)
    except ImportError:
        iter_ = loader

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(iter_):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs, cv_out = model(images, input_ids, attention_mask, labels=labels)
        loss = outputs.loss / grad_accum_steps

        # Optional key-slice loss (Stage 2: L_keyslice from grounding heads)
        if keyslice_alpha > 0 and 'slice_logits' in cv_out:
            ks_gt = batch.get("key_slices")  # [B, 7] or None
            if ks_gt is not None:
                import torch.nn.functional as F
                slice_logits = cv_out['slice_logits']  # [B, 7, D']
                B_ks, _, D_ks = slice_logits.shape
                ks_gt = ks_gt.to(device).float()
                input_Z = images.shape[3]
                ks_scaled = (ks_gt * D_ks / input_Z).long().clamp(0, D_ks - 1)
                label_mask = batch.get("mask")
                if label_mask is not None:
                    label_mask = label_mask.to(device)
                    valid = (ks_gt >= 0) & (label_mask > 0)
                else:
                    valid = ks_gt >= 0
                if valid.any() and D_ks > 1:
                    ks_loss = F.cross_entropy(
                        slice_logits[valid], ks_scaled[valid])
                    loss = loss + keyslice_alpha * ks_loss / grad_accum_steps

        loss.backward()
        accum_loss += loss.item()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += accum_loss
            num_steps += 1
            accum_loss = 0.0

        if hasattr(iter_, "set_postfix"):
            iter_.set_postfix(loss="%.4f" % (loss.item() * grad_accum_steps))

    avg_loss = total_loss / max(num_steps, 1)
    return {"loss": avg_loss, "steps": num_steps}


@torch.no_grad()
def validate(model, loader, tokenizer, device, max_new_tokens=512):
    """Validate: compute LM loss + generate and evaluate outputs."""
    model.eval()

    total_loss = 0.0
    num_steps = 0
    results_by_task = defaultdict(list)

    try:
        from tqdm import tqdm
        iter_ = tqdm(loader, desc="Validate", leave=False)
    except ImportError:
        iter_ = loader

    for batch in iter_:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Compute loss
        outputs, cv_out = model(images, input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_steps += 1

        # Generate for a subset (every 10th batch to save time)
        if num_steps % 10 == 1:
            # Build prompt-only input (mask output tokens)
            prompt_lens = batch["prompt_lens"]
            num_vis = model.num_visual_tokens

            for i in range(len(batch["exam_ids"])):
                plen = prompt_lens[i] + num_vis
                prompt_ids = input_ids[i:i+1, :plen]
                prompt_mask = attention_mask[i:i+1, :plen]
                img = images[i:i+1]

                try:
                    gen_ids = model.generate(
                        img, prompt_ids, prompt_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
                    # gen_ids includes prompt (input_ids passed to generate).
                    # Slice off prompt to get only generated tokens.
                    gen_len = gen_ids.shape[1]
                    if gen_len > plen:
                        gen_text = tokenizer.decode(
                            gen_ids[0][plen:], skip_special_tokens=True)
                    else:
                        # Fallback: if generate returned only new tokens
                        gen_text = tokenizer.decode(
                            gen_ids[0], skip_special_tokens=True)
                except Exception:
                    gen_text = ""

                task_type = batch["task_types"][i]
                gt_text = batch["output_texts"][i]

                metrics = evaluate_sample(task_type, gen_text, gt_text)
                results_by_task[task_type].append(metrics)

    avg_loss = total_loss / max(num_steps, 1)

    # Aggregate per-task metrics
    task_summaries = {}
    for task, results in results_by_task.items():
        task_summaries[task] = aggregate_metrics(results)

    return {
        "loss": avg_loss,
        "task_metrics": task_summaries,
        "num_eval_samples": sum(len(v) for v in results_by_task.values()),
    }


def train(config, args):
    """Main training function."""
    stage = config.get("stage", 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Stage: %d, Device: %s" % (stage, device))

    # ── Output directory ──
    exp_name = config["output"]["exp_name"]
    output_dir = os.path.join(config["output"]["output_dir"], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print("Output: %s" % output_dir)

    # Save resolved config
    with open(os.path.join(output_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    llm_path = config["llm"].get("model_path", config["llm"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load MRI-CV model ──
    mri_cv_cfg = config["mri_cv"]
    mri_cv = create_model(
        encoder=mri_cv_cfg["encoder"],
        num_diseases=7,
        pretrained=False,
        use_localizer=mri_cv_cfg.get("use_localizer", False),
        num_classes=2,
    )

    ckpt_path = mri_cv_cfg.get("checkpoint_path")
    if ckpt_path:
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mri_cv.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("Loaded MRI-CV checkpoint: %s" % ckpt_path)
        else:
            warnings.warn("MRI-CV checkpoint not found: %s" % ckpt_path)

    # ── Build projector ──
    feat_dim = mri_cv.encoders["axial_PD"].num_features
    proj_cfg = config.get("projector", {})
    num_visual_tokens = proj_cfg.get("num_visual_tokens", 10)

    # ── Load LLM + LoRA ──
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig

    llm_cfg = config["llm"]
    torch_dtype = getattr(torch, llm_cfg.get("torch_dtype", "bfloat16"))

    llm_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if llm_cfg.get("use_flash_attn", False):
        llm_kwargs["attn_implementation"] = "flash_attention_2"

    llm = AutoModelForCausalLM.from_pretrained(llm_path, **llm_kwargs)
    llm_dim = llm.config.hidden_size

    if llm_cfg.get("gradient_checkpointing", False):
        llm.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = LoraConfig(
        r=llm_cfg.get("lora_rank", 16),
        lora_alpha=llm_cfg.get("lora_alpha", 32),
        target_modules=llm_cfg.get("lora_target_modules",
                                    ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=llm_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # ── Assemble SFT model ──
    projector = VisualProjector(cv_dim=feat_dim, llm_dim=llm_dim)
    model = ShoulderSFTModel(mri_cv, projector, llm, num_visual_tokens)
    model = model.to(device)

    # Apply freeze strategy
    apply_freeze_strategy(model, stage, config)

    # ── Resume from checkpoint ──
    start_epoch = 0
    if args.resume:
        resume_path = args.resume
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.projector.load_state_dict(ckpt.get("projector_state_dict", {}))
            if "lora_state_dict" in ckpt:
                model.llm.load_state_dict(ckpt["lora_state_dict"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            print("Resumed from %s (epoch %d)" % (resume_path, start_epoch))

    # ── Data ──
    data_cfg = config["data"]

    # Resolve relative paths
    def resolve(p):
        if p and not os.path.isabs(p):
            return os.path.join(PROJECT_ROOT, p)
        return p

    train_dataset = SFTDataset(
        jsonl_paths=[resolve(p) for p in data_cfg["train_jsonl"]],
        cache_root=resolve(data_cfg.get("cache_root")),
        cache_index_path=resolve(data_cfg.get("cache_index")),
        tokenizer=tokenizer,
        max_length=data_cfg.get("max_seq_length", 2048),
        num_visual_tokens=num_visual_tokens,
    )

    # Limit samples for smoke test
    if args.max_samples > 0 and len(train_dataset.samples) > args.max_samples:
        train_dataset.samples = train_dataset.samples[:args.max_samples]
        print("Smoke test: limited to %d training samples" % args.max_samples)

    val_jsonl = data_cfg.get("val_jsonl", [])
    val_dataset = None
    if val_jsonl:
        val_dataset = SFTDataset(
            jsonl_paths=[resolve(p) for p in val_jsonl],
            cache_root=resolve(data_cfg.get("cache_root")),
            cache_index_path=resolve(data_cfg.get("cache_index")),
            tokenizer=tokenizer,
            max_length=data_cfg.get("max_seq_length", 2048),
            num_visual_tokens=num_visual_tokens,
        )
        if args.max_samples > 0 and len(val_dataset.samples) > args.max_samples:
            val_dataset.samples = val_dataset.samples[:args.max_samples]

    train_cfg = config["training"]
    collate = partial(sft_collate_fn,
                       pad_token_id=tokenizer.pad_token_id,
                       num_visual_tokens=num_visual_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate,
            pin_memory=True,
        )

    # ── Optimizer ──
    # Separate param groups for different LR
    param_groups = []
    # Projector
    param_groups.append({
        "params": [p for p in model.projector.parameters() if p.requires_grad],
        "lr": train_cfg["learning_rate"],
        "name": "projector",
    })
    # LoRA
    lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    if lora_params:
        param_groups.append({
            "params": lora_params,
            "lr": train_cfg["learning_rate"],
            "name": "lora",
        })
    # MRI-CV unfrozen (Stage 2)
    cv_params = [p for p in model.mri_cv.parameters() if p.requires_grad]
    if cv_params:
        cv_lr_mult = config.get("mri_cv", {}).get("lr_multiplier", 0.1)
        param_groups.append({
            "params": cv_params,
            "lr": train_cfg["learning_rate"] * cv_lr_mult,
            "name": "mri_cv_unfrozen",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    steps_per_epoch = (len(train_loader) + grad_accum - 1) // grad_accum
    total_steps = steps_per_epoch * train_cfg["max_epochs"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.03))

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ──
    best_val_loss = float("inf")
    jsonl_path = os.path.join(output_dir, "metrics_epoch.jsonl")
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    keyslice_alpha = train_cfg.get("keyslice_alpha", 0.0)
    print("Gradient accumulation steps: %d (effective batch = %d)" % (
        grad_accum_steps, train_cfg["batch_size"] * grad_accum_steps))
    if keyslice_alpha > 0:
        print("Key-slice loss weight: %.2f (Stage 2)" % keyslice_alpha)

    for epoch in range(start_epoch, train_cfg["max_epochs"]):
        print("\n[Epoch %d/%d]" % (epoch + 1, train_cfg["max_epochs"]))

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            grad_accum_steps=grad_accum_steps,
            keyslice_alpha=keyslice_alpha)

        print("  Train loss: %.4f (steps: %d)" % (
            train_metrics["loss"], train_metrics["steps"]))

        # Validation
        val_metrics = None
        val_interval = config.get("validation", {}).get("val_interval", 1)
        if val_loader and (epoch + 1) % val_interval == 0:
            gen_max = config.get("validation", {}).get(
                "generation_max_length", 512)
            val_metrics = validate(
                model, val_loader, tokenizer, device,
                max_new_tokens=gen_max)

            print("  Val loss: %.4f  (eval samples: %d)" % (
                val_metrics["loss"], val_metrics["num_eval_samples"]))

            for task, tm in val_metrics.get("task_metrics", {}).items():
                parts = ["  %s:" % task]
                for k, v in sorted(tm.items()):
                    parts.append("%s=%.3f" % (k, v))
                print(" ".join(parts))

            is_best = val_metrics["loss"] < best_val_loss
        else:
            is_best = False

        # Save metrics JSONL
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
        }
        if val_metrics:
            record["val_loss"] = val_metrics["loss"]
            record["val_task_metrics"] = val_metrics.get("task_metrics", {})
            record["is_best"] = is_best
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Save last checkpoint
        _save_checkpoint(model, optimizer, epoch, train_metrics,
                          os.path.join(output_dir, "last_checkpoint.pt"),
                          config)

        # Save best checkpoint
        if val_metrics and is_best:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(model, optimizer, epoch, val_metrics,
                              os.path.join(output_dir, "best_checkpoint.pt"),
                              config)
            print("  Saved best checkpoint (val_loss=%.4f)" % best_val_loss)

    print("\nTraining complete! Best val loss: %.4f" % best_val_loss)


def _save_checkpoint(model, optimizer, epoch, metrics, path, config):
    """Save SFT checkpoint."""
    # Extract projector and LoRA state dicts
    proj_sd = model.projector.state_dict()

    # For LoRA, save only adapter weights
    lora_sd = {}
    for n, p in model.llm.named_parameters():
        if p.requires_grad:
            lora_sd[n] = p.data.cpu()

    # MRI-CV unfrozen params (if any)
    cv_sd = {}
    for n, p in model.mri_cv.named_parameters():
        if p.requires_grad:
            cv_sd[n] = p.data.cpu()

    torch.save({
        "epoch": epoch,
        "projector_state_dict": proj_sd,
        "lora_state_dict": lora_sd,
        "mri_cv_unfrozen_state_dict": cv_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to SFT config YAML")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output", "-o", default=None,
                        help="Override output directory")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit training samples (0=all, for smoke test)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output:
        config["output"]["output_dir"] = args.output

    train(config, args)


if __name__ == "__main__":
    main()
