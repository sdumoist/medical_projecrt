#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO training script for shoulder MRI structured diagnosis.

Implements Group Relative Policy Optimization (GRPO) to fine-tune the
SFT-trained ShoulderSFTModel using task-specific reward signals.

Usage:
    # Single-GPU (debug/smoke)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python rl/train_grpo.py \
        --config configs/rl_stage1_clean_grpo.yaml --max_samples 20

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 rl/train_grpo.py \
        --config configs/rl_stage1_clean_grpo.yaml
"""
from __future__ import print_function

import os
import sys
import json
import copy
import yaml
import argparse
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import create_model
from sft.modeling import (
    VisualProjector, ShoulderSFTModel,
    freeze_module, unfreeze_module, apply_freeze_strategy,
)
from sft.dataset import SFTDataset
from rl.grpo_dataset import GRPODataset, make_grpo_collate
from rl.grpo_utils import (
    rollout_batch, compute_advantages, compute_gc_advantages,
    compute_token_log_probs, grpo_loss,
)
from rl.reward_functions import (
    compute_reward,
    extract_predicted_keyslices, extract_gt_keyslices, is_grounding_correct,
)


# ── Config ────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Model loading ─────────────────────────────────────────────────────────

def build_model(config, device):
    """Build ShoulderSFTModel from config.

    Expects config to have:
        mri_cv.checkpoint_path, mri_cv.encoder, mri_cv.use_localizer
        projector.num_visual_tokens
        llm.model_path, llm.lora_rank, ...
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    mri_cv_cfg = config["mri_cv"]
    proj_cfg = config["projector"]
    llm_cfg = config["llm"]

    # ── MRI-CV ──
    mri_cv = create_model(
        encoder=mri_cv_cfg.get("encoder", "swin3d_tiny"),
        use_localizer=mri_cv_cfg.get("use_localizer", True),
        use_roi_head=mri_cv_cfg.get("use_roi_tokens", False),
    )

    ckpt_path = mri_cv_cfg.get("checkpoint_path")
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt = state.get("model_state_dict", state)
        mri_cv.load_state_dict(ckpt, strict=False)
        print("Loaded MRI-CV checkpoint: %s" % ckpt_path)

    # ── LLM ──
    dtype = getattr(torch, llm_cfg.get("torch_dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(
        llm_cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_base = AutoModelForCausalLM.from_pretrained(
        llm_cfg["model_path"],
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=llm_cfg.get("lora_rank", 16),
        lora_alpha=llm_cfg.get("lora_alpha", 32),
        target_modules=llm_cfg.get("lora_target_modules",
                                   ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=llm_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load from SFT checkpoint if provided
    sft_checkpoint = config.get("rl", {}).get("sft_checkpoint")
    if sft_checkpoint:
        if not os.path.isabs(sft_checkpoint):
            sft_checkpoint = os.path.join(PROJECT_ROOT, sft_checkpoint)
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        # SFT checkpoint format: single .pt with projector_state_dict + lora_state_dict
        sft_ckpt = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
        llm = get_peft_model(llm_base, lora_config)
        if "lora_state_dict" in sft_ckpt:
            llm.load_state_dict(sft_ckpt["lora_state_dict"], strict=False)
        print("Loaded SFT LoRA from: %s" % sft_checkpoint)
    else:
        llm = get_peft_model(llm_base, lora_config)
        if sft_checkpoint:
            print("WARNING: SFT checkpoint not found: %s" % sft_checkpoint)
        print("Initialized fresh LoRA (no SFT checkpoint)")

    # ── Projector ──
    cv_dim = mri_cv.encoders["axial_PD"].num_features
    llm_dim = llm.config.hidden_size
    num_slots = proj_cfg.get("num_visual_tokens", 10)
    projector = VisualProjector(cv_dim=cv_dim, llm_dim=llm_dim, num_slots=num_slots)

    use_roi_tokens = mri_cv_cfg.get("use_roi_tokens", False)
    model = ShoulderSFTModel(
        mri_cv=mri_cv,
        projector=projector,
        llm=llm,
        num_visual_tokens=num_slots,
        use_roi_tokens=use_roi_tokens,
        cv_dim=cv_dim if use_roi_tokens else None,
        llm_dim=llm_dim if use_roi_tokens else None,
    )

    # Load projector weights from SFT checkpoint if provided
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        sft_ckpt_data = torch.load(sft_checkpoint, map_location="cpu",
                                   weights_only=False)
        if "projector_state_dict" in sft_ckpt_data:
            model.projector.load_state_dict(sft_ckpt_data["projector_state_dict"])
            print("Loaded projector from SFT checkpoint")

    model = model.to(device)
    return model, tokenizer


# ── Reference model (frozen SFT policy) ───────────────────────────────────

def build_ref_model(model):
    """Create a frozen copy of the model as reference policy."""
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    return ref


# ── Training ──────────────────────────────────────────────────────────────

def train_grpo(config, args):
    # ── Setup ──
    is_ddp = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)

    # ── Model ──
    model, tokenizer = build_model(config, device)

    # Freeze MRI-CV; LoRA + projector trainable
    apply_freeze_strategy(model, stage=1, config=config)

    # Reference policy (frozen)
    ref_model = build_ref_model(model)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
        model._set_static_graph()

    raw_model = model.module if is_ddp else model

    # ── Data ──
    data_cfg = config["data"]
    train_jsonl_files = data_cfg.get("train_jsonl", [])

    # Support single string or list
    if isinstance(train_jsonl_files, str):
        train_jsonl_files = [train_jsonl_files]

    def resolve(p):
        if p and not os.path.isabs(p):
            return os.path.join(PROJECT_ROOT, p)
        return p

    num_vis_tokens = config["projector"].get("num_visual_tokens", 10)
    train_dataset = GRPODataset(
        jsonl_paths=[resolve(p) for p in train_jsonl_files],
        cache_root=resolve(data_cfg.get("cache_root", "")),
        cache_index_path=resolve(data_cfg.get("cache_index", "")),
        tokenizer=tokenizer,
        max_length=data_cfg.get("max_seq_length", 2048),
        num_visual_tokens=num_vis_tokens,
    )

    if args.max_samples and args.max_samples > 0:
        train_dataset.samples = train_dataset.samples[:args.max_samples]
        print("Smoke test: limited to %d samples" % args.max_samples)

    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                     rank=local_rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 1),
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=make_grpo_collate(
            pad_token_id=tokenizer.pad_token_id,
            num_visual_tokens=num_vis_tokens,
        ),
        num_workers=data_cfg.get("num_workers", 2),
    )

    # ── Optimizer ──
    rl_cfg = config.get("rl", {})
    train_cfg = config["training"]
    lr = train_cfg.get("learning_rate", 5e-6)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01))

    # ── GRPO hyperparams ──
    num_generations = rl_cfg.get("num_generations", 4)
    max_new_tokens  = rl_cfg.get("max_new_tokens", 512)
    temperature     = rl_cfg.get("temperature", 0.9)
    beta            = rl_cfg.get("kl_beta", 0.01)
    max_epochs      = train_cfg.get("max_epochs", 2)
    max_grad_norm   = train_cfg.get("max_grad_norm", 1.0)
    algorithm       = rl_cfg.get("algorithm", "grpo")   # "grpo" or "gc_grpo"
    gc_alpha        = rl_cfg.get("gc_alpha", 0.5)       # GC-GRPO inter-group bonus
    gc_tolerance    = rl_cfg.get("gc_tolerance", 1)     # key-slice ±tolerance
    gc_min_frac     = rl_cfg.get("gc_min_disease_frac", 0.5)

    if is_main:
        print("Algorithm: %s%s" % (
            algorithm,
            "  (alpha=%.2f, tol=%d, min_frac=%.2f)" % (gc_alpha, gc_tolerance, gc_min_frac)
            if algorithm == "gc_grpo" else ""))

    # ── Output ──
    output_cfg = config.get("output", {})
    exp_name = output_cfg.get("exp_name", "grpo_run")
    output_dir = os.path.join(output_cfg.get("output_dir", "outputs/rl_experiments"),
                              exp_name)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # ── Training loop ──
    best_reward = 0.0

    for epoch in range(max_epochs):
        model.train()
        raw_model.mri_cv.eval()

        if is_ddp:
            sampler.set_epoch(epoch)

        epoch_rewards = []
        epoch_losses = []

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_strs = batch["output_texts"]
            task_types = batch["task_types"]

            B = images.shape[0]

            # 1. Rollout: generate G completions per prompt
            rollout_batch_data = {
                "image": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            generations, gen_ids, gen_attn, prompt_len = rollout_batch(
                raw_model, tokenizer, rollout_batch_data,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )

            # 2. Compute rewards [B*G]
            rewards_list = []
            for i in range(B):
                ref_str = output_strs[i]
                tt = task_types[i]
                for j in range(num_generations):
                    r = compute_reward(tt, generations[i][j], ref_str)
                    rewards_list.append(r)

            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            epoch_rewards.extend(rewards_list)

            # 3. Advantages [B*G]
            gc_stats = {}
            if algorithm == "gc_grpo":
                # Build grounded_mask: True if rollout correctly located key-slices
                grounded_list = []
                for i in range(B):
                    ref_str = output_strs[i]
                    tt = task_types[i]
                    # Only diagnosis_chain has visual_grounding; others → all G-
                    if tt == "diagnosis_chain":
                        gt_ks = extract_gt_keyslices(ref_str)
                    else:
                        gt_ks = {}
                    for j in range(num_generations):
                        pred_ks = extract_predicted_keyslices(generations[i][j])
                        grounded_list.append(
                            is_grounding_correct(pred_ks, gt_ks,
                                                 tolerance=gc_tolerance,
                                                 min_disease_frac=gc_min_frac)
                        )
                grounded_mask = torch.tensor(grounded_list, dtype=torch.bool, device=device)
                advantages, gc_stats = compute_gc_advantages(
                    rewards, grounded_mask, num_generations,
                    alpha=gc_alpha)
            else:
                advantages = compute_advantages(rewards, num_generations)

            # 4. Log-probs from current and reference policy
            images_rep = images.repeat_interleave(num_generations, dim=0)

            log_probs = compute_token_log_probs(
                raw_model, images_rep, gen_ids, gen_attn, prompt_len)

            with torch.no_grad():
                log_probs_ref = compute_token_log_probs(
                    ref_model, images_rep, gen_ids, gen_attn, prompt_len)

            # 5. GRPO loss
            loss, loss_dict = grpo_loss(log_probs, log_probs_ref, advantages, beta=beta)
            epoch_losses.append(loss_dict["total_loss"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_grad_norm)
            optimizer.step()

            if is_main and step % 10 == 0:
                gc_info = ""
                if gc_stats:
                    gc_info = " gc_frac=%.2f gnd=%.2f" % (
                        gc_stats.get("gc_applied_frac", 0),
                        gc_stats.get("mean_grounded_frac", 0))
                print("Epoch %d step %d | loss=%.4f policy=%.4f kl=%.4f reward=%.4f%s" % (
                    epoch + 1, step,
                    loss_dict["total_loss"], loss_dict["policy_loss"],
                    loss_dict["kl_loss"],
                    sum(rewards_list[-num_generations:]) / num_generations,
                    gc_info))

        # ── End of epoch ──
        mean_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        if is_main:
            print("Epoch %d | mean_reward=%.4f mean_loss=%.4f" % (
                epoch + 1, mean_reward, mean_loss))

            if mean_reward > best_reward:
                best_reward = mean_reward
                ckpt_path = os.path.join(output_dir, "best_checkpoint")
                os.makedirs(ckpt_path, exist_ok=True)
                raw_model.llm.save_pretrained(ckpt_path)
                torch.save(raw_model.projector.state_dict(),
                           os.path.join(ckpt_path, "projector.pt"))
                print("Saved best checkpoint (reward=%.4f) to %s" % (
                    best_reward, ckpt_path))

    if is_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="GRPO training for shoulder MRI SFT")
    parser.add_argument("--config", required=True, help="Path to GRPO config YAML")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset to N samples (smoke test)")
    args = parser.parse_args()

    config = load_config(args.config)
    train_grpo(config, args)


if __name__ == "__main__":
    main()
