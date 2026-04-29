"""
GRPO utilities: rollout, advantage computation, and GRPO policy loss.

Algorithm overview (per batch):
    1. For each prompt, sample G generations (group size)
    2. Compute reward r_i for each generation
    3. Compute advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
    4. Compute GRPO loss: -mean(A_i * log_prob(y_i | x))
       with KL penalty: + beta * KL(pi || pi_ref)

References:
    - DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300
    - TRL GRPOTrainer implementation
"""
from __future__ import print_function

import torch
import torch.nn.functional as F


# ── Rollout ────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_batch(model, tokenizer, batch, num_generations=4,
                  max_new_tokens=512, temperature=0.9, device="cuda"):
    """Generate G completions per prompt in a batch.

    Args:
        model:           ShoulderSFTModel
        tokenizer:       HuggingFace tokenizer
        batch:           dict from grpo_collate_fn (images, input_ids, attention_mask, ...)
        num_generations: G (group size per prompt)
        max_new_tokens:  max tokens to generate
        temperature:     sampling temperature (>0 for diversity)
        device:          cuda device

    Returns:
        generations:  list[list[str]], shape [B, G] — decoded text only (no prompt)
        gen_input_ids: LongTensor [B*G, L+T] — full sequences incl. prompt + gen
        gen_attention_mask: BoolTensor [B*G, L+T]
        prompt_len:    int — length of prompt tokens (for masking prompt loss)
    """
    images = batch["images"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    B, L = input_ids.shape

    # Repeat each prompt G times
    images_rep = images.repeat_interleave(num_generations, dim=0)        # [B*G, ...]
    input_ids_rep = input_ids.repeat_interleave(num_generations, dim=0)  # [B*G, L]
    attn_rep = attention_mask.repeat_interleave(num_generations, dim=0)  # [B*G, L]

    gen_ids = model.generate(
        images=images_rep,
        input_ids=input_ids_rep,
        attention_mask=attn_rep,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )  # [B*G, L+T]

    # Decode only the generated portion
    generations = []
    for i in range(B):
        group = []
        for j in range(num_generations):
            idx = i * num_generations + j
            gen_tok = gen_ids[idx, L:]
            text = tokenizer.decode(gen_tok, skip_special_tokens=True)
            group.append(text)
        generations.append(group)

    # Build full attention mask for generated sequences
    full_len = gen_ids.shape[1]
    gen_attention_mask = (gen_ids != (tokenizer.pad_token_id or tokenizer.eos_token_id)).long()

    return generations, gen_ids, gen_attention_mask, L


# ── Advantage computation ──────────────────────────────────────────────────

def compute_advantages(rewards, num_generations, eps=1e-8):
    """Compute per-group normalized advantages.

    Args:
        rewards:         Tensor [B*G] flat rewards
        num_generations: G
        eps:             stability epsilon

    Returns:
        advantages: Tensor [B*G] normalized within each group
    """
    B_G = rewards.shape[0]
    B = B_G // num_generations
    rewards_grouped = rewards.view(B, num_generations)  # [B, G]

    mean = rewards_grouped.mean(dim=1, keepdim=True)  # [B, 1]
    std  = rewards_grouped.std(dim=1, keepdim=True)   # [B, 1]
    advantages = (rewards_grouped - mean) / (std + eps)  # [B, G]

    return advantages.view(B_G)  # [B*G]


# ── Log-probability computation ────────────────────────────────────────────

def compute_token_log_probs(model, images, gen_ids, gen_attn, prompt_len):
    """Forward pass to get log-probs of generated tokens.

    Args:
        model:       ShoulderSFTModel
        images:      [B*G, 5, 1, Z, H, W]
        gen_ids:     [B*G, L+T] full token IDs
        gen_attn:    [B*G, L+T] attention mask
        prompt_len:  L (number of prompt tokens to exclude from loss)

    Returns:
        log_probs:   Tensor [B*G] — mean log-prob over generated tokens
    """
    # Only compute log-probs over the generated portion
    labels = gen_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt

    outputs, _ = model(
        images=images,
        input_ids=gen_ids,
        attention_mask=gen_attn,
        labels=labels,
    )
    # outputs.logits: [B*G, L+T, V]
    # Compute per-token log-probs for the generated tokens
    logits = outputs.logits[:, prompt_len - 1:-1, :]  # [B*G, T, V]
    target = gen_ids[:, prompt_len:]                  # [B*G, T]
    attn_gen = gen_attn[:, prompt_len:]               # [B*G, T]

    log_probs_tok = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).view(logits.size(0), logits.size(1))  # [B*G, T]

    # Mean over valid (non-pad) generated tokens
    n_valid = attn_gen.float().sum(dim=1).clamp(min=1)
    log_probs = (log_probs_tok * attn_gen.float()).sum(dim=1) / n_valid  # [B*G]

    return log_probs


# ── GRPO loss ─────────────────────────────────────────────────────────────

def grpo_loss(log_probs, log_probs_ref, advantages, beta=0.01):
    """Compute GRPO policy loss with KL penalty.

    L_GRPO = -mean(A_i * log_prob_i) + beta * mean(KL_i)
    KL approximated as: log_prob_ref - log_prob  (reversed KL, token-level)

    Args:
        log_probs:     Tensor [B*G] current policy log-probs
        log_probs_ref: Tensor [B*G] reference policy log-probs (frozen SFT)
        advantages:    Tensor [B*G] normalized advantages
        beta:          KL penalty weight

    Returns:
        loss:     scalar Tensor
        loss_dict: dict with 'policy_loss', 'kl_loss', 'mean_reward'
    """
    policy_loss = -(advantages * log_probs).mean()
    kl = (log_probs_ref - log_probs).mean()
    loss = policy_loss + beta * kl

    return loss, {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl.item(),
        "total_loss": loss.item(),
    }
