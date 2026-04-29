"""
GRPO (Group Relative Policy Optimization) for shoulder MRI structured diagnosis.

Builds on top of the SFT-trained model and uses task-specific reward functions
to reinforce accurate structured JSON output.

Modules:
    reward_functions  -- per-task reward signals (label F1, grounding IoU, etc.)
    grpo_dataset      -- dataset wrapper for GRPO rollouts
    grpo_utils        -- rollout, advantage computation, GRPO loss
    train_grpo        -- main GRPO training script
"""
