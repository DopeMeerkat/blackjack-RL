"""
deepcount/utils.py
-------------------
Small shared helpers used by both train.py and evaluate.py.
"""

from __future__ import annotations

import numpy as np
import torch

from env.blackjack_env import BlackjackShoeEnv, PHASE_PLAYING, DOUBLE, SPLIT


def obs_to_tensor(obs: dict, device: torch.device) -> dict:
    """Convert a single-step numpy obs dict to batched (B=1) tensors."""
    return {
        "shoe_history":  torch.tensor(obs["shoe_history"],    dtype=torch.long, device=device).unsqueeze(0),
        "hand":          torch.tensor(obs["hand"],            dtype=torch.long, device=device).unsqueeze(0),
        "hand_len":      torch.tensor([obs["hand_len"]],      dtype=torch.long, device=device),
        "dealer_upcard": torch.tensor([obs["dealer_upcard"]], dtype=torch.long, device=device),
        "phase":         torch.tensor([obs["phase"]],         dtype=torch.long, device=device),
    }


def build_action_mask(env: BlackjackShoeEnv, mask_complex: bool = False) -> np.ndarray:
    """
    Boolean (4,) mask — True where the play action is currently legal.
    During BETTING all play slots are True (the play head is masked by
    phase inside the network anyway).

    mask_complex: if True, additionally mask DOUBLE and SPLIT even when
                  legally available.  Used in Stage 1 so the value function
                  learns a clean HIT/STAND baseline (±1.0 reward scale
                  throughout) before the ±2.0 DOUBLE signal is introduced.
    """
    mask = np.ones(4, dtype=bool)
    if env._phase == PHASE_PLAYING:
        valid = env._valid_actions()
        for a in range(4):
            mask[a] = (a in valid)
        if mask_complex:
            mask[DOUBLE] = False
            mask[SPLIT]  = False
    return mask
