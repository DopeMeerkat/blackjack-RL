"""
Rollout Buffer
==============
Stores transitions for PPO updates with dual advantage streams
and correct multi-env GAE.

Multi-env GAE
-------------
When n_envs > 1 the buffer holds n_steps * n_envs transitions laid out
in step-major order:

    [step0_env0, step0_env1, ..., step0_env{N-1},
     step1_env0, step1_env1, ...,
     ...]

compute_returns_and_advantages reshapes this to (n_steps, n_envs),
computes GAE independently per env column, then flattens back.
This correctly handles the fact that env_i's terminal step must never
bootstrap into env_j's first step.

Dual reward streams
-------------------
  Play stream  — bet-normalised rewards (reward / bet), stationary target.
  Bet stream   — raw chip rewards, PopArt-lite normalised by running std.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """
    Args:
        n_steps     : number of environment steps per env per rollout
        n_envs      : number of parallel environments (default 1)
        gamma       : discount factor (default 0.99)
        gae_lambda  : GAE λ (default 0.95)
        device      : torch device
    """
    n_steps:    int
    n_envs:     int          = 1
    gamma:      float        = 0.99
    gae_lambda: float        = 0.95
    device:     torch.device = torch.device("cpu")

    # ── Storage (flat, length = n_steps * n_envs) ─────────────────────────
    shoe_history:   np.ndarray = field(init=False)
    hand:           np.ndarray = field(init=False)
    hand_len:       np.ndarray = field(init=False)
    dealer_upcard:  np.ndarray = field(init=False)
    phase:          np.ndarray = field(init=False)
    action_mask:    np.ndarray = field(init=False)

    bet_action:     np.ndarray = field(init=False)
    play_action:    np.ndarray = field(init=False)

    log_probs:      np.ndarray = field(init=False)
    dones:          np.ndarray = field(init=False)
    true_counts:    np.ndarray = field(init=False)

    # Play stream (bet-normalised)
    rewards_norm:    np.ndarray = field(init=False)
    values_norm:     np.ndarray = field(init=False)
    returns_norm:    np.ndarray = field(init=False)
    advantages_norm: np.ndarray = field(init=False)

    # Bet stream (raw chips, PopArt-lite)
    rewards_raw:    np.ndarray = field(init=False)
    values_raw:     np.ndarray = field(init=False)
    returns_raw:    np.ndarray = field(init=False)
    advantages_raw: np.ndarray = field(init=False)

    # Running return std for PopArt-lite (bet stream)
    _raw_return_std: float = field(init=False, default=1.0)

    _pos:  int  = field(init=False, default=0)
    _full: bool = field(init=False, default=False)

    def __post_init__(self):
        C = self.n_steps * self.n_envs
        from env.blackjack_env import MAX_HAND, HISTORY_WINDOW

        self.shoe_history  = np.zeros((C, HISTORY_WINDOW), dtype=np.int8)
        self.hand          = np.zeros((C, MAX_HAND),        dtype=np.int8)
        self.hand_len      = np.zeros(C,                    dtype=np.int8)
        self.dealer_upcard = np.zeros(C,                    dtype=np.int8)
        self.phase         = np.zeros(C,                    dtype=np.int8)
        self.action_mask   = np.ones((C, 4),                dtype=bool)
        self.bet_action    = np.zeros((C, 1),  dtype=np.float32)
        self.play_action   = np.zeros(C,       dtype=np.int64)
        self.log_probs     = np.zeros(C,       dtype=np.float32)
        self.dones         = np.zeros(C,       dtype=bool)
        self.true_counts   = np.zeros(C,       dtype=np.float32)

        for attr in (
            "rewards_norm", "values_norm", "returns_norm",  "advantages_norm",
            "rewards_raw",  "values_raw",  "returns_raw",   "advantages_raw",
        ):
            setattr(self, attr, np.zeros(C, dtype=np.float32))

    @property
    def capacity(self) -> int:
        return self.n_steps * self.n_envs

    def reset(self):
        self._pos  = 0
        self._full = False

    def add(
        self,
        obs:         dict,
        action:      dict,
        log_prob:    float,
        reward:      float,
        value_norm:  float,
        value_raw:   float,
        done:        bool,
        true_count:  float,
        action_mask: np.ndarray,
        current_bet: float,
    ):
        """Store a single transition. Call once per env per step."""
        p = self._pos
        self.shoe_history[p]  = obs["shoe_history"]
        self.hand[p]          = obs["hand"]
        self.hand_len[p]      = obs["hand_len"]
        self.dealer_upcard[p] = obs["dealer_upcard"]
        self.phase[p]         = obs["phase"]
        self.action_mask[p]   = action_mask
        self.bet_action[p]    = action["bet"]
        self.play_action[p]   = action["play"]
        self.log_probs[p]     = log_prob
        self.dones[p]         = done
        self.true_counts[p]   = true_count

        bet_scale             = max(float(current_bet), 1.0)
        self.rewards_norm[p]  = reward / bet_scale
        self.values_norm[p]   = value_norm
        self.rewards_raw[p]   = reward
        self.values_raw[p]    = value_raw

        self._pos += 1
        if self._pos >= self.capacity:
            self._full = True

    def compute_returns_and_advantages(
        self,
        last_values_norm: np.ndarray,   # (n_envs,) — bootstrap V_play per env
        last_values_raw:  np.ndarray,   # (n_envs,) — bootstrap V_bet  per env
    ):
        """
        GAE computed independently per env.

        Data layout is step-major: transitions are added as
        [step0_env0, step0_env1, ..., step1_env0, ...].
        We reshape to (n_steps, n_envs), run GAE per column, then flatten.
        """
        T, N = self.n_steps, self.n_envs
        self._gae(
            self.rewards_norm, self.values_norm,
            self.returns_norm, self.advantages_norm,
            last_values_norm, T, N,
        )
        self._gae(
            self.rewards_raw, self.values_raw,
            self.returns_raw, self.advantages_raw,
            last_values_raw, T, N,
        )

        # Update running std of raw returns (exponential moving average)
        raw_std = float(np.std(self.returns_raw)) + 1e-8
        self._raw_return_std = 0.9 * self._raw_return_std + 0.1 * raw_std

    def _gae(
        self,
        rewards_flat:    np.ndarray,   # (T*N,)
        values_flat:     np.ndarray,   # (T*N,)
        returns_flat:    np.ndarray,   # (T*N,) — written in place
        advantages_flat: np.ndarray,   # (T*N,) — written in place
        last_values:     np.ndarray,   # (N,)
        T: int,
        N: int,
    ):
        # Reshape to (T, N) — step-major layout
        rewards    = rewards_flat.reshape(T, N)
        values     = values_flat.reshape(T, N)
        dones      = self.dones.reshape(T, N)
        advantages = np.zeros((T, N), dtype=np.float32)

        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nt       = 1.0 - dones[t].astype(np.float32)      # (N,)
            next_val = last_values if t == T - 1 else values[t + 1]
            delta    = rewards[t] + self.gamma * next_val * nt - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nt * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        np.copyto(advantages_flat, advantages.flatten())
        np.copyto(returns_flat,    returns.flatten())

    def get_batches(self, batch_size: int):
        """Yield random mini-batches of tensors."""
        assert self._full, "Buffer must be full before sampling."
        indices = np.random.permutation(self.capacity)
        D = self.device
        rrs = self._raw_return_std   # snapshot for this update

        for start in range(0, self.capacity, batch_size):
            idx = indices[start: start + batch_size]

            # Normalise advantages per-phase so that BETTING and PLAYING
            # steps share consistent gradient scales when combined.
            phase_idx = self.phase[idx]
            is_bet = (phase_idx == 0)   # PHASE_BETTING == 0
            is_play = ~is_bet

            adv_n = self.advantages_norm[idx].copy()
            if is_play.any():
                mu, sigma = adv_n[is_play].mean(), adv_n[is_play].std() + 1e-8
                adv_n[is_play] = (adv_n[is_play] - mu) / sigma
            if is_bet.any():
                mu, sigma = adv_n[is_bet].mean(), adv_n[is_bet].std() + 1e-8
                adv_n[is_bet] = (adv_n[is_bet] - mu) / sigma

            adv_r = self.advantages_raw[idx] / rrs
            adv_r = adv_r.copy()
            if is_play.any():
                mu, sigma = adv_r[is_play].mean(), adv_r[is_play].std() + 1e-8
                adv_r[is_play] = (adv_r[is_play] - mu) / sigma
            if is_bet.any():
                mu, sigma = adv_r[is_bet].mean(), adv_r[is_bet].std() + 1e-8
                adv_r[is_bet] = (adv_r[is_bet] - mu) / sigma

            obs_batch = {
                "shoe_history":  torch.tensor(self.shoe_history[idx],  dtype=torch.long,  device=D),
                "hand":          torch.tensor(self.hand[idx],          dtype=torch.long,  device=D),
                "hand_len":      torch.tensor(self.hand_len[idx],      dtype=torch.long,  device=D),
                "dealer_upcard": torch.tensor(self.dealer_upcard[idx], dtype=torch.long,  device=D),
                "phase":         torch.tensor(self.phase[idx],         dtype=torch.long,  device=D),
            }
            yield {
                "obs":             obs_batch,
                "actions": {
                    "bet":  torch.tensor(self.bet_action[idx],  dtype=torch.float32, device=D),
                    "play": torch.tensor(self.play_action[idx], dtype=torch.long,    device=D),
                },
                "log_probs":       torch.tensor(self.log_probs[idx],       dtype=torch.float32, device=D),
                "returns_norm":    torch.tensor(self.returns_norm[idx],    dtype=torch.float32, device=D),
                "returns_raw":     torch.tensor(self.returns_raw[idx] / rrs, dtype=torch.float32, device=D),
                "advantages_norm": torch.tensor(adv_n,                     dtype=torch.float32, device=D),
                "advantages_raw":  torch.tensor(adv_r,                     dtype=torch.float32, device=D),
                "true_counts":     torch.tensor(self.true_counts[idx],     dtype=torch.float32, device=D),
                "action_mask":     torch.tensor(self.action_mask[idx],     dtype=torch.bool,    device=D),
                "phase":           torch.tensor(self.phase[idx],           dtype=torch.long,    device=D),
            }
