"""Blackjack DQN network: shared trunk + playing head + bet-sizing head.

Architecture (blackjack_rl_design.md §6, extended with Rainbow dueling heads):

  Trunk:
    Linear(28 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 256) → ReLU

  Dueling playing head (V + A − mean(A)):
    value_stream:     NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → 1)
    advantage_stream: NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → 4)
    Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

  Dueling bet-sizing head:
    value_stream:     NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → 1)
    advantage_stream: NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → 5)

When `dueling=False` (debug toggle), each head collapses to the pre-Rainbow
single-stream layout used by the baseline DQN.

Usage:
  net = BlackjackNet(config)
  net.reset_noise()                   # before each training forward
  play_q, bet_q = net(obs)            # obs: (B, 28)

  net.set_deterministic(True)         # for eval / Bellman target computation
  play_q, bet_q = net(obs)
  net.set_deterministic(False)        # restore stochastic mode
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agent.noisy_linear import NoisyLinear

# Default sizes (overridden by config dict)
_OBS_DIM        = 28
_TRUNK_HIDDEN   = 256
_HEAD_HIDDEN    = 256
_N_PLAY_ACTIONS = 4
_N_BET_ACTIONS  = 5
_NOISY_SIGMA0   = 0.5
_DUELING        = True


class _DuelingHead(nn.Module):
    """Dueling head: value stream (scalar) + advantage stream (per-action)."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        n_actions: int,
        sigma0: float,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.value_stream = nn.Sequential(
            NoisyLinear(in_features, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, 1, sigma0),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(in_features, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, n_actions, sigma0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value     = self.value_stream(features)               # (B, 1)
        advantage = self.advantage_stream(features)           # (B, n_actions)
        # Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class _VanillaHead(nn.Module):
    """Pre-Rainbow single-stream head (for dueling=False debug mode)."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        n_actions: int,
        sigma0: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            NoisyLinear(in_features, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, n_actions, sigma0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class BlackjackNet(nn.Module):
    """Dual-head DQN network for blackjack with card counting.

    Args:
        config: dict with keys from configs/default.yaml (merged from
                ``state`` and ``network`` sections).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        obs_dim   = config.get("obs_dim",         _OBS_DIM)
        trunk_h   = config.get("trunk_hidden",    _TRUNK_HIDDEN)
        head_h    = config.get("head_hidden",     _HEAD_HIDDEN)
        n_play    = config.get("playing_actions", _N_PLAY_ACTIONS)
        n_bet     = config.get("bet_actions",     _N_BET_ACTIONS)
        sigma0    = config.get("noisy_sigma0",    _NOISY_SIGMA0)
        dueling   = bool(config.get("dueling",    _DUELING))
        self.dueling = dueling

        # --- Trunk (standard Linear, no noise) ---
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, trunk_h),
            nn.ReLU(),
            nn.Linear(trunk_h, trunk_h),
            nn.ReLU(),
            nn.Linear(trunk_h, trunk_h),
            nn.ReLU(),
        )

        head_cls = _DuelingHead if dueling else _VanillaHead
        self.play_head = head_cls(trunk_h, head_h, n_play, sigma0)
        self.bet_head  = head_cls(trunk_h, head_h, n_bet,  sigma0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (play_q, bet_q) for a batch of observations.

        Args:
            obs: float32 tensor of shape (B, 28).

        Returns:
            play_q: (B, 4) — Q-values for the four playing actions.
            bet_q:  (B, 5) — Q-values for the five bet multipliers.
        """
        features = self.trunk(obs)
        play_q   = self.play_head(features)
        bet_q    = self.bet_head(features)
        return play_q, bet_q

    # ------------------------------------------------------------------
    # Noise helpers (delegating to NoisyLinear instances)
    # ------------------------------------------------------------------

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers.

        Call once before each forward pass during training rollouts.
        The same noise sample is used for both the action-selection query
        and the Q-value computation in the training step.
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def set_deterministic(self, val: bool) -> None:
        """Toggle deterministic mode in all NoisyLinear layers.

        True  → use mean weights (μ_W, μ_b) only; suitable for eval and
                for the Bellman target computation in training.
        False → use noisy weights (requires reset_noise() before use).
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.set_deterministic(val)

    # ------------------------------------------------------------------
    # Convenience: masked action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_play_actions(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy action selection with illegal-action masking.

        Args:
            obs:   (B, 28) float32 — observation batch.
            masks: (B, 4)  bool   — True where action is legal.

        Returns:
            actions: (B,) long — argmax of masked Q-values.
        """
        play_q, _ = self.forward(obs)
        play_q = play_q.clone()
        play_q[~masks] = -1e9
        return play_q.argmax(dim=1)

    @torch.no_grad()
    def select_bet_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy bet-size selection (no masking — all bets are always legal).

        Args:
            obs: (B, 28) float32.

        Returns:
            actions: (B,) long — argmax of bet Q-values.
        """
        _, bet_q = self.forward(obs)
        return bet_q.argmax(dim=1)
