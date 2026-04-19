"""Blackjack Rainbow DQN network: shared trunk + dueling, distributional heads.

Architecture (blackjack_rl_design.md §6, extended with Rainbow):

  Trunk (standard Linear, no noise):
    Linear(28 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 256) → ReLU

  Dueling, distributional (C51) playing head:
    value_stream:     NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → n_atoms)
    advantage_stream: NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → n_play · n_atoms)
    logits(s,a,k) = V(s,k) + A(s,a,k) − mean_a A(s,a,k)
    p(s,a,·)      = softmax_k(logits)

  Dueling, distributional bet-sizing head (n_bet outputs per atom, analogous).

  Atom support z_k ∈ [V_min, V_max] (inclusive), evenly spaced, stored as a
  buffer on the network.  Expected Q-value for action selection:
    Q(s,a) = Σ_k z_k · p(s,a,k)

When ``dueling=False`` (debug toggle), each head collapses to a single-stream
distributional layout (no V/A split) that still outputs n_actions·n_atoms
logits per sample.

Usage:
  net = BlackjackNet(config)
  net.reset_noise()
  play_q, bet_q   = net(obs)             # expected Q, shapes (B, 4) / (B, 5)
  play_d, bet_d   = net.forward_dist(obs) # raw dists, shapes (B, 4, A) / (B, 5, A)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.noisy_linear import NoisyLinear

# Default sizes (overridden by config dict)
_OBS_DIM        = 28
_TRUNK_HIDDEN   = 256
_HEAD_HIDDEN    = 256
_N_PLAY_ACTIONS = 4
_N_BET_ACTIONS  = 5
_NOISY_SIGMA0   = 0.5
_DUELING        = True
_N_ATOMS        = 51
_V_MIN          = -15.0
_V_MAX          =  15.0


class _DuelingDistHead(nn.Module):
    """Dueling distributional head: V(s, k) + A(s, a, k) − mean_a A(s, a, k)."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        n_actions: int,
        n_atoms: int,
        sigma0: float,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.value_stream = nn.Sequential(
            NoisyLinear(in_features, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, n_atoms, sigma0),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(in_features, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, n_actions * n_atoms, sigma0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        v = self.value_stream(features)                          # (B, n_atoms)
        a = self.advantage_stream(features).view(
            B, self.n_actions, self.n_atoms
        )
        logits = v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1)                         # (B, n_actions, n_atoms)


class _VanillaDistHead(nn.Module):
    """Single-stream distributional head (dueling disabled)."""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        n_actions: int,
        n_atoms: int,
        sigma0: float,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms
        self.net = nn.Sequential(
            NoisyLinear(in_features, hidden, sigma0),
            nn.ReLU(),
            NoisyLinear(hidden, n_actions * n_atoms, sigma0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        logits = self.net(features).view(B, self.n_actions, self.n_atoms)
        return F.softmax(logits, dim=-1)


class BlackjackNet(nn.Module):
    """Dual-head Rainbow DQN network for blackjack with card counting.

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
        n_atoms   = int(config.get("n_atoms",     _N_ATOMS))
        v_min     = float(config.get("v_min",     _V_MIN))
        v_max     = float(config.get("v_max",     _V_MAX))

        assert n_atoms >= 2, "n_atoms must be >= 2"
        assert v_max > v_min, "v_max must be > v_min"

        self.dueling  = dueling
        self.n_atoms  = n_atoms
        self.v_min    = v_min
        self.v_max    = v_max
        self.delta_z  = (v_max - v_min) / (n_atoms - 1)

        # Atom support: z_k ∈ [v_min, v_max], evenly spaced. Buffer → moves with .to(device).
        self.register_buffer(
            "support", torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)
        )

        # --- Trunk (standard Linear, no noise) ---
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, trunk_h),
            nn.ReLU(),
            nn.Linear(trunk_h, trunk_h),
            nn.ReLU(),
            nn.Linear(trunk_h, trunk_h),
            nn.ReLU(),
        )

        head_cls = _DuelingDistHead if dueling else _VanillaDistHead
        self.play_head = head_cls(trunk_h, head_h, n_play, n_atoms, sigma0)
        self.bet_head  = head_cls(trunk_h, head_h, n_bet,  n_atoms, sigma0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_dist(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw action distributions.

        Returns:
            play_dist: (B, n_play, n_atoms) — softmax-normalised probabilities.
            bet_dist:  (B, n_bet,  n_atoms).
        """
        features = self.trunk(obs)
        return self.play_head(features), self.bet_head(features)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expected Q-values E_z[p(s,a,·)] derived from the atom support.

        Args:
            obs: float32 tensor of shape (B, 28).

        Returns:
            play_q: (B, 4) — expected Q-values over the four playing actions.
            bet_q:  (B, 5) — expected Q-values over the five bet multipliers.
        """
        play_dist, bet_dist = self.forward_dist(obs)
        play_q = (play_dist * self.support).sum(dim=-1)
        bet_q  = (bet_dist  * self.support).sum(dim=-1)
        return play_q, bet_q

    # ------------------------------------------------------------------
    # Noise helpers (delegating to NoisyLinear instances)
    # ------------------------------------------------------------------

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def set_deterministic(self, val: bool) -> None:
        """Toggle deterministic mode in all NoisyLinear layers."""
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
        """Greedy action selection with illegal-action masking (expected Q)."""
        play_q, _ = self.forward(obs)
        play_q = play_q.clone()
        play_q[~masks] = -1e9
        return play_q.argmax(dim=1)

    @torch.no_grad()
    def select_bet_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy bet-size selection (no masking — all bets are always legal)."""
        _, bet_q = self.forward(obs)
        return bet_q.argmax(dim=1)
