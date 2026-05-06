"""Blackjack Rainbow DQN play-agent network: trunk + dueling, distributional head.

Architecture:

  Trunk (standard Linear, no noise):
    Linear(27 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 256) → ReLU

  Dueling, distributional (C51) playing head:
    value_stream:     NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → n_atoms)
    advantage_stream: NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → n_play · n_atoms)
    logits(s,a,k) = V(s,k) + A(s,a,k) − mean_a A(s,a,k)
    p(s,a,·)      = softmax_k(logits)

  Atom support z_k ∈ [V_min, V_max] (inclusive), evenly spaced, stored as a
  buffer on the network.  Expected Q-value for action selection:
    Q(s,a) = Σ_k z_k · p(s,a,k)

When ``dueling=False`` (debug toggle), the head collapses to a single-stream
distributional layout (no V/A split) that still outputs n_actions·n_atoms
logits per sample.

The bet decision is handled by a separate agent (see ``agent/bet_agent.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.noisy_linear import NoisyLinear

# Default sizes (overridden by config dict)
_OBS_DIM        = 27
_TRUNK_HIDDEN   = 256
_HEAD_HIDDEN    = 256
_N_PLAY_ACTIONS = 4
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
    """Single-head Rainbow DQN network for the blackjack play agent."""

    def __init__(self, config: dict) -> None:
        super().__init__()

        obs_dim   = config.get("obs_dim",         _OBS_DIM)
        trunk_h   = config.get("trunk_hidden",    _TRUNK_HIDDEN)
        head_h    = config.get("head_hidden",     _HEAD_HIDDEN)
        n_play    = config.get("playing_actions", _N_PLAY_ACTIONS)
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

        self.register_buffer(
            "support", torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)
        )

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

    def forward_dist(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the play-action distribution (B, n_play, n_atoms)."""
        return self.play_head(self.trunk(obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values over the four play actions, shape (B, 4)."""
        play_dist = self.forward_dist(obs)
        return (play_dist * self.support).sum(dim=-1)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def set_deterministic(self, val: bool) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.set_deterministic(val)

    @torch.no_grad()
    def select_play_actions(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy action selection with illegal-action masking (expected Q)."""
        play_q = self.forward(obs).clone()
        play_q[~masks] = -1e9
        return play_q.argmax(dim=1)
