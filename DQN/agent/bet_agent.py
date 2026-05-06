"""Contextual bandit bet-sizing agent.

Architecture
------------
A small MLP regresses directly on realized hand returns:

    Linear(12 → 128) → ReLU → Linear(128 → 128) → ReLU → Linear(128 → 5)

There is no target network and no bootstrapping — the agent learns
by direct MSE regression on the terminal reward received for each hand.

    loss = mean((Q_pred(s, a_taken) − norm_reward)²)

where norm_reward = hand_reward / bet_multiplier (per-unit outcome ≈ ±1 std).

Exploration
-----------
ε-greedy with linear annealing:

    ε(t) = max(eps_end, eps_start − (eps_start − eps_end) * t / eps_decay_hands)

where t = total hands seen so far.  At ε = 1 the policy is fully random;
at ε = 0 it is fully greedy.  Evaluation always uses ε = 0.

Action selection (greedy)
-------------------------
Q-values are weighted by bet multipliers before argmax/comparison so that
the policy maximises expected absolute profit rather than per-unit EV:

    q_weighted[k] = Q(obs, k) * multiplier[k]
    greedy action  = argmax(q_weighted)

Replay
------
A simple uniform circular buffer (no PER).  Each entry stores
(obs, action, norm_reward) — a single-step bandit transition.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.bet_encoding import BET_OBS_DIM

_N_BET_ACTIONS   = 5
_HIDDEN          = 128
_LR              = 1e-4
_BATCH_SIZE      = 512
_CAPACITY        = 500_000
_EPS_START       = 1.0
_EPS_END         = 0.05
_EPS_DECAY_HANDS = 2_000_000


# ---------------------------------------------------------------------------
# Uniform replay buffer
# ---------------------------------------------------------------------------

class BetReplayBuffer:
    """Simple circular replay buffer for (obs, action, reward) tuples."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self._write   = 0
        self._n       = 0

    def add(self, obs: np.ndarray, action: int, reward: float) -> None:
        i = self._write
        self.obs[i]     = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self._write = (self._write + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self._n, size=batch_size)
        return {
            "obs":     self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
        }

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Bet agent
# ---------------------------------------------------------------------------

class BetAgent:
    """Contextual bandit for bet sizing with ε-greedy exploration.

    Args:
        config: Flat dict; relevant keys are documented under
                ``bet_agent`` in configs/default.yaml.
        device: torch.device to run inference and training on.
    """

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device      = device
        obs_dim          = int(config.get("bet_obs_dim",        BET_OBS_DIM))
        hidden           = int(config.get("bet_hidden",         _HIDDEN))
        n_actions        = int(config.get("bet_actions",        _N_BET_ACTIONS))
        lr               = float(config.get("bet_lr",           _LR))
        self.batch_size  = int(config.get("bet_batch_size",     _BATCH_SIZE))
        capacity         = int(config.get("bet_replay_cap",     _CAPACITY))
        self._eps_start  = float(config.get("bet_eps_start",    _EPS_START))
        self._eps_end    = float(config.get("bet_eps_end",      _EPS_END))
        self._eps_decay  = int(config.get("bet_eps_decay_hands", _EPS_DECAY_HANDS))

        mults = config.get("bet_multipliers", [1, 2, 4, 8, 12])
        self.multipliers = torch.tensor(mults, dtype=torch.float32, device=device)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(device)

        self.optimizer    = optim.Adam(self.net.parameters(), lr=lr)
        self.replay       = BetReplayBuffer(capacity, obs_dim)
        self._train_steps = 0
        self._hands_seen  = 0

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current exploration rate, linearly annealed from eps_start to eps_end."""
        if self._eps_decay <= 0:
            return self._eps_end
        t = min(self._hands_seen, self._eps_decay)
        return self._eps_end + (self._eps_start - self._eps_end) * (1.0 - t / self._eps_decay)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _weighted(self, q: torch.Tensor) -> torch.Tensor:
        """Scale raw Q-values by bet multipliers: q_weighted[k] = Q[k] * mult[k]."""
        return q * self.multipliers

    @torch.no_grad()
    def _greedy_actions(self, obs_t: torch.Tensor) -> np.ndarray:
        """Return argmax(Q × multiplier) for each row of obs_t."""
        return self._weighted(self.net(obs_t)).argmax(dim=1).cpu().numpy().astype(np.int32)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        """Greedy (ε=0) bet action for a single observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self._greedy_actions(obs_t)[0])

    @torch.no_grad()
    def select_actions_batch(
        self, obs: np.ndarray, greedy: bool = False
    ) -> np.ndarray:
        """Select bet actions for a batch of observations.

        Args:
            obs:    (K, obs_dim) float32 array.
            greedy: If True use ε=0 (eval); otherwise apply current ε schedule.

        Returns:
            (K,) int32 action array.
        """
        K = len(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = self._greedy_actions(obs_t)   # start with greedy for all

        if not greedy:
            eps = self.epsilon
            random_mask = np.random.random(K) < eps
            if random_mask.any():
                n_random = int(random_mask.sum())
                actions[random_mask] = np.random.randint(
                    0, len(self.multipliers), size=n_random, dtype=np.int32
                )

        return actions

    # ------------------------------------------------------------------
    # Replay insertion
    # ------------------------------------------------------------------

    def add_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
    ) -> None:
        """Store a completed hand's (bet_obs, bet_action, norm_reward)."""
        self.replay.add(obs, action, reward)
        self._hands_seen += 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """One MSE gradient update.  Returns loss, or None if buffer too small."""
        if len(self.replay) < self.batch_size:
            return None

        batch   = self.replay.sample(self.batch_size)
        obs_t   = torch.tensor(batch["obs"],     dtype=torch.float32, device=self.device)
        acts_t  = torch.tensor(batch["actions"], dtype=torch.long,    device=self.device)
        rews_t  = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)

        q_all  = self.net(obs_t)                                        # (B, n_actions)
        q_pred = q_all.gather(1, acts_t.unsqueeze(1)).squeeze(1)        # (B,)

        loss = F.mse_loss(q_pred, rews_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        return float(loss.item())

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "net":         self.net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "train_steps": self._train_steps,
            "hands_seen":  self._hands_seen,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd["net"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self._train_steps = sd.get("train_steps", 0)
        self._hands_seen  = sd.get("hands_seen",  0)
