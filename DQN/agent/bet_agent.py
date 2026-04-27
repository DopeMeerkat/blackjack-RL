"""Contextual bandit bet-sizing agent.

Architecture
------------
A small MLP regresses directly on realized hand returns:

    Linear(12 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 5)

There is no target network and no bootstrapping — the agent learns
by direct MSE regression on the terminal reward received for each hand.

    loss = mean((Q_pred(s, a_taken) − reward)²)

Exploration
-----------
During training the agent samples from a softmax over the predicted
Q-values divided by a temperature τ:

    P(a | s) ∝ exp(Q(s, a) / τ)

A high τ → nearly uniform bets (full exploration); τ → 0 → greedy.
Evaluation always uses argmax (τ = 0).

Replay
------
A simple uniform circular buffer (no PER).  Each entry stores
(obs, action, reward) — a single-step bandit transition.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.bet_encoding import BET_OBS_DIM

_N_BET_ACTIONS = 5
_HIDDEN        = 64
_LR            = 1e-3
_BATCH_SIZE    = 256
_CAPACITY      = 100_000
_TEMPERATURE   = 1.0


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
    """Contextual bandit for bet sizing.

    Args:
        config: Flat dict; relevant keys are documented under
                ``bet_agent`` in configs/default.yaml.
        device: torch.device to run inference and training on.
    """

    def __init__(self, config: dict, device: torch.device) -> None:
        self.device      = device
        obs_dim          = int(config.get("bet_obs_dim",      BET_OBS_DIM))
        hidden           = int(config.get("bet_hidden",       _HIDDEN))
        n_actions        = int(config.get("bet_actions",      _N_BET_ACTIONS))
        lr               = float(config.get("bet_lr",         _LR))
        self.batch_size  = int(config.get("bet_batch_size",   _BATCH_SIZE))
        capacity         = int(config.get("bet_replay_cap",   _CAPACITY))
        self.temperature = float(config.get("bet_temperature",_TEMPERATURE))

        # Multiplier weights applied to Q-values before softmax/argmax so that
        # the policy correctly reflects E[profit] = Q(obs, k) * multiplier[k].
        # Q-values estimate per-unit EV; multiplying by the actual stake gives
        # the expected absolute profit for each bet size.
        mults = config.get("bet_multipliers", [1, 2, 4, 8, 12])
        self.multipliers = torch.tensor(mults, dtype=torch.float32, device=device)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.replay    = BetReplayBuffer(capacity, obs_dim)
        self._train_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _weighted(self, q: torch.Tensor) -> torch.Tensor:
        """Scale raw Q-values by bet multipliers: q_weighted[k] = Q[k] * mult[k].

        The network outputs per-unit EV estimates.  Multiplying by the stake
        converts these to expected absolute profit, which is the correct signal
        for both softmax exploration and greedy selection.
        """
        return q * self.multipliers

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> int:
        """Sample a bet action using softmax-temperature exploration."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_w = self._weighted(self.net(obs_t).squeeze(0))
        probs = F.softmax(q_w / self.temperature, dim=0).cpu().numpy()
        return int(np.random.choice(len(probs), p=probs))

    @torch.no_grad()
    def select_action_greedy(self, obs: np.ndarray) -> int:
        """Argmax bet action (deterministic evaluation)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self._weighted(self.net(obs_t).squeeze(0)).argmax().item())

    @torch.no_grad()
    def select_actions_batch(
        self, obs: np.ndarray, greedy: bool = False
    ) -> np.ndarray:
        """Select bet actions for a batch of observations.

        Args:
            obs:    (K, obs_dim) float32 array.
            greedy: If True, use argmax; otherwise softmax-temperature.

        Returns:
            (K,) int32 action array.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        q_w = self._weighted(self.net(obs_t))
        if greedy:
            return q_w.argmax(dim=1).cpu().numpy().astype(np.int32)
        probs = F.softmax(q_w / self.temperature, dim=1)
        return torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Replay insertion
    # ------------------------------------------------------------------

    def add_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
    ) -> None:
        """Store a completed hand's (bet_obs, bet_action, hand_reward)."""
        self.replay.add(obs, action, reward)

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

        q_all  = self.net(obs_t)                               # (B, n_actions)
        q_pred = q_all.gather(1, acts_t.unsqueeze(1)).squeeze(1)  # (B,)

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
        }

    def load_state_dict(self, sd: dict) -> None:
        self.net.load_state_dict(sd["net"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self._train_steps = sd.get("train_steps", 0)
