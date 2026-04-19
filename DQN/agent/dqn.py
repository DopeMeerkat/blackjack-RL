"""Double DQN training step.

Implements the core learning algorithm described in blackjack_rl_design.md §7
and §15.3:

  Double DQN Bellman target (for playing-head transitions):
    a* = argmax_a [Q_online(s', a) + illegal_mask(s', a)]
    y  = r + γ * Q_target(s', a*)             (terminal: y = r)

  Bet-head transitions are one-step bandits (§15.4):
    y  = r    (done=True always; no next-state bootstrap)

Both heads share the trunk; gradients from both flow into the trunk.
The target network is updated via Polyak averaging after every training step.

Action masking in the Bellman target:
  illegal_mask(s, a) = 0 if a is legal, else −1e9.
  Applied additively to Q-values before argmax/value-read.
  This prevents the bootstrap from selecting or evaluating illegal actions
  even when the target network is stale.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.network import BlackjackNet
from agent.replay import PrioritizedReplayBuffer


class DQNAgent:
    """Wraps online + target networks and exposes a single training step.

    Args:
        net_config:      dict with network hyperparameters (merged state +
                         network sections of default.yaml).
        train_config:    dict with training hyperparameters.
        replay_capacity: Replay buffer size.
        device:          torch.device.
    """

    def __init__(
        self,
        net_config: dict,
        train_config: dict,
        replay_capacity: int,
        device: torch.device,
    ) -> None:
        self.device       = device
        self.gamma        = train_config["gamma"]
        self.tau          = train_config["target_tau"]
        self.batch_size   = train_config["batch_size"]
        self.grad_clip    = train_config.get("grad_clip", 10.0)
        self.obs_dim      = net_config.get("obs_dim", 28)

        # Online and target networks
        self.online_net = BlackjackNet(net_config).to(device)
        self.target_net = BlackjackNet(net_config).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.set_deterministic(True)  # target always deterministic

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=train_config["learning_rate"],
        )

        alpha = train_config.get("replay_alpha", 0.6)
        self.replay = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            obs_dim=self.obs_dim,
            alpha=alpha,
        )

        self._train_steps = 0

    # ------------------------------------------------------------------
    # Action selection (during environment rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_play_actions_batch(
        self,
        obs: np.ndarray,     # (K, 28)
        masks: np.ndarray,   # (K, 4) bool
    ) -> np.ndarray:
        """Select playing actions for a batch of envs.

        Resamples noise before inference.  Mask is applied additively
        (illegal actions set to −1e9).

        Returns: (K,) int32 action array.
        """
        obs_t  = torch.tensor(obs,   dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(masks, dtype=torch.bool,    device=self.device)

        self.online_net.reset_noise()
        play_q, _ = self.online_net(obs_t)
        play_q = play_q.clone()
        play_q[~mask_t] = -1e9
        return play_q.argmax(dim=1).cpu().numpy().astype(np.int32)

    @torch.no_grad()
    def select_bet_actions_batch(self, obs: np.ndarray) -> np.ndarray:
        """Select bet actions for a batch of envs.

        Returns: (K,) int32 bet-index array.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.online_net.reset_noise()
        _, bet_q = self.online_net(obs_t)
        return bet_q.argmax(dim=1).cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, beta: float) -> float | None:
        """One gradient update.

        Args:
            beta: IS correction exponent for PER (annealed from 0.4 → 1.0).

        Returns:
            Loss value (float), or None if the buffer is too small.
        """
        if len(self.replay) < self.batch_size:
            return None

        batch, weights, leaf_indices = self.replay.sample(self.batch_size, beta)

        obs       = torch.tensor(batch["obs"],      dtype=torch.float32, device=self.device)
        actions   = torch.tensor(batch["actions"],  dtype=torch.long,    device=self.device)
        rewards   = torch.tensor(batch["rewards"],  dtype=torch.float32, device=self.device)
        next_obs  = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones     = torch.tensor(batch["dones"],    dtype=torch.bool,    device=self.device)
        next_masks= torch.tensor(batch["next_masks"],dtype=torch.bool,   device=self.device)
        head_ids  = batch["head_ids"]    # numpy (B,) — stays on CPU for indexing
        weights_t = torch.tensor(weights,dtype=torch.float32,device=self.device)

        play_mask = (head_ids == 0)
        bet_mask  = (head_ids == 1)

        td_errors   = np.zeros(len(head_ids), dtype=np.float32)
        total_loss  = torch.zeros(1, device=self.device)

        # ----------------------------------------------------------------
        # Playing-head loss (Double DQN with action masking)
        # ----------------------------------------------------------------
        if play_mask.any():
            p_idx = np.where(play_mask)[0]

            p_obs       = obs[p_idx]
            p_actions   = actions[p_idx]
            p_rewards   = rewards[p_idx]
            p_next_obs  = next_obs[p_idx]
            p_dones     = dones[p_idx]
            p_next_masks= next_masks[p_idx]
            p_weights   = weights_t[p_idx]

            # Current Q-values (online net, noisy)
            self.online_net.reset_noise()
            self.online_net.set_deterministic(False)
            play_q_cur, _ = self.online_net(p_obs)
            q_current = play_q_cur.gather(1, p_actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Double DQN: online net picks the action (deterministic)
                self.online_net.set_deterministic(True)
                play_q_next_online, _ = self.online_net(p_next_obs)
                self.online_net.set_deterministic(False)

                play_q_next_online = play_q_next_online.clone()
                play_q_next_online[~p_next_masks] = -1e9
                a_star = play_q_next_online.argmax(dim=1)

                # Target net evaluates the chosen action (always deterministic)
                play_q_next_target, _ = self.target_net(p_next_obs)
                play_q_next_target = play_q_next_target.clone()
                play_q_next_target[~p_next_masks] = -1e9
                q_target_vals = play_q_next_target.gather(
                    1, a_star.unsqueeze(1)
                ).squeeze(1)

                # Bootstrap only for non-terminal transitions
                not_done_f = (~p_dones).float()
                q_targets = p_rewards + self.gamma * q_target_vals * not_done_f

            td_err_play = (q_targets - q_current).detach().cpu().numpy()

            # Normalize TD errors used for PER priorities only.
            # DOUBLE (action=2) yields 2× rewards, so |TD errors| are
            # ~2× larger, giving doubled transitions ~2× higher sampling rate.
            # Dividing by 2 equalizes priority without touching the loss.
            priority_td_play = td_err_play.copy()
            priority_td_play[p_actions.cpu().numpy() == 2] /= 2.0
            td_errors[p_idx] = priority_td_play

            play_loss = (
                p_weights * F.mse_loss(q_current, q_targets, reduction="none")
            ).mean()
            total_loss = total_loss + play_loss

        # ----------------------------------------------------------------
        # Bet-head loss (one-step bandit, §15.4 — no bootstrap)
        # ----------------------------------------------------------------
        if bet_mask.any():
            b_idx = np.where(bet_mask)[0]

            b_obs     = obs[b_idx]
            b_actions = actions[b_idx]
            b_rewards = rewards[b_idx]
            b_weights = weights_t[b_idx]

            # Current bet Q-values
            _, bet_q_cur = self.online_net(b_obs)
            q_current_bet = bet_q_cur.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            td_err_bet = (b_rewards - q_current_bet).detach().cpu().numpy()
            td_errors[b_idx] = td_err_bet

            bet_loss = (
                b_weights * F.mse_loss(q_current_bet, b_rewards, reduction="none")
            ).mean()
            total_loss = total_loss + bet_loss

        # ----------------------------------------------------------------
        # Optimiser step
        # ----------------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Update replay priorities with new TD errors
        self.replay.update_priorities(leaf_indices, td_errors)

        # Polyak update of target network
        self._polyak_update()

        self._train_steps += 1
        return float(total_loss.item())

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _polyak_update(self) -> None:
        """θ_target ← τ θ_online + (1−τ) θ_target."""
        tau = self.tau
        with torch.no_grad():
            for p_online, p_target in zip(
                self.online_net.parameters(),
                self.target_net.parameters(),
            ):
                p_target.data.mul_(1.0 - tau)
                p_target.data.add_(tau * p_online.data)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "train_steps":  self._train_steps,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.online_net.load_state_dict(sd["online_net"])
        self.target_net.load_state_dict(sd["target_net"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self._train_steps = sd.get("train_steps", 0)
