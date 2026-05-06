"""Rainbow DQN training step (distributional C51 + Double DQN + n-step + PER)
for the play agent only.

Implements the core learning algorithm:

  Atoms z_k ∈ [V_min, V_max] evenly spaced.  The network outputs, for each
  (state, action), a probability distribution p(s, a, ·) over the atoms.
  Expected Q-value for action selection is E[Z] = Σ_k z_k · p(s, a, k).

  Double DQN + distributional Bellman target (n-step):
    a* = argmax_a E_z[p_online(s_{t+n}, a, ·)]   — masked to legal actions
    T z_k = clip(R_n + γ^n · (1 − done) · z_k, V_min, V_max)
    p_target = p_target_net(s_{t+n}, a*, ·)
    m = Π p_target onto {z_k}                    — categorical projection
    loss = −Σ_k m[k] · log p_online(s_t, a_t, k)  — cross-entropy (≡ KL + H)

The target network is updated via Polyak averaging after every training step.
The bet decision is made by a separate, simpler agent (see agent/bet_agent.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import BlackjackNet
from agent.replay import NStepAccumulator, PrioritizedReplayBuffer


class DQNAgent:
    """Wraps online + target networks and exposes a single training step.

    Args:
        net_config:      dict with network hyperparameters.
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
        self.obs_dim      = net_config.get("obs_dim", 27)
        self.n_step       = int(train_config.get("n_step", 1))
        self._gamma_n     = self.gamma ** self.n_step

        self.online_net = BlackjackNet(net_config).to(device)
        self.target_net = BlackjackNet(net_config).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.set_deterministic(True)

        self.n_atoms  = self.online_net.n_atoms
        self.v_min    = self.online_net.v_min
        self.v_max    = self.online_net.v_max
        self.delta_z  = self.online_net.delta_z
        self.support  = self.online_net.support

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

        self._nstep_accumulators: dict[int, NStepAccumulator] = {}
        self._train_steps = 0

    # ------------------------------------------------------------------
    # Replay insertion
    # ------------------------------------------------------------------

    def _accumulator(self, env_id: int) -> NStepAccumulator:
        acc = self._nstep_accumulators.get(env_id)
        if acc is None:
            acc = NStepAccumulator(self.n_step, self.gamma)
            self._nstep_accumulators[env_id] = acc
        return acc

    def add_play_transition(
        self,
        env_id: int,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        """Push a single-step playing transition through the n-step accumulator."""
        acc = self._accumulator(env_id)
        flushed = acc.push(dict(
            obs=obs, action=action, reward=reward, next_obs=next_obs,
            done=done, mask=mask, next_mask=next_mask,
        ))
        for t in flushed:
            self.replay.add(
                obs=t["obs"],
                action=t["action"],
                reward=t["reward"],
                next_obs=t["next_obs"],
                done=t["done"],
                mask=t["mask"],
                next_mask=t["next_mask"],
            )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_play_actions_batch(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Select playing actions for a batch of envs via expected Q.

        Returns: (K,) int32 action array.
        """
        obs_t  = torch.tensor(obs,   dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(masks, dtype=torch.bool,    device=self.device)

        self.online_net.reset_noise()
        play_q = self.online_net(obs_t).clone()
        play_q[~mask_t] = -1e9
        return play_q.argmax(dim=1).cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Categorical projection (C51)
    # ------------------------------------------------------------------

    def _project(
        self,
        rewards: torch.Tensor,
        dones:   torch.Tensor,
        target_dist: torch.Tensor,
        bootstrap_factor: float,
    ) -> torch.Tensor:
        """Project a bootstrapped target distribution onto the atom support."""
        B       = rewards.shape[0]
        n_atoms = self.n_atoms
        device  = rewards.device

        not_done = (~dones).float().unsqueeze(1)
        Tz = rewards.unsqueeze(1) + bootstrap_factor * not_done * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)

        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long().clamp_(0, n_atoms - 1)
        u = b.ceil().long().clamp_(0, n_atoms - 1)

        l_coef = (u.float() - b)
        u_coef = (b - l.float())
        eq = (l == u)
        l_coef = torch.where(eq, torch.ones_like(l_coef), l_coef)

        m = torch.zeros(B, n_atoms, device=device)
        offset = (torch.arange(B, device=device) * n_atoms).unsqueeze(1)
        m_flat = m.view(-1)
        m_flat.index_add_(0, (l + offset).view(-1), (target_dist * l_coef).view(-1))
        m_flat.index_add_(0, (u + offset).view(-1), (target_dist * u_coef).view(-1))
        return m

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, beta: float) -> float | None:
        """One gradient update.  Returns loss (float), or None if buffer too small."""
        if len(self.replay) < self.batch_size:
            return None

        batch, weights, leaf_indices = self.replay.sample(self.batch_size, beta)

        obs       = torch.tensor(batch["obs"],       dtype=torch.float32, device=self.device)
        actions   = torch.tensor(batch["actions"],   dtype=torch.long,    device=self.device)
        rewards   = torch.tensor(batch["rewards"],   dtype=torch.float32, device=self.device)
        next_obs  = torch.tensor(batch["next_obs"],  dtype=torch.float32, device=self.device)
        dones     = torch.tensor(batch["dones"],     dtype=torch.bool,    device=self.device)
        next_masks= torch.tensor(batch["next_masks"],dtype=torch.bool,    device=self.device)
        weights_t = torch.tensor(weights,            dtype=torch.float32, device=self.device)

        B       = obs.shape[0]
        n_atoms = self.n_atoms

        # Current distribution p_online(s, a_t, ·) — noisy
        self.online_net.reset_noise()
        self.online_net.set_deterministic(False)
        play_dist_cur = self.online_net.forward_dist(obs)              # (B, n_play, n_atoms)
        gather_idx    = actions.view(B, 1, 1).expand(-1, 1, n_atoms)
        p_online_sa   = play_dist_cur.gather(1, gather_idx).squeeze(1) # (B, n_atoms)

        with torch.no_grad():
            # Double DQN: action selection via online net (deterministic for stability)
            self.online_net.set_deterministic(True)
            play_q_next_online = self.online_net(next_obs).clone()
            self.online_net.set_deterministic(False)
            play_q_next_online[~next_masks] = -1e9
            a_star = play_q_next_online.argmax(dim=1)                  # (B,)

            # Target distribution at a*
            play_dist_next_target = self.target_net.forward_dist(next_obs)
            gather_star = a_star.view(B, 1, 1).expand(-1, 1, n_atoms)
            p_target_sa = play_dist_next_target.gather(1, gather_star).squeeze(1)

            m = self._project(
                rewards=rewards,
                dones=dones,
                target_dist=p_target_sa,
                bootstrap_factor=self._gamma_n,
            )

        log_p = torch.log(p_online_sa.clamp_min(1e-8))
        per_sample_loss = -(m * log_p).sum(dim=1)
        loss = (weights_t * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        td_errors = per_sample_loss.detach().cpu().numpy().astype(np.float32)
        self.replay.update_priorities(leaf_indices, td_errors)

        self._polyak_update()
        self._train_steps += 1
        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _polyak_update(self) -> None:
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
