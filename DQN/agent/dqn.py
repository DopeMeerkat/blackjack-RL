"""Rainbow DQN training step (distributional C51 + Double DQN + n-step + PER).

Implements the core learning algorithm described in blackjack_rl_design.md §7
and §15.3, extended with Rainbow:

  Atoms z_k ∈ [V_min, V_max] evenly spaced.  The network outputs, for each
  (state, action), a probability distribution p(s, a, ·) over the atoms.
  Expected Q-value for action selection is E[Z] = Σ_k z_k · p(s, a, k).

  Double DQN + distributional Bellman target (play head, n-step):
    a* = argmax_a E_z[p_online(s_{t+n}, a, ·)]   — masked to legal actions
    T z_k = clip(R_n + γ^n · (1 − done) · z_k, V_min, V_max)
    p_target = p_target_net(s_{t+n}, a*, ·)
    m = Π p_target onto {z_k}                    — categorical projection
    loss = −Σ_k m[k] · log p_online(s_t, a_t, k)  — cross-entropy (≡ KL + H)

  Bet head (one-step bandit, §15.4): same projection with bootstrap = 0, i.e.
  the target distribution collapses to a delta at clip(R, V_min, V_max).

Both heads share the trunk; gradients from both flow into the trunk.
The target network is updated via Polyak averaging after every training step.
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
        self.n_step       = int(train_config.get("n_step", 1))
        # γ^n_step is the effective bootstrap discount for n-step targets.
        self._gamma_n     = self.gamma ** self.n_step

        # Online and target networks
        self.online_net = BlackjackNet(net_config).to(device)
        self.target_net = BlackjackNet(net_config).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.set_deterministic(True)  # target always deterministic

        # Distributional (C51) parameters mirror the network.
        self.n_atoms  = self.online_net.n_atoms
        self.v_min    = self.online_net.v_min
        self.v_max    = self.online_net.v_max
        self.delta_z  = self.online_net.delta_z
        # Cache support on device for projection arithmetic.
        self.support  = self.online_net.support  # (n_atoms,)

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

        # Per-env n-step accumulators for play-head transitions (lazy-created).
        self._nstep_accumulators: dict[int, NStepAccumulator] = {}

        self._train_steps = 0

    # ------------------------------------------------------------------
    # Replay insertion helpers (n-step for play head, direct for bet head)
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
        """Push a single-step playing transition through the n-step accumulator.

        The accumulator emits n-step transitions into the PER buffer when it
        has collected ``n_step`` steps or on a terminal flush.
        """
        acc = self._accumulator(env_id)
        flushed = acc.push(dict(
            obs=obs, action=action, reward=reward, next_obs=next_obs,
            done=done, mask=mask, next_mask=next_mask, head_id=0,
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
                head_id=t["head_id"],
            )

    def add_bet_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        """Insert a one-step bandit transition for the bet head (always done)."""
        self.replay.add(
            obs=obs, action=action, reward=reward,
            next_obs=next_obs, done=True,
            mask=mask, next_mask=next_mask, head_id=1,
        )

    # ------------------------------------------------------------------
    # Action selection (during environment rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_play_actions_batch(
        self,
        obs: np.ndarray,     # (K, 28)
        masks: np.ndarray,   # (K, 4) bool
    ) -> np.ndarray:
        """Select playing actions for a batch of envs via expected Q.

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
        """Select bet actions for a batch of envs via expected Q.

        Returns: (K,) int32 bet-index array.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.online_net.reset_noise()
        _, bet_q = self.online_net(obs_t)
        return bet_q.argmax(dim=1).cpu().numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # Categorical projection (C51)
    # ------------------------------------------------------------------

    def _project(
        self,
        rewards: torch.Tensor,      # (B,)
        dones:   torch.Tensor,      # (B,) bool
        target_dist: torch.Tensor,  # (B, n_atoms)
        bootstrap_factor: float,    # γ^n for play head, 0 for bet head
    ) -> torch.Tensor:
        """Project a bootstrapped target distribution onto the atom support.

        For each sample i and source-atom j:
          Tz_{i,j} = clip(rewards[i] + bootstrap_factor · (1−done_i) · z_j, V_min, V_max)
        Mass ``target_dist[i, j]`` is split between floor(b) and ceil(b), where
        b = (Tz_{i,j} − V_min) / Δz.  When b is an integer, all mass goes to
        that atom.

        Returns:
            m: (B, n_atoms) projected target distribution (sums to 1 per row).
        """
        B       = rewards.shape[0]
        n_atoms = self.n_atoms
        device  = rewards.device

        not_done = (~dones).float().unsqueeze(1)                  # (B, 1)
        Tz = rewards.unsqueeze(1) + bootstrap_factor * not_done * self.support.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)                     # (B, n_atoms)

        b = (Tz - self.v_min) / self.delta_z                      # (B, n_atoms)
        l = b.floor().long().clamp_(0, n_atoms - 1)
        u = b.ceil().long().clamp_(0, n_atoms - 1)

        l_coef = (u.float() - b)
        u_coef = (b - l.float())
        # When Tz lands exactly on an atom (l == u), both coefs are 0.
        # Force the lower (==upper) atom to receive the full mass.
        eq = (l == u)
        l_coef = torch.where(eq, torch.ones_like(l_coef), l_coef)

        m = torch.zeros(B, n_atoms, device=device)
        offset = (torch.arange(B, device=device) * n_atoms).unsqueeze(1)  # (B, 1)
        m_flat = m.view(-1)
        m_flat.index_add_(0, (l + offset).view(-1), (target_dist * l_coef).view(-1))
        m_flat.index_add_(0, (u + offset).view(-1), (target_dist * u_coef).view(-1))
        return m

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
        n_atoms     = self.n_atoms

        # ----------------------------------------------------------------
        # Play-head distributional loss (Double DQN + n-step + C51)
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
            B_p         = p_obs.shape[0]

            # Current distribution p_online(s, a_t, ·) — noisy
            self.online_net.reset_noise()
            self.online_net.set_deterministic(False)
            play_dist_cur, _ = self.online_net.forward_dist(p_obs)    # (B_p, n_play, n_atoms)
            gather_idx = p_actions.view(B_p, 1, 1).expand(-1, 1, n_atoms)
            p_online_sa = play_dist_cur.gather(1, gather_idx).squeeze(1)   # (B_p, n_atoms)

            with torch.no_grad():
                # Double DQN action selection via online net (deterministic)
                self.online_net.set_deterministic(True)
                play_q_next_online, _ = self.online_net(p_next_obs)   # expected Q
                self.online_net.set_deterministic(False)
                play_q_next_online = play_q_next_online.clone()
                play_q_next_online[~p_next_masks] = -1e9
                a_star = play_q_next_online.argmax(dim=1)             # (B_p,)

                # Target distribution at a* (target net always deterministic)
                play_dist_next_target, _ = self.target_net.forward_dist(p_next_obs)
                gather_star = a_star.view(B_p, 1, 1).expand(-1, 1, n_atoms)
                p_target_sa = play_dist_next_target.gather(1, gather_star).squeeze(1)

                m_play = self._project(
                    rewards=p_rewards,
                    dones=p_dones,
                    target_dist=p_target_sa,
                    bootstrap_factor=self._gamma_n,
                )                                                      # (B_p, n_atoms)

            # Cross-entropy loss −Σ_k m_k · log p_online_k
            log_p = torch.log(p_online_sa.clamp_min(1e-8))
            per_sample_loss = -(m_play * log_p).sum(dim=1)             # (B_p,)
            play_loss = (p_weights * per_sample_loss).mean()
            total_loss = total_loss + play_loss

            # PER priority ← |KL|-style per-sample loss (already ≥ 0).
            td_errors[p_idx] = per_sample_loss.detach().cpu().numpy()

        # ----------------------------------------------------------------
        # Bet-head distributional loss (one-step bandit; done=True always)
        # ----------------------------------------------------------------
        if bet_mask.any():
            b_idx = np.where(bet_mask)[0]

            b_obs     = obs[b_idx]
            b_actions = actions[b_idx]
            b_rewards = rewards[b_idx]
            b_dones   = dones[b_idx]
            b_weights = weights_t[b_idx]
            B_b       = b_obs.shape[0]

            # Current distribution p_online(s, a_t, ·) for the bet head
            _, bet_dist_cur = self.online_net.forward_dist(b_obs)     # (B_b, n_bet, n_atoms)
            gather_idx_b = b_actions.view(B_b, 1, 1).expand(-1, 1, n_atoms)
            p_online_bet = bet_dist_cur.gather(1, gather_idx_b).squeeze(1)   # (B_b, n_atoms)

            with torch.no_grad():
                # Bootstrap factor = 0 → target collapses to delta at clip(R, V_min, V_max).
                # The target_dist argument is immaterial for the projection when bootstrap
                # is zero (all atoms map to the same Tz), so pass a uniform dist.
                uniform_target = torch.full(
                    (B_b, n_atoms), 1.0 / n_atoms, device=self.device
                )
                m_bet = self._project(
                    rewards=b_rewards,
                    dones=b_dones,
                    target_dist=uniform_target,
                    bootstrap_factor=0.0,
                )                                                      # (B_b, n_atoms)

            log_p_bet = torch.log(p_online_bet.clamp_min(1e-8))
            per_sample_loss_bet = -(m_bet * log_p_bet).sum(dim=1)
            bet_loss = (b_weights * per_sample_loss_bet).mean()
            total_loss = total_loss + bet_loss

            td_errors[b_idx] = per_sample_loss_bet.detach().cpu().numpy()

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
