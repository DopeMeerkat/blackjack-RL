"""
PPO Trainer for DeepCount — dual-stream value edition
======================================================
Key design decisions:

1. Phase-routed advantages
   BETTING steps  → advantages_raw  (chip-scale, PopArt-lite normalised)
   PLAYING steps  → advantages_norm (bet-normalised, stationary)

2. Dual value heads
   value_head     trained on returns_norm  (bet-normalised)
   bet_value_head trained on returns_raw   (PopArt-lite chip-scale)

3. Separate aux optimiser for ShoeEncoder (higher lr, isolated backward).

4. Frozen bet head (stages 1 & 2).

5. Per-minibatch KL early stopping (CleanRL-style).
   The check and break happen BEFORE applying the gradient update for
   that minibatch. This means the policy never takes a step that would
   push it past target_kl, rather than stopping only after a full epoch
   of destructive updates has already occurred.

   The KL estimate uses a fresh forward pass on the current (post-update)
   weights against the stored rollout log-probs, matching CleanRL exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models.policy_network import DeepCountNet, N_QUANTILES
from env.blackjack_env import PHASE_BETTING


class PPOTrainer:
    def __init__(
        self,
        policy:        DeepCountNet,
        lr:            float = 3e-4,
        aux_lr:        float = 1e-3,
        clip_eps:      float = 0.2,
        n_epochs:      int   = 10,
        vf_bet_coef:   float = 0.5,
        vf_play_coef:  float = 0.5,
        ent_coef:      float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl:     float = 0.02,
        kappa:         float = 1.0,
        device:        torch.device = torch.device("cpu"),
    ):
        self.policy        = policy
        self.clip_eps      = clip_eps
        self.n_epochs      = n_epochs
        self.vf_bet_coef   = vf_bet_coef
        self.vf_play_coef  = vf_play_coef
        self.ent_coef      = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl     = target_kl
        self.kappa         = kappa
        self.device        = device

        shoe_ids    = {id(p) for p in policy.shoe_encoder.parameters()}
        main_params = [p for p in policy.parameters() if id(p) not in shoe_ids]
        self.optimizer     = Adam(main_params, lr=lr, eps=1e-5)
        self.aux_optimizer = Adam(policy.shoe_encoder.parameters(), lr=aux_lr, eps=1e-5)

        taus = torch.arange(1, N_QUANTILES + 1, device=device)
        self.taus = (2 * taus - 1) / (2 * N_QUANTILES)

    def update(
        self,
        buffer,
        batch_size:      int,
        aux_lambda:      float,
        freeze_bet_head: bool,
    ) -> dict:
        """
        Run up to n_epochs of PPO updates.

        Early stopping is checked PER MINIBATCH (CleanRL-style):
          1. Compute a fresh forward pass on the current weights.
          2. Estimate approx KL against the rollout log-probs.
          3. If KL > target_kl, stop the CURRENT EPOCH without applying
             the gradient step for this minibatch.

        Each epoch resets independently — a KL breach in epoch 1 does
        not cancel epochs 2–n. This preserves training throughput while
        still guarding against large single-step policy shifts.
        """
        metrics: dict[str, list[float]] = {
            "loss_policy": [], "loss_value_play": [], "loss_value_bet": [],
            "loss_entropy": [], "loss_aux": [], "loss_total": [],
            "approx_kl": [], "clip_fraction": [],
        }

        for _epoch in range(self.n_epochs):
            for batch in buffer.get_batches(batch_size):
                stop, stats = self._update_step(
                    batch, aux_lambda, freeze_bet_head
                )
                for k, v in stats.items():
                    metrics[k].append(v)
                if stop:
                    break  # stop this epoch only; next epoch starts fresh

        return {k: sum(v) / len(v) for k, v in metrics.items() if v}

    def _update_step(
        self,
        batch:           dict,
        aux_lambda:      float,
        freeze_bet_head: bool,
    ) -> tuple[bool, dict]:
        """
        One minibatch update.

        Returns:
            stop  : True if approx KL exceeded target_kl (caller should halt).
            stats : dict of scalar diagnostics.

        KL check (CleanRL-style)
        ------------------------
        Before computing gradients, run a no-grad forward pass to measure
        how far the CURRENT policy already is from the rollout policy.
        If this exceeds target_kl, skip the update entirely and signal stop.

        This is subtly different from checking KL after the backward pass:
        it prevents any update that would push the policy over the threshold,
        whereas a post-update check only stops the *next* minibatch.
        """
        obs           = batch["obs"]
        actions       = batch["actions"]
        old_log_probs = batch["log_probs"]
        phase         = batch["phase"]
        action_mask   = batch["action_mask"]

        is_betting   = (phase == PHASE_BETTING)
        adv_combined = torch.where(is_betting, batch["advantages_raw"], batch["advantages_norm"])

        # ── Pre-update KL check (CleanRL-style) ───────────────────────────
        # Measure KL between current policy and rollout policy BEFORE the
        # gradient step. If already too far, skip this minibatch.
        #
        # The check MUST run in eval() mode. Collection was performed in
        # eval() mode (no dropout), so comparing against train()-mode
        # log-probs would inflate KL due to stochastic dropout masks even
        # when the weights have not changed at all.
        #
        # When the bet head is frozen (stages 1+2), BETTING-phase log-probs
        # can drift spuriously: the frozen bet head still receives indirect
        # updates through the shared fusion MLP, shifting bet_mu and thus
        # bet_lp even though play decisions haven't changed. Excluding
        # BETTING steps from the KL estimate during freeze prevents this
        # from triggering false early-stops.
        self.policy.eval()
        with torch.no_grad():
            new_log_probs, _, _, _, _ = self.policy.evaluate_actions(
                obs, actions, action_mask
            )
            log_ratio = new_log_probs - old_log_probs
            if freeze_bet_head:
                # Only measure KL on PLAYING-phase steps
                play_mask = ~is_betting
                if play_mask.any():
                    approx_kl = ((log_ratio[play_mask].exp() - 1) - log_ratio[play_mask]).mean().item()
                else:
                    approx_kl = 0.0
            else:
                approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
        self.policy.train()

        if approx_kl > self.target_kl:
            # Log the KL and signal caller to stop, but do NOT include
            # zero-valued loss entries.  Averaging 0.0 into the loss metrics
            # for skipped batches corrupts the reported value (e.g. a rollout
            # where half the batches are KL-stopped would report half the true
            # loss), causing the sawtooth pattern visible in TensorBoard.
            return True, {"approx_kl": approx_kl}

        # ── Full forward pass for gradient computation ────────────────────
        log_probs, entropy, quantiles, bet_quantiles, count_pred = \
            self.policy.evaluate_actions(obs, actions, action_mask)

        ratio       = torch.exp(log_probs - old_log_probs)
        surr1       = ratio * adv_combined
        surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_combined
        policy_loss = -torch.min(surr1, surr2).mean()

        is_playing = ~is_betting
        if is_playing.any():
            value_loss_play = self._quantile_huber_loss(
                quantiles[is_playing], batch["returns_norm"][is_playing])
        else:
            value_loss_play = torch.tensor(0.0, device=self.device)
        if is_betting.any():
            value_loss_bet = self._quantile_huber_loss(
                bet_quantiles[is_betting], batch["returns_raw"][is_betting])
        else:
            value_loss_bet = torch.tensor(0.0, device=self.device)
        value_loss = self.vf_play_coef * value_loss_play + self.vf_bet_coef * value_loss_bet

        entropy_loss = -entropy.mean()
        ppo_loss     = policy_loss + value_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        ppo_loss.backward()

        if freeze_bet_head:
            for head in (self.policy.bet_head, self.policy.bet_value_head):
                for p in head.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            if self.policy.log_bet_std.grad is not None:
                self.policy.log_bet_std.grad.zero_()

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── Aux count loss ────────────────────────────────────────────────
        aux_loss = torch.tensor(0.0, device=self.device)
        if aux_lambda > 0.0:
            _, count_pred_aux = self.policy.shoe_encoder(
                obs["shoe_history"], return_aux=True
            )
            aux_loss = F.mse_loss(count_pred_aux.squeeze(-1), batch["true_counts"])
            self.aux_optimizer.zero_grad()
            (aux_lambda * aux_loss).backward()
            nn.utils.clip_grad_norm_(
                self.policy.shoe_encoder.parameters(), self.max_grad_norm
            )
            self.aux_optimizer.step()

        # ── Post-update clip fraction ─────────────────────────────────────
        with torch.no_grad():
            clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

        return False, {
            "loss_policy":     policy_loss.item(),
            "loss_value_play": value_loss_play.item(),
            "loss_value_bet":  value_loss_bet.item(),
            "loss_entropy":    entropy_loss.item(),
            "loss_aux":        aux_loss.item(),
            "loss_total":      ppo_loss.item() + aux_lambda * aux_loss.item(),
            "approx_kl":       approx_kl,
            "clip_fraction":   clip_frac,
        }

    def _quantile_huber_loss(self, quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.unsqueeze(1).expand_as(quantiles)
        taus    = self.taus.unsqueeze(0).expand_as(quantiles)
        u       = targets - quantiles
        huber   = torch.where(
            u.abs() <= self.kappa,
            0.5 * u.pow(2),
            self.kappa * (u.abs() - 0.5 * self.kappa),
        )
        return ((taus - (u < 0).float()).abs() * huber).mean()
