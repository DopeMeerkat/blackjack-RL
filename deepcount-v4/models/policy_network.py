"""
DeepCount Policy Network
========================
Unified actor-critic with dual value heads for correct bet-sizing gradients.

Architecture
------------
                ┌──────────────────┐    ┌────────────────────┐
shoe_history ──▶│  ShoeEncoder     │    │  HandEncoder       │◀── hand
                │  (Transformer)   │    │  (Set-Transformer) │◀── dealer_upcard
                └────────┬─────────┘    └────────┬───────────┘
                         │ shoe_state (d_shoe)    │ hand_state (d_hand)
                         └──────────┬─────────────┘
                                    │ cat → fusion MLP → z (d_fused)
                    ┌───────────────┼──────────────────┬───────────────────┐
              ┌─────▼──────┐  ┌────▼────┐  ┌──────────▼──────┐  ┌────────▼──────┐
              │ Play head  │  │Bet head │  │  value_head     │  │ bet_value_head│
              │ Discrete(4)│  │ Scalar  │  │ (N_QUANTILES)   │  │ (N_QUANTILES) │
              └────────────┘  └─────────┘  │ bet-normalised  │  │ chip-scale    │
                                           │ returns         │  │ returns       │
                                           └─────────────────┘  └───────────────┘

Dual value heads
----------------
value_head      — trained on bet-normalised returns (∈ [-1, 1.5]).
                  Used for play-action advantages. Stationary target.

bet_value_head  — trained on raw chip returns, normalised by running std
                  (PopArt-lite). Used for bet-sizing advantages. Captures
                  the fact that a TC=+5 shoe is worth more real chips than
                  a TC=-3 shoe, which is the signal the bet head needs.

Phase routing in evaluate_actions() and get_values() selects the correct
head for each phase so gradients never cross-contaminate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shoe_encoder import ShoeEncoder
from models.hand_encoder import HandEncoder
from env.card_utils import VOCAB_SIZE
from env.blackjack_env import (
    PHASE_BETTING, PHASE_PLAYING,
    MIN_BET, MAX_BET,
)

N_PLAY_ACTIONS = 4
N_QUANTILES    = 51


class DeepCountNet(nn.Module):
    """
    Unified actor-critic for DeepCount.

    Args:
        d_shoe        : ShoeEncoder hidden dim (default 64)
        d_hand        : HandEncoder output dim (default 32)
        d_fused       : joint feature dim after fusion (default 128)
        n_heads_shoe  : attention heads in ShoeEncoder (default 4)
        n_heads_hand  : attention heads in HandEncoder (default 4)
        n_quantiles   : return-distribution quantiles (default 51)
        dropout       : dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_shoe:       int   = 64,
        d_hand:       int   = 32,
        d_fused:      int   = 128,
        n_heads_shoe: int   = 4,
        n_heads_hand: int   = 4,
        n_quantiles:  int   = N_QUANTILES,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d_shoe      = d_shoe
        self.d_hand      = d_hand
        self.d_fused     = d_fused
        self.n_quantiles = n_quantiles

        # ── Encoders ──────────────────────────────────────────────────────
        self.shoe_encoder = ShoeEncoder(
            vocab_size=VOCAB_SIZE, d_model=d_shoe,
            n_heads=n_heads_shoe, dropout=dropout,
        )
        self.hand_encoder = HandEncoder(
            vocab_size=VOCAB_SIZE, d_hand=d_hand,
            n_heads=n_heads_hand, dropout=dropout,
        )

        # ── Fusion MLP ────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(d_shoe + d_hand, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Linear(d_fused, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
        )

        # ── Policy heads ──────────────────────────────────────────────────
        self.play_head  = nn.Linear(d_fused, N_PLAY_ACTIONS)
        self.bet_head   = nn.Sequential(
            nn.Linear(d_fused, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.log_bet_std = nn.Parameter(torch.zeros(1))

        # ── Value heads ───────────────────────────────────────────────────
        # value_head     : bet-normalised returns → play advantages
        self.value_head = nn.Sequential(
            nn.Linear(d_fused, 64),
            nn.ReLU(),
            nn.Linear(64, n_quantiles),
        )
        # bet_value_head : raw chip returns (PopArt-lite) → bet advantages
        self.bet_value_head = nn.Sequential(
            nn.Linear(d_fused, 64),
            nn.ReLU(),
            nn.Linear(64, n_quantiles),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, obs: dict) -> dict:
        """
        Returns dict with all head outputs. Keys:
          play_logits    (B, 5)
          bet_mu         (B, 1)  ∈ (0,1)
          bet_std        (B, 1)
          quantiles      (B, N_QUANTILES)  — bet-normalised value
          bet_quantiles  (B, N_QUANTILES)  — chip-scale value (PopArt-lite)
          shoe_state     (B, d_shoe)
          count_pred     (B, 1)
          z              (B, d_fused)
        """
        shoe_state, count_pred = self.shoe_encoder(
            obs["shoe_history"].long(), return_aux=True
        )
        hand_state = self.hand_encoder(
            obs["hand"].long(),
            obs["hand_len"].long(),
            obs["dealer_upcard"].long(),
        )

        z = self.fusion(torch.cat([shoe_state, hand_state], dim=-1))

        play_logits   = self.play_head(z)
        bet_raw       = self.bet_head(z)
        bet_mu        = torch.sigmoid(bet_raw)
        bet_std       = self.log_bet_std.exp().expand_as(bet_mu)
        quantiles     = self.value_head(z)
        bet_quantiles = self.bet_value_head(z)

        return {
            "play_logits":   play_logits,
            "bet_mu":        bet_mu,
            "bet_std":       bet_std,
            "quantiles":     quantiles,
            "bet_quantiles": bet_quantiles,
            "shoe_state":    shoe_state,
            "count_pred":    count_pred,
            "z":             z,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Action sampling
    # ─────────────────────────────────────────────────────────────────────────

    def get_action_and_logprob(
        self,
        obs:           dict,
        action_mask:   torch.Tensor | None = None,
        deterministic: bool  = False,
        flat_bet:      float | None = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return its log-probability.

        Args:
            obs           : batched observation dict.
            action_mask   : (B, N_PLAY_ACTIONS) bool — True = legal.
            deterministic : use argmax / mean instead of sampling.
            flat_bet      : if not None, override the sampled bet with this
                            chip value and compute log_prob for that value.
                            This keeps the (action, log_prob) pair consistent
                            when the curriculum forces a fixed bet size.

        Returns:
            action        : dict with 'bet' (B,1) and 'play' (B,)
            log_prob      : (B,) — log π(a|s) for the ACTUAL action returned
            quantiles     : (B, N_QUANTILES)  — play value head
            bet_quantiles : (B, N_QUANTILES)  — bet value head
        """
        out   = self.forward(obs)
        phase = obs["phase"].long()

        # ── Play action ───────────────────────────────────────────────────
        logits = out["play_logits"].clone()
        if action_mask is not None:
            logits[~action_mask] = -1e9
        play_dist   = torch.distributions.Categorical(logits=logits)
        play_action = logits.argmax(-1) if deterministic else play_dist.sample()
        play_lp     = play_dist.log_prob(play_action)

        # ── Bet action ────────────────────────────────────────────────────
        bet_mu, bet_std = out["bet_mu"], out["bet_std"]
        bet_dist = torch.distributions.Normal(bet_mu.squeeze(-1), bet_std.squeeze(-1))

        if flat_bet is not None:
            # Curriculum override: use flat_bet as the action.
            # Compute log_prob for flat_bet so the buffer invariant holds:
            #   stored log_prob == log_prob of stored action.
            bet_norm = torch.tensor(
                (flat_bet - MIN_BET) / (MAX_BET - MIN_BET),
                dtype=torch.float32, device=bet_mu.device,
            ).clamp(1e-4, 1 - 1e-4)
            bet_raw    = torch.logit(bet_norm).expand_as(bet_mu.squeeze(-1))
            bet_sq     = bet_norm.expand_as(bet_mu.squeeze(-1))
            bet_scaled = torch.full_like(bet_mu.squeeze(-1), flat_bet)
        else:
            bet_raw    = bet_mu.squeeze(-1) if deterministic else bet_dist.sample()
            bet_sq     = torch.sigmoid(bet_raw)
            bet_scaled = MIN_BET + bet_sq * (MAX_BET - MIN_BET)

        bet_lp = bet_dist.log_prob(bet_raw) - torch.log(bet_sq * (1 - bet_sq) + 1e-6)

        # ── Phase-selective log_prob ──────────────────────────────────────
        is_betting = (phase == PHASE_BETTING).float()
        log_prob   = is_betting * bet_lp + (1 - is_betting) * play_lp

        action = {
            "bet":  bet_scaled.unsqueeze(-1).detach(),
            "play": play_action.detach(),
        }
        return action, log_prob, out["quantiles"], out["bet_quantiles"]

    def evaluate_actions(
        self,
        obs:         dict,
        actions:     dict,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under current policy params.

        Returns:
            log_prob      : (B,)
            entropy       : (B,)
            quantiles     : (B, N_QUANTILES)  — play value
            bet_quantiles : (B, N_QUANTILES)  — bet value
            count_pred    : (B, 1)
        """
        out   = self.forward(obs)
        phase = obs["phase"].long()

        logits = out["play_logits"].clone()
        if action_mask is not None:
            logits[~action_mask] = -1e9
        play_dist = torch.distributions.Categorical(logits=logits)
        play_lp   = play_dist.log_prob(actions["play"].long())
        play_ent  = play_dist.entropy()

        bet_mu, bet_std = out["bet_mu"], out["bet_std"]
        bet_dist  = torch.distributions.Normal(bet_mu.squeeze(-1), bet_std.squeeze(-1))
        bet_norm  = (actions["bet"].squeeze(-1) - MIN_BET) / (MAX_BET - MIN_BET)
        bet_norm  = bet_norm.clamp(1e-4, 1 - 1e-4)
        bet_raw   = torch.logit(bet_norm)
        bet_lp    = bet_dist.log_prob(bet_raw) - \
                    torch.log(bet_norm * (1 - bet_norm) + 1e-6)
        bet_ent   = bet_dist.entropy()

        is_betting = (phase == PHASE_BETTING).float()
        log_prob   = is_betting * bet_lp   + (1 - is_betting) * play_lp
        entropy    = is_betting * bet_ent  + (1 - is_betting) * play_ent

        return log_prob, entropy, out["quantiles"], out["bet_quantiles"], out["count_pred"]

    # ─────────────────────────────────────────────────────────────────────────
    # Value helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_values(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            value_norm : (B,) — mean of play quantiles (bet-normalised units)
            value_raw  : (B,) — mean of bet quantiles  (chip units, PopArt-lite)
        """
        out = self.forward(obs)
        return (
            out["quantiles"].mean(dim=-1),
            out["bet_quantiles"].mean(dim=-1),
        )
