"""
DeepCount Blackjack Environment
================================
A shoe-scoped, POMDP blackjack environment for reinforcement learning.

Episode = one full 6-deck shoe (~312 cards, reshuffled at 75% penetration).

Two sequential decision phases per hand
----------------------------------------
Phase 0 — BETTING   : agent sizes its bet before any cards are dealt.
Phase 1 — PLAYING   : agent plays the hand card-by-card.

Observation space (Dict)
-------------------------
  shoe_history  : (HISTORY_WINDOW,) int8  — cards VISIBLE to the agent so far
                  this shoe, left-padded with 0 (UNSEEN). The dealer's hole
                  card is NOT added until it is explicitly revealed.
  hand          : (MAX_HAND,) int8        — player's cards; 0 = empty slot
  hand_len      : () int8                 — number of real cards in hand
  dealer_upcard : () int8                 — dealer's face-up card (0 if BETTING)
  phase         : () int8                 — 0=BETTING, 1=PLAYING

Action space (Dict)
--------------------
  bet  : Box(1,) float32 in [0,1]  — linearly scaled to [MIN_BET, MAX_BET]
  play : Discrete(4)               — 0=Hit 1=Stand 2=Double 3=Split

The active phase uses only its own action; the other is ignored.

Reward
-------
  Net chip change emitted after each hand resolves.
  A small living penalty (LIVING_PENALTY) is subtracted each hand.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.card_utils import (
    VOCAB_SIZE, SHOE_SIZE, RESHUFFLE_THRESHOLD,
    shuffle_shoe, hand_value, is_bust, is_blackjack, can_split,
    true_count, normalize_rank,
)

# ── Action indices ────────────────────────────────────────────────────────────
HIT       = 0
STAND     = 1
DOUBLE    = 2
SPLIT     = 3

PHASE_BETTING = 0
PHASE_PLAYING = 1

# ── Constants ─────────────────────────────────────────────────────────────────
HISTORY_WINDOW = 234   # sliding window fed to the Transformer
MAX_HAND       = 10   # maximum cards a player hand can hold
MIN_BET        = 1.0
MAX_BET        = 500.0
LIVING_PENALTY = 0.00


class BlackjackShoeEnv(gym.Env):
    """
    Shoe-scoped blackjack: one gymnasium episode = one shoe.

    Args:
        flat_bet    : if not None, overrides the agent's bet choice with this
                      fixed value (used by curriculum stages 1 & 2).
        seed        : optional RNG seed.
        render_mode : 'human' prints a text summary each step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        flat_bet: float | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.flat_bet    = flat_bet
        self.render_mode = render_mode

        # ── Spaces ────────────────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "shoe_history":  spaces.Box(0, VOCAB_SIZE - 1, (HISTORY_WINDOW,), dtype=np.int8),
            "hand":          spaces.Box(0, VOCAB_SIZE - 1, (MAX_HAND,),        dtype=np.int8),
            "hand_len":      spaces.Box(0, MAX_HAND,       (),                 dtype=np.int8),
            "dealer_upcard": spaces.Box(0, VOCAB_SIZE - 1, (),                 dtype=np.int8),
            "phase":         spaces.Box(0, 1,              (),                 dtype=np.int8),
        })
        self.action_space = spaces.Dict({
            "bet":  spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "play": spaces.Discrete(4),
        })

        self._rng = np.random.default_rng(seed)
        self._reset_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        self._shoe = shuffle_shoe(self._rng)
        return self._obs(), self._info()

    def step(self, action: dict):
        if self._phase == PHASE_BETTING:
            return self._step_betting(action["bet"])
        else:
            return self._step_playing(action["play"])

    def render(self):
        if self.render_mode != "human":
            return
        total, soft = hand_value(self._player_hand) if self._player_hand else (0, False)
        tc = true_count(self._visible_history)
        print(
            f"{'BET' if self._phase == PHASE_BETTING else 'PLAY'} | "
            f"TC={tc:+.1f} | "
            f"Hand={self._player_hand}({total}{'s' if soft else ''}) | "
            f"Dealer↑={self._dealer_hand[0] if self._dealer_hand else '?'} | "
            f"Bet={self._current_bet:.0f} | Chips={self._episode_chips:+.1f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _step_betting(self, bet_action: np.ndarray):
        raw = float(np.clip(bet_action[0], 0.0, 1.0))
        if self.flat_bet is not None:
            self._current_bet = float(self.flat_bet)
        else:
            self._current_bet = MIN_BET + raw * (MAX_BET - MIN_BET)

        # Deal four cards: P1, D_upcard, P2, D_hole
        # D_hole is drawn hidden — not added to visible history yet.
        p1     = self._draw_visible()
        d_up   = self._draw_visible()
        p2     = self._draw_visible()
        d_hole = self._draw_hidden()

        self._player_hand = [p1, p2]
        self._dealer_hand = [d_up, d_hole]
        self._doubled     = False
        self._split_hands = None
        self._phase       = PHASE_PLAYING

        # Immediate resolution when either side has a natural blackjack
        player_bj = is_blackjack(self._player_hand)
        dealer_bj = is_blackjack(self._dealer_hand)
        if player_bj or dealer_bj:
            self._reveal_hole()
            reward = self._blackjack_payout(player_bj, dealer_bj)
            self._episode_chips += reward
            return self._transition_or_end(reward)

        return self._obs(), 0.0, False, False, self._info()

    def _step_playing(self, play_action: int):
        valid = self._valid_actions()
        if play_action not in valid:
            play_action = STAND   # safe fallback during early training

        reward    = 0.0
        hand_done = False

        if play_action == HIT:
            self._player_hand.append(self._draw_visible())
            if is_bust(self._player_hand):
                reward    = -self._current_bet - LIVING_PENALTY
                hand_done = True

        elif play_action == STAND:
            self._reveal_hole()
            reward    = self._dealer_play() - LIVING_PENALTY
            hand_done = True

        elif play_action == DOUBLE:
            self._player_hand.append(self._draw_visible())
            self._current_bet *= 2
            self._doubled = True
            if is_bust(self._player_hand):
                reward = -self._current_bet - LIVING_PENALTY
            else:
                self._reveal_hole()
                reward = self._dealer_play() - LIVING_PENALTY
            hand_done = True

        elif play_action == SPLIT:
            card_b            = self._player_hand[1]
            self._player_hand = [self._player_hand[0], self._draw_visible()]
            self._split_hands = [[card_b, self._draw_visible()]]
            # No reward yet; continue playing first split hand


        if hand_done:
            self._episode_chips += reward
            if self._split_hands:
                # Advance to the queued split hand (same dealer upcard)
                self._player_hand = self._split_hands.pop(0)
                self._doubled     = False
                return self._obs(), reward, False, False, self._info()
            return self._transition_or_end(reward)

        return self._obs(), reward, False, False, self._info()

    # ─────────────────────────────────────────────────────────────────────────
    # Card draw helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_visible(self) -> int:
        """Draw from the shoe and record in visible history."""
        card = self._shoe[self._shoe_ptr]
        self._shoe_ptr += 1
        self._visible_history.append(card)
        return card

    def _draw_hidden(self) -> int:
        """Draw from the shoe WITHOUT recording in visible history."""
        card = self._shoe[self._shoe_ptr]
        self._shoe_ptr += 1
        self._hidden_card = card
        return card

    def _reveal_hole(self):
        """Add the previously hidden dealer hole card to visible history."""
        if self._hidden_card is not None:
            self._visible_history.append(self._hidden_card)
            self._hidden_card = None

    # ─────────────────────────────────────────────────────────────────────────
    # Dealer play & payouts
    # ─────────────────────────────────────────────────────────────────────────

    def _dealer_play(self) -> float:
        """Dealer draws to hard 17+ (hits soft 17), then compute payout."""
        while True:
            total, soft = hand_value(self._dealer_hand)
            if total < 17 or (total == 17 and soft):
                self._dealer_hand.append(self._draw_visible())
            else:
                break

        p_total, _ = hand_value(self._player_hand)
        d_total, _ = hand_value(self._dealer_hand)

        if d_total > 21 or p_total > d_total:
            return self._current_bet
        elif p_total == d_total:
            return 0.0
        else:
            return -self._current_bet

    def _blackjack_payout(self, player_bj: bool, dealer_bj: bool) -> float:
        if player_bj and dealer_bj:
            return -LIVING_PENALTY
        elif player_bj:
            return self._current_bet * 1.5 - LIVING_PENALTY
        else:
            return -self._current_bet - LIVING_PENALTY

    # ─────────────────────────────────────────────────────────────────────────
    # Episode transitions
    # ─────────────────────────────────────────────────────────────────────────

    def _shoe_exhausted(self) -> bool:
        return (
            self._shoe_ptr >= RESHUFFLE_THRESHOLD
            or self._shoe_ptr + 4 >= SHOE_SIZE
        )

    def _transition_or_end(self, reward: float):
        terminated = self._shoe_exhausted()
        if not terminated:
            self._phase       = PHASE_BETTING
            self._player_hand = []
            self._dealer_hand = []
            self._hidden_card = None
        return self._obs(), reward, terminated, False, self._info()

    # ─────────────────────────────────────────────────────────────────────────
    # Observation & info
    # ─────────────────────────────────────────────────────────────────────────

    def _obs(self) -> dict:
        hist = np.zeros(HISTORY_WINDOW, dtype=np.int8)
        n = min(len(self._visible_history), HISTORY_WINDOW)
        if n > 0:
            # Collapse 10/J/Q/K → token 10 before feeding to the network
            hist[-n:] = [normalize_rank(r) for r in self._visible_history[-n:]]

        hand_arr = np.zeros(MAX_HAND, dtype=np.int8)
        hl = min(len(self._player_hand), MAX_HAND)
        hand_arr[:hl] = [normalize_rank(r) for r in self._player_hand[:hl]]

        dealer_up = (
            np.int8(normalize_rank(self._dealer_hand[0]))
            if self._dealer_hand else np.int8(0)
        )

        return {
            "shoe_history":  hist,
            "hand":          hand_arr,
            "hand_len":      np.int8(hl),
            "dealer_upcard": dealer_up,
            "phase":         np.int8(self._phase),
        }

    def _info(self) -> dict:
        return {
            "true_count":    true_count(self._visible_history),
            "episode_chips": self._episode_chips,
            "cards_seen":    len(self._visible_history),
            "phase":         self._phase,
            "shoe_pct":      self._shoe_ptr / SHOE_SIZE,
        }

    def _valid_actions(self) -> set[int]:
        valid = {HIT, STAND}
        if len(self._player_hand) == 2 and not self._doubled:
            valid.add(DOUBLE)
            if can_split(self._player_hand):
                valid.add(SPLIT)
        return valid

    # ─────────────────────────────────────────────────────────────────────────
    # State reset
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self._shoe:            list[int]   = []
        self._shoe_ptr:        int         = 0
        self._visible_history: list[int]   = []
        self._hidden_card:     int | None  = None
        self._phase:           int         = PHASE_BETTING
        self._player_hand:     list[int]   = []
        self._dealer_hand:     list[int]   = []
        self._current_bet:     float       = 0.0
        self._doubled:         bool        = False
        self._split_hands:     list | None = None
        self._episode_chips:   float       = 0.0
