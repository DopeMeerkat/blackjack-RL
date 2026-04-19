"""Single-environment blackjack simulator.

Implements a six-deck shoe game with the exact rules from
blackjack_rl_design.md §2:
  - Dealer stands on soft 17 (S17)
  - Double after split (DAS) allowed
  - Resplit up to 3 times (max 4 sub-hands)
  - Blackjack pays 3:2
  - No surrender
  - Insurance: hard-coded (take when true count >= +3, not a learned action)

Interface mirrors Gymnasium's reset()/step() convention but is pure NumPy —
no Gymnasium dependency.

Episode structure:
  reset()          → bet-phase observation; agent selects bet (0-4 index)
  step(bet_idx)    → deals cards, enters play phase (or resolves immediately
                     if BJ/dealer-BJ)
  step(play_action)→ hit/stand/double/split; done=True when hand is fully
                     resolved

The first call after reset() is ALWAYS a bet-sizing step.  The training loop
identifies bet vs. play steps via the ``in_bet_phase`` property or
``info["phase"]``.

Reward convention (§5):
  win:       +1.0 × bet_multiplier  (× 2 if doubled)
  loss:      -1.0 × bet_multiplier  (× 2 if doubled)
  push:       0.0
  nat. BJ:  +1.5 × bet_multiplier  (non-split hand only)
  Sub-hands from splits are summed; each has the original bet_multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.count import HiLoCount
from env.encoding import compute_mask, encode_state

# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

_BET_MULTIPLIERS = [1, 2, 4, 8, 12]  # indexed 0-4


def _card_value(rank: int) -> int:
    """Blackjack value of a card rank.

    Ace is returned as 11 (caller responsible for soft/hard logic).
    10/J/Q/K all return 10.
    """
    if rank == 1:
        return 11
    if rank >= 10:
        return 10
    return rank


def hand_sum(cards: list[int]) -> tuple[int, bool]:
    """Return (best_total, usable_ace) for a list of card ranks.

    best_total: the highest achievable total <= 21, or the smallest possible
                total if busting is unavoidable.
    usable_ace: True if exactly one Ace is being counted as 11.

    The algorithm treats all Aces as 1 (hard total) then adds 10 once if that
    brings the total to at most 21, regardless of how many Aces are in the
    hand.  This correctly handles multiple Aces.
    """
    hard_total = sum(1 if r == 1 else (10 if r >= 10 else r) for r in cards)
    has_ace = any(r == 1 for r in cards)
    if has_ace and hard_total + 10 <= 21:
        return hard_total + 10, True
    return hard_total, False


def _is_pair(cards: list[int]) -> bool:
    """True when exactly two cards have equal blackjack value (10/J/Q/K match)."""
    return len(cards) == 2 and _card_value(cards[0]) == _card_value(cards[1])


def _pair_canonical_rank(cards: list[int]) -> int:
    """Return the canonical rank for encoding the pair.

    Uses the first card's rank, clamping all 10-value ranks to 10 (Ace stays 1).
    """
    r = cards[0]
    if r == 1:
        return 1
    if r >= 10:
        return 10
    return r


# ---------------------------------------------------------------------------
# Hand dataclass
# ---------------------------------------------------------------------------

@dataclass
class Hand:
    cards: list[int] = field(default_factory=list)
    doubled: bool = False
    from_split: bool = False
    split_from_aces: bool = False  # special: split aces get one card, no hit
    bet_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# BlackjackEnv
# ---------------------------------------------------------------------------

class BlackjackEnv:
    """Single-environment six-deck blackjack with Hi-Lo card counting.

    Args:
        config: dict loaded from configs/default.yaml (env section).
        shoe: Optional pre-built shoe array (shape (312,), ranks 1-13).
              When provided, the shoe is used as-is without shuffling —
              intended for deterministic testing only.
        seed: RNG seed.  Ignored when ``shoe`` is provided.
    """

    def __init__(
        self,
        config: dict,
        shoe: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(seed)
        self._count = HiLoCount(num_decks=config["num_decks"])
        self._test_shoe = shoe  # fixed shoe for testing; None in production

        # Build and (optionally) shuffle the initial shoe
        self._shoe: np.ndarray = np.empty(0, dtype=np.int8)
        self._shoe_pos: int = 0
        self._init_shoe()

        # Hand state (populated by reset/step)
        self._phase: str = "bet"          # "bet" or "play"
        self._bet_multiplier: float = 1.0
        self._hands: list[Hand] = []
        self._active_hand_idx: int = 0
        self._dealer_cards: list[int] = []
        self._insurance_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Begin a new hand.  Returns (obs, mask, info) in bet phase."""
        # Reshuffle check: done at the START of each hand (never mid-hand).
        if self._test_shoe is None and self._should_reshuffle():
            self._init_shoe()

        # Reset hand state
        self._phase = "bet"
        self._bet_multiplier = 1.0
        self._hands = []
        self._active_hand_idx = 0
        self._dealer_cards = []
        self._insurance_reward = 0.0

        obs = self._bet_phase_obs()
        mask = np.zeros(4, dtype=bool)  # all play actions illegal in bet phase
        info = self._make_info()
        return obs, mask, info

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Advance the environment by one action.

        During bet phase: action is a bet index (0-4).
        During play phase: action is a play index (0=hit, 1=stand, 2=double, 3=split).

        Returns (obs, mask, reward, done, info).
        """
        if self._phase == "bet":
            return self._bet_step(action)
        return self._play_step(action)

    @property
    def in_bet_phase(self) -> bool:
        return self._phase == "bet"

    @property
    def cards_in_shoe(self) -> int:
        return int(len(self._shoe) - self._shoe_pos)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Shoe management
    # ------------------------------------------------------------------

    def _init_shoe(self) -> None:
        """Build and shuffle a fresh shoe (or load the test shoe)."""
        if self._test_shoe is not None:
            self._shoe = self._test_shoe.copy()
            self._shoe_pos = 0
            self._count.reset()
            return

        # Build 6-deck shoe: ranks 1-13, four of each per deck
        single_deck = list(range(1, 14)) * 4   # 52 cards
        shoe_list = single_deck * self._cfg["num_decks"]
        self._shoe = np.array(shoe_list, dtype=np.int8)
        self._rng.shuffle(self._shoe)
        self._shoe_pos = 0
        self._count.reset()

    def _should_reshuffle(self) -> bool:
        cards_remaining = self.cards_in_shoe
        return (cards_remaining / 52.0) < self._cfg["reshuffle_threshold"]

    def _deal_card(self) -> int:
        """Deal one card, update the Hi-Lo count, and return the rank."""
        if self._shoe_pos >= len(self._shoe):
            # Safety: reshuffle if we somehow run out (shouldn't happen with
            # proper penetration settings, but prevents an index error).
            self._init_shoe()
        rank = int(self._shoe[self._shoe_pos])
        self._shoe_pos += 1
        self._count.update(rank)
        return rank

    # ------------------------------------------------------------------
    # Bet phase
    # ------------------------------------------------------------------

    def _bet_step(self, bet_idx: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Process the bet-sizing action and deal the initial four cards."""
        self._bet_multiplier = float(_BET_MULTIPLIERS[bet_idx])

        # Deal: player c1, dealer c1 (up), player c2, dealer c2 (hole).
        # All four cards are counted immediately — the env is for training and
        # has full information; no hidden card tracking needed.
        p1 = self._deal_card()
        d1 = self._deal_card()  # dealer up-card
        p2 = self._deal_card()
        d2 = self._deal_card()  # dealer hole card

        self._dealer_cards = [d1, d2]
        player_hand = Hand(
            cards=[p1, p2],
            bet_multiplier=self._bet_multiplier,
        )
        self._hands = [player_hand]
        self._active_hand_idx = 0

        # Insurance (hard-coded, not a learned action).
        # Offer only when dealer shows an Ace.
        if d1 == 1:
            tc = self._count.true_count(self.cards_in_shoe)
            if tc >= self._cfg["insurance_tc_threshold"]:
                # Insurance bet = 0.5 × bet_multiplier; pays 2:1 if dealer has BJ.
                # Net effect: if dealer BJ → insurance cancels main bet loss → net 0.
                # If no dealer BJ → insurance side bet lost → -0.5 × multiplier.
                if _is_natural_bj(self._dealer_cards):
                    self._insurance_reward = 0.5 * self._bet_multiplier
                else:
                    self._insurance_reward = -0.5 * self._bet_multiplier

        dealer_bj = _is_natural_bj(self._dealer_cards)
        player_bj = _is_natural_bj(player_hand.cards)

        # Immediate resolution cases (dealer/player blackjack).
        if dealer_bj or player_bj:
            reward = self._resolve_immediate_bj(player_bj, dealer_bj)
            obs = self._bet_phase_obs()          # hand is over; next reset will re-enter bet
            mask = np.zeros(4, dtype=bool)
            return obs, mask, reward, True, self._make_info()

        # Normal play: enter play phase.
        self._phase = "play"
        obs, mask = self._play_phase_obs()
        return obs, mask, 0.0, False, self._make_info()

    def _resolve_immediate_bj(self, player_bj: bool, dealer_bj: bool) -> float:
        """Resolve the hand when at least one side has a natural blackjack."""
        if player_bj and not dealer_bj:
            outcome = self._cfg["blackjack_payout"]   # +1.5
        elif dealer_bj and not player_bj:
            outcome = -1.0
        else:
            outcome = 0.0   # both BJ → push

        reward = outcome * self._bet_multiplier + self._insurance_reward
        return float(reward)

    # ------------------------------------------------------------------
    # Play phase
    # ------------------------------------------------------------------

    def _play_step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Process a playing action (hit/stand/double/split)."""
        hand = self._hands[self._active_hand_idx]

        if action == 0:   # Hit
            card = self._deal_card()
            hand.cards.append(card)
            total, _ = hand_sum(hand.cards)
            if total > 21:
                # Bust: advance to next sub-hand or resolve.
                return self._finish_hand_or_advance()
            # Still alive: return updated obs.
            obs, mask = self._play_phase_obs()
            return obs, mask, 0.0, False, self._make_info()

        elif action == 1:   # Stand
            return self._finish_hand_or_advance()

        elif action == 2:   # Double
            card = self._deal_card()
            hand.cards.append(card)
            hand.doubled = True
            # Double → forced stand after receiving one card.
            return self._finish_hand_or_advance()

        elif action == 3:   # Split
            return self._execute_split()

        else:
            raise ValueError(f"Invalid play action: {action}")

    def _execute_split(self) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Split the current hand into two sub-hands."""
        hand = self._hands[self._active_hand_idx]
        c1, c2 = hand.cards[0], hand.cards[1]
        splitting_aces = (c1 == 1)   # raw rank 1 = Ace

        new1 = Hand(
            cards=[c1],
            from_split=True,
            split_from_aces=splitting_aces,
            bet_multiplier=self._bet_multiplier,
        )
        new2 = Hand(
            cards=[c2],
            from_split=True,
            split_from_aces=splitting_aces,
            bet_multiplier=self._bet_multiplier,
        )

        # Replace current hand with the two new sub-hands.
        self._hands[self._active_hand_idx : self._active_hand_idx + 1] = [new1, new2]

        # Deal the second card to the first (now active) sub-hand.
        card = self._deal_card()
        new1.cards.append(card)

        # If we split aces, the first hand auto-completes after this one card.
        # (Standard rule: split aces receive exactly one card each, then stand.)
        if splitting_aces:
            return self._finish_hand_or_advance()

        obs, mask = self._play_phase_obs()
        return obs, mask, 0.0, False, self._make_info()

    def _finish_hand_or_advance(self) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """Move past the current sub-hand.

        If more sub-hands remain, deal cards to the next one and continue.
        If all sub-hands are complete, run the dealer and resolve.
        """
        self._active_hand_idx += 1

        if self._active_hand_idx < len(self._hands):
            # There's a next sub-hand (from a split).  Deal its second card
            # if it only has one card (i.e., just created by the split).
            next_hand = self._hands[self._active_hand_idx]
            if len(next_hand.cards) == 1:
                card = self._deal_card()
                next_hand.cards.append(card)

            # If this is a split-aces sub-hand, it also auto-stands immediately.
            if next_hand.split_from_aces:
                return self._finish_hand_or_advance()

            obs, mask = self._play_phase_obs()
            return obs, mask, 0.0, False, self._make_info()

        # All sub-hands are done — run dealer and resolve.
        reward = self._run_dealer_and_resolve()
        self._phase = "bet"   # ready for next hand via reset()
        obs = self._bet_phase_obs()
        mask = np.zeros(4, dtype=bool)
        return obs, mask, reward, True, self._make_info()

    # ------------------------------------------------------------------
    # Dealer and resolution
    # ------------------------------------------------------------------

    def _run_dealer(self) -> None:
        """Dealer plays by the S17 rule: hit on hard ≤16 or soft ≤16."""
        while True:
            total, usable_ace = hand_sum(self._dealer_cards)
            if total < 17:
                card = self._deal_card()
                self._dealer_cards.append(card)
            elif total == 17 and usable_ace:
                # Soft 17: dealer stands (S17 rule).
                break
            else:
                break

    def _run_dealer_and_resolve(self) -> float:
        """Run dealer play loop then score all sub-hands."""
        # Only run dealer if at least one player hand hasn't busted.
        any_alive = any(hand_sum(h.cards)[0] <= 21 for h in self._hands)
        if any_alive:
            self._run_dealer()

        dealer_total, _ = hand_sum(self._dealer_cards)
        dealer_busted = dealer_total > 21

        total_reward = 0.0
        for hand in self._hands:
            player_total, _ = hand_sum(hand.cards)
            player_busted = player_total > 21
            double_mult = 2.0 if hand.doubled else 1.0

            # Natural BJ is impossible for split sub-hands.
            if player_busted:
                outcome = -1.0
            elif dealer_busted:
                outcome = 1.0
            elif player_total > dealer_total:
                outcome = 1.0
            elif player_total < dealer_total:
                outcome = -1.0
            else:
                outcome = 0.0   # push

            total_reward += outcome * double_mult * hand.bet_multiplier

        total_reward += self._insurance_reward
        return float(total_reward)

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _bet_phase_obs(self) -> np.ndarray:
        """28-dim observation with playing-state features zeroed."""
        tc = self._count.true_count(self.cards_in_shoe)
        dr = self._count.decks_remaining(self.cards_in_shoe)
        return encode_state(
            player_sum=0,
            usable_ace=False,
            dealer_upcard_rank=1,   # placeholder; zeroed by bet_phase=True
            is_pair=False,
            pair_rank=None,
            can_double=False,
            can_split=False,
            true_count=tc,
            decks_remaining=dr,
            bet_multiplier=0.0,     # multiplier not yet chosen; obs[27] = 0
            bet_phase=True,
        )

    def _play_phase_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs, mask) for the current active sub-hand."""
        hand = self._hands[self._active_hand_idx]
        player_total, usable_ace = hand_sum(hand.cards)
        dealer_upcard = self._dealer_cards[0]

        can_hit, can_stand, can_double, can_split = self._legal_actions(hand)

        tc = self._count.true_count(self.cards_in_shoe)
        dr = self._count.decks_remaining(self.cards_in_shoe)

        obs = encode_state(
            player_sum=player_total,
            usable_ace=usable_ace,
            dealer_upcard_rank=dealer_upcard,
            is_pair=_is_pair(hand.cards),
            pair_rank=_pair_canonical_rank(hand.cards) if _is_pair(hand.cards) else None,
            can_double=can_double,
            can_split=can_split,
            true_count=tc,
            decks_remaining=dr,
            bet_multiplier=self._bet_multiplier,
            bet_phase=False,
        )
        mask = compute_mask(
            can_hit=can_hit,
            can_stand=can_stand,
            can_double=can_double,
            can_split=can_split,
        )
        return obs, mask

    def _legal_actions(self, hand: Hand) -> tuple[bool, bool, bool, bool]:
        """Return (can_hit, can_stand, can_double, can_split) for the hand."""
        num_cards = len(hand.cards)

        # Split aces: receive exactly one card then auto-stand (enforced in
        # _execute_split/_finish_hand_or_advance, but reflect here for mask).
        if hand.split_from_aces:
            # Should never be asked for legal actions for a split-ace hand
            # (they auto-complete), but be safe.
            return False, True, False, False

        can_hit = True
        can_stand = True
        can_double = (
            num_cards == 2
            and (not hand.from_split or self._cfg.get("double_after_split", True))
        )
        can_split = (
            num_cards == 2
            and _is_pair(hand.cards)
            and len(self._hands) < self._cfg.get("resplit_max_hands", 4)
        )
        return can_hit, can_stand, can_double, can_split

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _make_info(self) -> dict:
        tc = self._count.true_count(self.cards_in_shoe)
        dr = self._count.decks_remaining(self.cards_in_shoe)
        return {
            "phase": self._phase,
            "true_count": tc,
            "decks_remaining": dr,
            "bet_multiplier": self._bet_multiplier,
        }


# ---------------------------------------------------------------------------
# Module-level helper (used in tests)
# ---------------------------------------------------------------------------

def _is_natural_bj(cards: list[int]) -> bool:
    """True when the two-card hand is a natural blackjack (21 on first two cards)."""
    return len(cards) == 2 and hand_sum(cards)[0] == 21
