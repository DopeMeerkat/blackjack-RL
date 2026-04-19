"""State vector encoder and action mask builder.

Produces the authoritative 28-dimensional observation vector described in
blackjack_rl_design.md §3 and the boolean action mask from §4.

All functions are pure (no side effects) and operate on scalar Python values
for clarity.  The blackjack env calls these at each decision point.

State vector layout (indices are 0-based):
  [0]      player hand sum: (sum - 4) / 17
  [1]      usable ace flag: {0, 1}
  [2..11]  dealer up-card one-hot over [A,2,3,4,5,6,7,8,9,10+face] (10 slots)
  [12]     pair flag: {0, 1}
  [13..22] pair rank one-hot (same 10-slot mapping); all zeros when no pair
  [23]     can-double flag: {0, 1}
  [24]     can-split flag: {0, 1}
  [25]     true count: clip(TC, -5, +5) / 5
  [26]     decks remaining: decks_remaining / 6
  [27]     current bet (normalised): bet_multiplier / 12

At bet-sizing time the playing-state features are not yet defined and are
zeroed: indices 0, 1, 2-11, 12-24, 27.  Only obs[25] and obs[26] (count and
decks) are populated, because cards have not been dealt yet so the dealer
upcard is also unknown.
"""

from __future__ import annotations

import numpy as np

# Mapping from raw card rank [1-13] to one-hot index in [0, 9].
# Ace(1)→0, 2→1, 3→2, ..., 9→8, 10/J/Q/K→9.
_RANK_TO_UPCARD_IDX: dict[int, int] = {
    1: 0,   # Ace
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 9,  # Jack
    12: 9,  # Queen
    13: 9,  # King
}

# The pair-rank one-hot uses the same 10-slot bucketing as the dealer upcard.
_RANK_TO_PAIR_IDX = _RANK_TO_UPCARD_IDX


def encode_state(
    *,
    player_sum: int,
    usable_ace: bool,
    dealer_upcard_rank: int,       # raw rank 1-13
    is_pair: bool,
    pair_rank: int | None,         # raw rank, or None when no pair
    can_double: bool,
    can_split: bool,
    true_count: float,
    decks_remaining: float,
    bet_multiplier: float,         # actual multiplier value (1, 2, 4, 8, or 12)
    bet_phase: bool = False,
) -> np.ndarray:
    """Return a (28,) float32 observation vector.

    When ``bet_phase=True`` the playing-state features (indices 0, 1, 2-11,
    12-24, 27) are zeroed because the hand has not been dealt yet.  Only the
    count and decks-remaining features are populated.
    """
    obs = np.zeros(28, dtype=np.float32)

    if not bet_phase:
        obs[0] = (player_sum - 4) / 17.0
        obs[1] = float(usable_ace)

        # Dealer up-card one-hot: obs[2] through obs[11]
        upcard_idx = _RANK_TO_UPCARD_IDX[dealer_upcard_rank]
        obs[2 + upcard_idx] = 1.0

        # Pair info
        obs[12] = float(is_pair)
        if is_pair and pair_rank is not None:
            pair_idx = _RANK_TO_PAIR_IDX[pair_rank]
            obs[13 + pair_idx] = 1.0

        obs[23] = float(can_double)
        obs[24] = float(can_split)

        obs[27] = bet_multiplier / 12.0

    # Count features are always populated (the agent needs them during bet phase).
    obs[25] = float(np.clip(true_count, -5.0, 5.0)) / 5.0
    obs[26] = decks_remaining / 6.0

    return obs


def compute_mask(
    *,
    can_hit: bool,
    can_stand: bool,
    can_double: bool,
    can_split: bool,
) -> np.ndarray:
    """Return a (4,) boolean action mask.

    Index mapping: hit=0, stand=1, double=2, split=3.
    True means the action is *legal*.
    """
    return np.array([can_hit, can_stand, can_double, can_split], dtype=bool)
