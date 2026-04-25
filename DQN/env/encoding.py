"""State vector encoder and action mask builder for the play agent.

Produces a 27-dimensional observation vector covering the playing-state
features and the count features used by the play agent.  The bet agent
uses a separate observation (per-rank shoe composition) — see
``env/bet_encoding.py``.

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

The play agent never sees the bet phase: the training loop reads the
per-rank composition for the bet agent and steps the env's bet phase
itself before any play-phase observation is produced.
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 27

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
) -> np.ndarray:
    """Return a (27,) float32 play-phase observation vector."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

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
