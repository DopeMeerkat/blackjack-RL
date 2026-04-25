"""Observation encoder for the bet-sizing agent.

The bet agent sees a 12-dimensional vector describing the current shoe
composition and count features — enough to reproduce and generalise Hi-Lo
bet-ramp logic without access to per-card history.

Observation layout (indices 0-based):
  [0..9]  Rank composition fractions: remaining cards of each rank bucket
          divided by total remaining cards.  Bucket order mirrors
          _RANK_TO_BUCKET in env/blackjack.py:
            0=Ace, 1=2, 2=3, 3=4, 4=5, 5=6, 6=7, 7=8, 8=9, 9=10-group.
          In a balanced shoe all buckets equal their full-shoe frequency
          (≈7.7% for A–9, ≈30.8% for 10-group).
  [10]    True count: clip(TC, −5, +5) / 5  ∈ [−1, 1]
  [11]    Decks remaining: decks_remaining / 6  ∈ [0, 1]

At the start of a fresh shoe the composition features equal their
full-shoe frequencies regardless of penetration, making the observation
stationary in that limit.
"""

from __future__ import annotations

import numpy as np

BET_OBS_DIM: int = 12

# Full 6-deck shoe expected card count per rank bucket (integer).
# Ace×24, ranks 2-9 ×24 each, 10-group (10/J/Q/K) ×96.
_FULL_SHOE_COUNTS = np.array(
    [24, 24, 24, 24, 24, 24, 24, 24, 24, 96], dtype=np.float32
)
_FULL_SHOE_TOTAL: float = float(_FULL_SHOE_COUNTS.sum())   # 312


def encode_bet_obs(
    rank_counts: np.ndarray,    # (10,) int — from BlackjackEnv.get_rank_counts()
    true_count: float,
    decks_remaining: float,
) -> np.ndarray:
    """Return a (12,) float32 bet-agent observation vector.

    Args:
        rank_counts:     Remaining card counts per rank bucket (10 buckets).
        true_count:      Hi-Lo true count (running_count / decks_remaining).
        decks_remaining: Decks of cards remaining in the shoe.
    """
    obs = np.zeros(BET_OBS_DIM, dtype=np.float32)

    total = float(rank_counts.sum())
    if total > 0.0:
        obs[:10] = rank_counts.astype(np.float32) / total
    else:
        # Empty shoe — fall back to full-shoe proportions.
        obs[:10] = _FULL_SHOE_COUNTS / _FULL_SHOE_TOTAL

    obs[10] = float(np.clip(true_count, -5.0, 5.0)) / 5.0
    obs[11] = decks_remaining / 6.0

    return obs
