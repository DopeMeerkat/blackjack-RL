"""Hi-Lo running count tracker.

Card ranks use the shoe convention: 1=Ace, 2-9=pip value, 10=Ten, 11=Jack,
12=Queen, 13=King.  All 10-value cards (10, J, Q, K) and Aces decrement the
count; low cards (2-6) increment it; neutral cards (7-9) leave it unchanged.

See blackjack_rl_design.md §15.2 for the authoritative specification.
"""

# Hi-Lo delta per rank bucket:
#   ranks 2-6  → +1
#   ranks 7-9  → 0
#   ranks 1, 10-13 → -1
_DELTA: dict[int, int] = {
    1: -1,   # Ace
    2: +1,
    3: +1,
    4: +1,
    5: +1,
    6: +1,
    7: 0,
    8: 0,
    9: 0,
    10: -1,
    11: -1,  # Jack
    12: -1,  # Queen
    13: -1,  # King
}


class HiLoCount:
    """Maintains the Hi-Lo running count across a shoe.

    The running count persists between hands within a shoe and resets only
    when the shoe is reshuffled (via ``reset()``).
    """

    def __init__(self, num_decks: int = 6) -> None:
        self._num_decks = num_decks
        self._running_count: int = 0

    def reset(self) -> None:
        """Reset running count to zero (call on shoe reshuffle)."""
        self._running_count = 0

    def update(self, card_rank: int) -> None:
        """Update running count for one newly revealed card.

        Args:
            card_rank: Integer rank in [1, 13].  1=Ace, 2-9=pip, 10=Ten,
                       11=Jack, 12=Queen, 13=King.
        """
        self._running_count += _DELTA[card_rank]

    def running_count(self) -> int:
        """Return the current running count."""
        return self._running_count

    def decks_remaining(self, cards_in_shoe: int) -> float:
        """Return estimated decks remaining given cards left in the shoe."""
        return cards_in_shoe / 52.0

    def true_count(self, cards_in_shoe: int) -> float:
        """Return the true count: running_count / max(decks_remaining, 0.5).

        The floor of 0.5 decks prevents division blow-ups near the cut card.
        """
        dr = max(self.decks_remaining(cards_in_shoe), 0.5)
        return self._running_count / dr
