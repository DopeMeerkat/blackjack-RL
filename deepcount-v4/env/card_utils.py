"""
Card utilities for the DeepCount blackjack environment.

Internal card representation (used by game logic):
  1=Ace, 2-10=pip cards, 11=Jack, 12=Queen, 13=King

Observation token representation (used by the neural network):
  0=UNSEEN (pad), 1=Ace, 2-9=pip cards, 10=any ten-value card (10/J/Q/K)

Jack, Queen, and King are collapsed to token 10 via normalize_rank().
They are functionally identical in blackjack — same point value, same
Hi-Lo count contribution, same splitting eligibility — so giving the
model four distinct tokens for them would only waste capacity and force
it to learn their equivalence from scratch.

VOCAB_SIZE = 11  (tokens 0-10)
"""

import numpy as np

# Token for padding / unseen position
UNSEEN = 0

# Tokens 0-10: UNSEEN + Ace + 2-9 + ten-value
VOCAB_SIZE = 11

# Standard 6-deck shoe
NUM_DECKS = 6
CARDS_PER_DECK = 52
SHOE_SIZE = NUM_DECKS * CARDS_PER_DECK  # 312

# Penetration: reshuffle after ~75% of shoe is dealt
PENETRATION = 0.75
RESHUFFLE_THRESHOLD = int(SHOE_SIZE * PENETRATION)  # ~234


def card_blackjack_value(rank: int) -> int:
    """Return the hard blackjack point value for a card rank."""
    if rank == 0:
        return 0  # UNSEEN token
    return min(rank, 10)  # Ace=1 (hard), 2-10=face, J/Q/K=10


def normalize_rank(rank: int) -> int:
    """
    Map an internal card rank to its observation token.

    10, J (11), Q (12), K (13) → 10  (all ten-value cards)
    Everything else is unchanged.
    0 (UNSEEN pad) is preserved as 0.
    """
    if rank >= 10:
        return 10
    return rank


def hand_value(ranks: list[int]) -> tuple[int, bool]:
    """
    Compute the best blackjack total for a hand.

    Returns:
        (total, soft) where soft=True if an Ace is being counted as 11.
    """
    total = sum(card_blackjack_value(r) for r in ranks)
    aces = ranks.count(1)  # Aces
    soft = False
    # Try to use one Ace as 11
    if aces > 0 and total + 10 <= 21:
        total += 10
        soft = True
    return total, soft


def is_bust(ranks: list[int]) -> bool:
    total, _ = hand_value(ranks)
    return total > 21


def is_blackjack(ranks: list[int]) -> bool:
    """Natural blackjack: exactly two cards summing to 21."""
    return len(ranks) == 2 and hand_value(ranks)[0] == 21


def hilo_count(card_history: list[int]) -> float:
    """
    Compute the Hi-Lo running count from a card history.
    +1 for 2-6, -1 for 10/J/Q/K/A, 0 for 7-9.
    """
    count = 0
    for rank in card_history:
        if rank == 0:
            continue  # Skip UNSEEN tokens
        val = card_blackjack_value(rank)
        if 2 <= val <= 6:
            count += 1
        elif val == 10 or rank == 1:  # 10-value or Ace
            count -= 1
    return float(count)


def true_count(card_history: list[int]) -> float:
    """
    Hi-Lo true count: running count / decks remaining.
    Clamped to [-10, +10] for stability.
    """
    running = hilo_count(card_history)
    cards_seen = sum(1 for r in card_history if r != 0)
    decks_remaining = max((SHOE_SIZE - cards_seen) / 52.0, 0.5)
    tc = running / decks_remaining
    return float(np.clip(tc, -10.0, 10.0))


def build_shoe() -> list[int]:
    """Create a fresh 6-deck shoe (unshuffled)."""
    single_deck = list(range(1, 14)) * 4  # 52 cards
    return single_deck * NUM_DECKS


def shuffle_shoe(rng: np.random.Generator) -> list[int]:
    """Return a freshly shuffled 6-deck shoe."""
    shoe = build_shoe()
    rng.shuffle(shoe)
    return shoe


def can_split(hand: list[int]) -> bool:
    """Can the hand be split? Requires exactly 2 cards of equal BJ value."""
    if len(hand) != 2:
        return False
    return card_blackjack_value(hand[0]) == card_blackjack_value(hand[1])
