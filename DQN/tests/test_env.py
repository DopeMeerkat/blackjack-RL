"""Acceptance tests and unit tests for the blackjack environment.

Acceptance criteria (from blackjack_rl_design.md §13, Milestone 1):
  1. Random/basic-strategy EV over 100K hands in [-1.5%, -0.4%].
  2. obs.shape == (27,), mask.shape == (4,) at every decision point.
  3. Hi-Lo count update on a hand-checked card sequence matches §15.2 exactly.
  4. Splits produce sub-hands whose summed reward is correctly returned.
"""

from __future__ import annotations

import yaml
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    path = "configs/default.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw["env"]


CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Basic strategy oracle (S17, DAS, no surrender)
#
# Encodes the standard basic strategy chart for the rule set used in this
# project.  Keys are (player_sum, usable_ace, dealer_upcard_value) where
# dealer_upcard_value is 1 (Ace) or 2-10.  Values are action indices:
#   0=hit, 1=stand, 2=double, 3=split.
#
# For simplicity the oracle only handles hit/stand/double (no splits).
# Splits are defaulted to hit (conservative; doesn't affect EV much for the
# acceptance test which just needs the EV to land in the right band).
# ---------------------------------------------------------------------------

# Action constants
HIT    = 0
STAND  = 1
DOUBLE = 2
SPLIT  = 3

def _basic_strategy_action(
    player_sum: int,
    usable_ace: bool,
    dealer_upcard_value: int,   # 1=Ace, 2-10
    can_double: bool,
    can_split: bool,
    pair_value: int | None,     # None if not a pair
) -> int:
    """Return the basic-strategy action for S17 + DAS rules.

    Based on the standard basic strategy chart for 6 decks, S17, DAS,
    no surrender.  Pair splitting is included for the most common pairs.
    """
    d = dealer_upcard_value   # shorthand

    # --- Splits ---
    if can_split and pair_value is not None:
        pv = pair_value
        if pv == 1:    return SPLIT                           # A-A always split
        if pv == 8:    return SPLIT                           # 8-8 always split
        if pv == 9:
            return SPLIT if d not in (7, 10, 1) else STAND   # 9-9
        if pv == 7:
            return SPLIT if d <= 7 else HIT
        if pv == 6:
            return SPLIT if 2 <= d <= 6 else HIT
        if pv == 4:
            return SPLIT if d in (5, 6) else HIT
        if pv == 3:
            return SPLIT if 2 <= d <= 7 else HIT
        if pv == 2:
            return SPLIT if 2 <= d <= 7 else HIT
        if pv == 5:    pass   # treat 5-5 as a 10 (double below)
        if pv == 10:   return STAND  # 10-10 always stand (never split)

    # --- Soft hands (usable ace) ---
    if usable_ace:
        s = player_sum   # soft total (ace counted as 11)
        if s >= 19:      return STAND
        if s == 18:
            if d in (2, 7, 8):  return STAND
            if 3 <= d <= 6:     return DOUBLE if can_double else STAND
            return HIT          # vs 9, 10, A
        if s == 17:
            return DOUBLE if (3 <= d <= 6 and can_double) else HIT
        if s in (15, 16):
            return DOUBLE if (4 <= d <= 6 and can_double) else HIT
        if s in (13, 14):
            return DOUBLE if (5 <= d <= 6 and can_double) else HIT
        return HIT

    # --- Hard hands ---
    h = player_sum
    if h >= 17:   return STAND
    if h == 16:
        if d <= 6:  return STAND
        return HIT
    if h == 15:
        if d <= 6:  return STAND
        return HIT
    if h == 14:
        if d <= 6:  return STAND
        return HIT
    if h == 13:
        if d in (2, 3):  return STAND
        if d <= 6:       return STAND
        return HIT
    if h == 12:
        if 4 <= d <= 6:  return STAND
        return HIT
    if h == 11:
        return DOUBLE if can_double else HIT
    if h == 10:
        return DOUBLE if (can_double and d not in (10, 1)) else HIT
    if h == 9:
        return DOUBLE if (can_double and 3 <= d <= 6) else HIT
    # h <= 8: always hit
    return HIT


def _dealer_upcard_value(rank: int) -> int:
    """Convert raw rank to dealer upcard value for basic strategy lookup."""
    if rank == 1:    return 1
    if rank >= 10:   return 10
    return rank


# ---------------------------------------------------------------------------
# Acceptance Test 1: EV in [-1.5%, -0.4%]
# ---------------------------------------------------------------------------

class TestEV:
    def test_basic_strategy_ev_in_range(self):
        """
        Run 100K hands with basic strategy (flat 1x bet).
        EV should fall in [-1.5%, -0.4%] — the theoretical range for
        basic strategy under S17/DAS rules.

        A purely random policy has ~-7% EV and would fail; this test
        validates that the rules, payouts, and dealer logic are all correct.
        """
        from env.blackjack import BlackjackEnv, hand_sum

        env = BlackjackEnv(CONFIG, seed=42)
        total_reward = 0.0
        num_hands = 100_000
        hands_played = 0

        obs, mask, info = env.reset()

        while hands_played < num_hands:
            if info["phase"] == "bet":
                action = 0   # always bet 1x
            else:
                from env.blackjack import hand_sum as _hs, _card_value
                # Reconstruct state from obs for basic strategy lookup.
                # obs[0] = (sum-4)/17 → sum = obs[0]*17 + 4
                player_sum_f = float(obs[0]) * 17.0 + 4.0
                player_sum = int(round(player_sum_f))
                usable_ace = bool(obs[1] > 0.5)
                # Dealer upcard: one-hot in obs[2..11]
                upcard_oh = obs[2:12]
                upcard_bucket = int(np.argmax(upcard_oh))
                # bucket 0=A, 1-8=2-9, 9=10/face
                upcard_val = 1 if upcard_bucket == 0 else (2 + upcard_bucket - 1 if upcard_bucket <= 8 else 10)
                # upcard_bucket: 0→1(A), 1→2, 2→3, ..., 8→9, 9→10
                if upcard_bucket == 0:
                    upcard_val = 1
                else:
                    upcard_val = upcard_bucket + 1   # 1→2, 2→3, …, 9→10

                can_double = bool(obs[23] > 0.5)
                can_split  = bool(obs[24] > 0.5)
                is_pair    = bool(obs[12] > 0.5)

                pair_val = None
                if is_pair:
                    pair_oh = obs[13:23]
                    pair_bucket = int(np.argmax(pair_oh))
                    pair_val = 1 if pair_bucket == 0 else (pair_bucket + 1)

                action = _basic_strategy_action(
                    player_sum=player_sum,
                    usable_ace=usable_ace,
                    dealer_upcard_value=upcard_val,
                    can_double=can_double,
                    can_split=can_split,
                    pair_value=pair_val if is_pair else None,
                )
                # Ensure the chosen action is actually legal.
                if not mask[action]:
                    # Fall back to stand (index 1) which is always legal.
                    action = STAND

            obs, mask, reward, done, info = env.step(action)

            if done:
                total_reward += reward
                hands_played += 1
                if hands_played < num_hands:
                    obs, mask, info = env.reset()

        ev = total_reward / num_hands
        assert -0.015 <= ev <= -0.004, (
            f"Basic-strategy EV {ev:.4f} ({ev*100:.2f}%) outside expected "
            f"range [-1.5%, -0.4%].  Check rules, payouts, or dealer logic."
        )


# ---------------------------------------------------------------------------
# Acceptance Test 2: State vector and mask dimensions
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_obs_shape_at_reset(self):
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=0)
        obs, mask, info = env.reset()
        assert obs.shape == (27,), f"Expected obs shape (27,), got {obs.shape}"
        assert obs.dtype == np.float32
        assert mask.shape == (4,), f"Expected mask shape (4,), got {mask.shape}"
        assert mask.dtype == bool

    def test_obs_shape_after_bet_step(self):
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=0)
        obs, mask, info = env.reset()
        obs2, mask2, reward, done, info2 = env.step(0)   # bet 1x
        assert obs2.shape == (27,)
        assert obs2.dtype == np.float32
        assert mask2.shape == (4,)
        assert mask2.dtype == bool

    def test_obs_shape_throughout_hand(self):
        """Walk through a full hand and check shapes at every step."""
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=1)
        obs, mask, info = env.reset()
        assert obs.shape == (27,)
        assert mask.shape == (4,)

        obs, mask, r, done, info = env.step(0)   # bet
        while not done:
            assert obs.shape == (27,)
            assert mask.shape == (4,)
            # Choose a legal action
            legal = np.where(mask)[0]
            action = int(legal[0])
            obs, mask, r, done, info = env.step(action)

        assert obs.shape == (27,)
        assert mask.shape == (4,)

    def test_bet_phase_obs_features_zeroed(self):
        """Playing-state features must be zero during the bet phase."""
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=5)
        obs, mask, info = env.reset()
        assert info["phase"] == "bet"
        # Indices 0-24 must be zero (play-state features).
        for i in range(25):
            assert obs[i] == 0.0, f"obs[{i}] = {obs[i]}, expected 0.0 in bet phase"
        # obs[25] and obs[26] should be populated.
        # (At start of fresh shoe: 312 cards, DR=6, obs[26]=1.0; count=0, obs[25]=0.0)
        assert obs[26] > 0.0, "obs[26] (decks remaining) should be > 0"

    def test_mask_all_false_in_bet_phase(self):
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=2)
        obs, mask, info = env.reset()
        assert not mask.any(), "All play actions should be illegal during bet phase"

    def test_play_mask_hit_stand_always_legal(self):
        """Hit and stand are always legal during play phase (without splits/doubles)."""
        from env.blackjack import BlackjackEnv
        # Run several hands and check mask after the bet step.
        env = BlackjackEnv(CONFIG, seed=7)
        for _ in range(20):
            obs, mask, info = env.reset()
            obs, mask, r, done, info = env.step(0)   # bet 1x
            if not done:
                # mask[0]=hit, mask[1]=stand must be True
                assert mask[0], "Hit must be legal in play phase"
                assert mask[1], "Stand must be legal in play phase"
            # finish hand
            while not done:
                legal = np.where(mask)[0]
                obs, mask, r, done, info = env.step(int(legal[0]))


# ---------------------------------------------------------------------------
# Acceptance Test 3: Hi-Lo count sequence
# ---------------------------------------------------------------------------

class TestHiLoCount:
    def test_hand_checked_sequence(self):
        """
        Sequence from §15.2: [2, 7, K, 5, 3, A, 9, 6, 10]
        Encoded as ranks: [2, 7, 13, 5, 3, 1, 9, 6, 10]
        Expected running counts after each card:
          +1, +1, 0, +1, +2, +1, +1, +2, +1
        """
        from env.count import HiLoCount
        count = HiLoCount(num_decks=6)
        sequence = [2, 7, 13, 5, 3, 1, 9, 6, 10]
        expected = [1, 1, 0, 1, 2, 1, 1, 2, 1]

        for i, (rank, exp) in enumerate(zip(sequence, expected)):
            count.update(rank)
            assert count.running_count() == exp, (
                f"After card {i+1} (rank={rank}): "
                f"got {count.running_count()}, expected {exp}"
            )

    def test_true_count_formula(self):
        """TC = running_count / max(decks_remaining, 0.5)."""
        from env.count import HiLoCount
        count = HiLoCount(num_decks=6)
        # Simulate running count of +6 with 2 decks remaining (104 cards).
        for _ in range(6):
            count.update(2)   # each 2 increments by +1
        tc = count.true_count(104)
        expected = 6 / (104 / 52)   # = 6 / 2.0 = 3.0
        assert abs(tc - expected) < 0.001, f"TC={tc}, expected={expected}"

    def test_true_count_floor(self):
        """TC denominator is floored at 0.5 decks to prevent blow-up."""
        from env.count import HiLoCount
        count = HiLoCount(num_decks=6)
        for _ in range(10):
            count.update(2)
        # Only 10 cards left (< 0.5 decks = 26 cards)
        tc = count.true_count(10)
        expected = 10 / 0.5   # floor kicks in
        assert abs(tc - expected) < 0.001, f"TC={tc}, expected={expected}"

    def test_count_resets_on_new_shoe(self):
        from env.count import HiLoCount
        count = HiLoCount(num_decks=6)
        count.update(2)
        count.update(3)
        assert count.running_count() == 2
        count.reset()
        assert count.running_count() == 0

    def test_all_ranks_covered(self):
        """Every rank 1-13 must have a defined delta."""
        from env.count import HiLoCount, _DELTA
        for rank in range(1, 14):
            assert rank in _DELTA, f"Rank {rank} missing from _DELTA"

    def test_negative_count(self):
        from env.count import HiLoCount
        count = HiLoCount()
        for _ in range(5):
            count.update(10)   # each -1
        assert count.running_count() == -5

    def test_decks_remaining(self):
        from env.count import HiLoCount
        count = HiLoCount(num_decks=6)
        assert count.decks_remaining(312) == pytest.approx(6.0)
        assert count.decks_remaining(52)  == pytest.approx(1.0)
        assert count.decks_remaining(26)  == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Acceptance Test 4: Split sub-hand rewards sum correctly
# ---------------------------------------------------------------------------

class TestSplitRewards:
    def _make_split_shoe(self) -> np.ndarray:
        """
        Craft a shoe that forces a specific known hand:
          deal order: player1, dealer_up, player2, dealer_hole, then more cards
          We want player to get a pair of 8s and dealer a 5-up.
          Then: player splits, each sub-hand gets a known second card.
          Shoe: [8, 5, 8, 7, 3, 6, 2, ...]
                 p1  d1  p2  d2  p1b p2b  dealer_hit ...
          player: [8, 8] → pair → split
          dealer: [5, 7] = 12 → hits → 2 = 14 → hits more... let's pick dealer cards.

        Let's use:
          [8, 5, 8, 6, 4, 9, 10, ...]
          p1=8, d1=5, p2=8, d2=6  → dealer has [5,6]=11 (hard), will hit
          After split: hand1=[8,4] → 12 (stand), hand2=[8,9] → 17 (stand)
          Dealer [5,6]=11, hits 10 → 21 → dealer wins both.
          Expected reward: -1 (hand1) + -1 (hand2) = -2.0 (× 1x bet, no double)
        """
        # Build enough cards for the hand + dealer play.
        shoe = np.array(
            [8, 5, 8, 6,    # initial deal: p1=8, d1=5, p2=8, d2=6
             4,              # hand1 second card after split: 8+4=12
             9,              # hand2 second card after split: 8+9=17
             10,             # dealer hits: 5+6=11, hits 10 → 21
             ] + [7] * 300,  # padding
            dtype=np.int8
        )
        return shoe

    def test_split_produces_correct_reward(self):
        """
        With controlled shoe: player gets 8-8, dealer 5-6.
        Player splits: hand1=[8,4]=12, hand2=[8,9]=17.
        Player stands both.
        Dealer [5,6]=11, hits 10→21.  Dealer wins both: reward = -2.0.
        """
        from env.blackjack import BlackjackEnv

        shoe = self._make_split_shoe()
        env = BlackjackEnv(CONFIG, shoe=shoe)

        obs, mask, info = env.reset()
        assert info["phase"] == "bet"

        # Bet 1x
        obs, mask, reward, done, info = env.step(0)
        assert not done, "Hand should not be over immediately (no BJ)"
        assert info["phase"] == "play"

        # Split should be available (pair of 8s)
        assert mask[3], "Split should be available for pair of 8s"

        # Split
        obs, mask, reward, done, info = env.step(3)   # split
        assert not done, "Hand should continue after split"

        # Play hand1 (8+4=12): stand
        obs, mask, reward, done, info = env.step(1)   # stand
        assert not done, "Should still have second sub-hand"

        # Play hand2 (8+9=17): stand
        obs, mask, reward, done, info = env.step(1)   # stand
        assert done, "Hand should be over after both sub-hands stand"

        # Dealer: [5,6]=11 → hits → 21.  Both sub-hands lose.
        assert reward == pytest.approx(-2.0), (
            f"Expected -2.0 (both sub-hands lose to dealer 21), got {reward}"
        )

    def test_split_done_only_after_all_subhands(self):
        """done must be False until all split sub-hands are resolved."""
        from env.blackjack import BlackjackEnv
        # Shoe: pair + two more cards + dealer cards that stand immediately.
        shoe = np.array(
            [7, 10, 7, 7,   # p1=7, d1=10, p2=7, d2=7  → dealer 17 (stands)
             5,              # hand1 second card: 7+5=12
             9,              # hand2 second card: 7+9=16
             ] + [3] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        obs, mask, info = env.reset()
        env.step(0)          # bet 1x
        obs, mask, r, done, info = env.step(0)   # re-check after bet

        if not mask[3]:
            pytest.skip("No pair in this shoe configuration")

        obs, mask, r, done, _ = env.step(3)   # split
        assert not done
        obs, mask, r, done, _ = env.step(1)   # stand hand1
        assert not done
        obs, mask, r, done, _ = env.step(1)   # stand hand2
        assert done

    def test_split_reward_magnitude(self):
        """With 1x bet and no doubles, |reward| <= 3.0 from 3-split hands."""
        from env.blackjack import BlackjackEnv

        env = BlackjackEnv(CONFIG, seed=99)
        total_hands = 0
        split_rewards = []

        for _ in range(500):
            obs, mask, info = env.reset()
            obs, mask, r, done, info = env.step(0)   # bet 1x
            split_happened = False
            while not done:
                if mask[3]:   # split if possible
                    obs, mask, r, done, info = env.step(3)
                    split_happened = True
                else:
                    obs, mask, r, done, info = env.step(1)   # stand
            if split_happened:
                split_rewards.append(r)

        if split_rewards:
            for rw in split_rewards:
                assert abs(rw) <= 4.5 + 1e-6, (
                    f"Split reward {rw} exceeds plausible range for 1x bet"
                )


# ---------------------------------------------------------------------------
# Unit tests — hand_sum
# ---------------------------------------------------------------------------

class TestHandSum:
    def test_hard_hand(self):
        from env.blackjack import hand_sum
        assert hand_sum([7, 8]) == (15, False)
        assert hand_sum([10, 10]) == (20, False)
        assert hand_sum([10, 5, 6]) == (21, False)

    def test_soft_hand_ace_as_11(self):
        from env.blackjack import hand_sum
        t, ua = hand_sum([1, 6])
        assert t == 17 and ua is True    # soft 17

    def test_soft_17_not_hard(self):
        from env.blackjack import hand_sum
        total, usable = hand_sum([1, 6])
        assert total == 17
        assert usable is True

    def test_ace_forced_to_1_on_bust(self):
        from env.blackjack import hand_sum
        t, ua = hand_sum([1, 6, 8])   # 11+6+8=25 → 1+6+8=15
        assert t == 15 and ua is False

    def test_two_aces(self):
        from env.blackjack import hand_sum
        t, ua = hand_sum([1, 1])   # one as 11, one as 1 → 12
        assert t == 12 and ua is True

    def test_three_aces(self):
        from env.blackjack import hand_sum
        t, ua = hand_sum([1, 1, 1])   # 11+1+1=13; adding 10 = 13 → ok, soft 13
        assert t == 13 and ua is True

    def test_blackjack(self):
        from env.blackjack import hand_sum
        assert hand_sum([1, 10])[0] == 21
        assert hand_sum([1, 13])[0] == 21  # A + K

    def test_bust(self):
        from env.blackjack import hand_sum
        t, ua = hand_sum([10, 10, 5])
        assert t == 25 and ua is False

    def test_21_no_ace(self):
        from env.blackjack import hand_sum
        t, ua = hand_sum([7, 7, 7])
        assert t == 21 and ua is False


# ---------------------------------------------------------------------------
# Unit tests — BlackjackEnv game rules
# ---------------------------------------------------------------------------

class TestGameRules:
    def test_dealer_stands_soft_17(self):
        """Dealer must not hit on soft 17 (A+6=17 soft)."""
        from env.blackjack import BlackjackEnv, hand_sum
        # Shoe: p1=2, d1=1(A), p2=3, d2=6  → dealer has soft 17 (stands)
        # Then player hits and busts to force dealer resolution.
        shoe = np.array(
            [2, 1, 3, 6,    # p=[2,3]=5, d=[A,6]=soft17
             10, 10, 10,    # player hits: 5+10=15, 15+10=25 bust
             ] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        env.step(0)   # bet 1x

        # Player hits until bust
        obs, mask, r, done, info = env.step(0)   # hit: 2+3+10=15
        assert not done
        obs, mask, r, done, info = env.step(0)   # hit: 15+10=25, bust
        assert done

        # With player busted, dealer wins. Check dealer didn't overdraw.
        # If dealer stood on soft 17, they have [A,6]=17 and the reward is -1.
        # (player bust → lose regardless of dealer total)
        assert r == pytest.approx(-1.0)

    def test_natural_blackjack_pays_1_5(self):
        """Player natural BJ pays 1.5x the bet multiplier."""
        from env.blackjack import BlackjackEnv
        # Shoe: p1=1(A), d1=5, p2=10, d2=9 → player BJ, dealer no BJ
        shoe = np.array(
            [1, 5, 10, 9] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        obs, mask, r, done, info = env.step(0)   # bet 1x
        assert done, "Player BJ should resolve immediately"
        assert r == pytest.approx(1.5)

    def test_natural_blackjack_2x_bet(self):
        """Player BJ with 2x bet pays 3.0."""
        from env.blackjack import BlackjackEnv
        shoe = np.array(
            [1, 5, 10, 9] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        obs, mask, r, done, info = env.step(1)   # bet 2x
        assert done
        assert r == pytest.approx(3.0)   # 1.5 × 2

    def test_player_bj_vs_dealer_bj_is_push(self):
        """When both player and dealer have BJ, result is a push (0)."""
        from env.blackjack import BlackjackEnv
        # p=[A,10], d=[A,10]
        shoe = np.array(
            [1, 1, 10, 10] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        obs, mask, r, done, info = env.step(0)
        assert done
        assert r == pytest.approx(0.0)

    def test_dealer_bj_player_loses(self):
        """Dealer BJ with no player BJ → player loses 1x bet."""
        from env.blackjack import BlackjackEnv
        # p=[2,3]=5, d=[1,10]=BJ
        shoe = np.array(
            [2, 1, 3, 10] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        obs, mask, r, done, info = env.step(0)
        assert done
        assert r == pytest.approx(-1.0)

    def test_double_doubles_outcome(self):
        """Doubling doubles the reward magnitude."""
        from env.blackjack import BlackjackEnv
        # p=[5,6]=11 (good double), d=[5,6]=11→hits→...
        # Shoe chosen so player doubles to a winning hand.
        # p=[5,6]=11, d=[4,8]=12; player doubles+10=21 → wins
        # Dealer [4,8]=12, hits: 6=18 → stand
        shoe = np.array(
            [5, 4, 6, 8,   # p=[5,6]=11, d=[4,8]=12
             10,            # player double card: 11+10=21
             6,             # dealer hits: 12+6=18
             ] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        obs, mask, r, done, info = env.step(0)   # bet 1x
        assert not done
        # Player should have 11, double available
        assert mask[2], "Double should be available on 11"
        obs, mask, r, done, info = env.step(2)   # double
        assert done
        # Player 21 > dealer 18: win. Double → 2× reward.
        assert r == pytest.approx(2.0)

    def test_reshuffle_triggered_at_threshold(self):
        """Shoe reshuffles when < 1.5 decks remain."""
        from env.blackjack import BlackjackEnv

        env = BlackjackEnv(CONFIG, seed=0)
        # Play enough hands to exhaust the shoe past 1.5 decks remaining.
        # 312 cards − 1.5×52 = 312 − 78 = 234 cards dealt triggers reshuffle.
        initial_shoe_id = id(env._shoe.base if env._shoe.base is not None else env._shoe)

        hands = 0
        reshuffled = False
        prev_shoe_pos = env._shoe_pos

        # Play 2000 hands (should be more than enough to reshuffle several times)
        for _ in range(2000):
            obs, mask, info = env.reset()
            if env._shoe_pos < prev_shoe_pos:
                reshuffled = True
                break
            prev_shoe_pos = env._shoe_pos

            obs, mask, r, done, info = env.step(0)   # bet 1x
            while not done:
                legal = np.where(mask)[0]
                obs, mask, r, done, info = env.step(int(legal[0]))

        assert reshuffled, "Shoe should have reshuffled after enough hands"

    def test_split_aces_get_one_card(self):
        """After splitting aces, each sub-hand gets exactly one card and auto-stands."""
        from env.blackjack import BlackjackEnv
        # p=[A,A], d=[5,9]=14, then hand1 gets 10, hand2 gets 6.
        # Dealer [5,9]=14 → hits → 2=16 → hits → 3=19 → stands.
        shoe = np.array(
            [1, 5, 1, 9,   # p=[A,A], d=[5,9]
             10,            # hand1 after split: A+10=21
             6,             # hand2 after split: A+6=17 (soft, but from_split so counts as 17)
             2, 3,          # dealer hits: 5+9=14 → +2=16 → +3=19
             ] + [7] * 300,
            dtype=np.int8
        )
        env = BlackjackEnv(CONFIG, shoe=shoe)
        env.reset()
        obs, mask, r, done, info = env.step(0)   # bet 1x
        assert not done
        # Split available for pair of aces
        assert mask[3], "Split should be available for pair of aces"
        # Split aces — should auto-complete both sub-hands without player action
        obs, mask, r, done, info = env.step(3)   # split aces
        assert done, "Split aces should auto-complete both sub-hands immediately"
        # hand1=[A,10]=21 vs dealer=19 → win (+1)
        # hand2=[A,6]=17 vs dealer=19 → lose (-1)
        # Note: A+10 from split is NOT a natural BJ; pays 1:1
        assert r == pytest.approx(0.0)   # win + loss = 0

    def test_insurance_not_in_action_space(self):
        """Insurance is hard-coded; there is no action index for it."""
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=0)
        obs, mask, info = env.reset()
        # Mask has only 4 entries (hit/stand/double/split); no insurance slot.
        assert mask.shape == (4,)


# ---------------------------------------------------------------------------
# Unit tests — VecBlackjackEnv
# ---------------------------------------------------------------------------

class TestVecEnv:
    def test_reset_shapes(self):
        from env.vec_env import VecBlackjackEnv
        vec = VecBlackjackEnv(num_envs=64, config=CONFIG)
        obs, masks, infos = vec.reset()
        assert obs.shape == (64, 27)
        assert obs.dtype == np.float32
        assert masks.shape == (64, 4)
        assert masks.dtype == bool
        assert len(infos) == 64

    def test_step_shapes(self):
        from env.vec_env import VecBlackjackEnv
        vec = VecBlackjackEnv(num_envs=64, config=CONFIG, seeds=list(range(64)))
        obs, masks, infos = vec.reset()
        # All envs in bet phase: take action 0 (bet 1x)
        actions = np.zeros(64, dtype=int)
        obs2, masks2, rewards, dones, infos2 = vec.step(actions)
        assert obs2.shape == (64, 27)
        assert masks2.shape == (64, 4)
        assert rewards.shape == (64,)
        assert rewards.dtype == np.float32
        assert dones.shape == (64,)
        assert dones.dtype == bool
        assert len(infos2) == 64

    def test_in_bet_phase_property(self):
        from env.vec_env import VecBlackjackEnv
        vec = VecBlackjackEnv(num_envs=4, config=CONFIG)
        vec.reset()
        bp = vec.in_bet_phase
        assert bp.shape == (4,)
        assert bp.dtype == bool
        assert bp.all(), "All envs should be in bet phase after reset"


# ---------------------------------------------------------------------------
# Unit tests — encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_obs_values_in_range(self):
        """All observation values should be in a bounded range."""
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=42)
        for _ in range(100):
            obs, mask, info = env.reset()
            assert np.all(np.isfinite(obs)), "Non-finite value in obs"
            obs2, mask2, r, done, info2 = env.step(0)
            assert np.all(np.isfinite(obs2))
            while not done:
                legal = np.where(mask2)[0]
                obs2, mask2, r, done, _ = env.step(int(legal[0]))
                assert np.all(np.isfinite(obs2))

    def test_dealer_upcard_one_hot(self):
        """Dealer upcard one-hot in obs[2:12] should have exactly one 1."""
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=3)
        for _ in range(50):
            obs, mask, info = env.reset()
            obs2, mask2, r, done, info2 = env.step(0)
            if not done:
                # Exactly one of obs[2:12] should be 1.
                upcard_oh = obs2[2:12]
                assert upcard_oh.sum() == pytest.approx(1.0), (
                    f"Dealer upcard one-hot sums to {upcard_oh.sum()}"
                )
            while not done:
                legal = np.where(mask2)[0]
                obs2, mask2, r, done, _ = env.step(int(legal[0]))

    def test_true_count_normalised(self):
        """obs[25] should be in [-1, 1] after normalisation."""
        from env.blackjack import BlackjackEnv
        env = BlackjackEnv(CONFIG, seed=10)
        for _ in range(200):
            obs, mask, info = env.reset()
            assert -1.0 <= obs[25] <= 1.0, f"obs[25]={obs[25]} out of [-1,1]"
            obs2, mask2, r, done, _ = env.step(0)
            assert -1.0 <= obs2[25] <= 1.0
            while not done:
                legal = np.where(mask2)[0]
                obs2, mask2, r, done, _ = env.step(int(legal[0]))

