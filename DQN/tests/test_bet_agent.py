"""Tests for BetReplayBuffer, BetAgent, and encode_bet_obs."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# encode_bet_obs tests
# ---------------------------------------------------------------------------

class TestEncodeBetObs:
    def _call(self, rank_counts=None, true_count=0.0, decks_remaining=3.0):
        from env.bet_encoding import encode_bet_obs, BET_OBS_DIM
        if rank_counts is None:
            rank_counts = np.array([24, 24, 24, 24, 24, 24, 24, 24, 24, 96], dtype=np.int32)
        return encode_bet_obs(rank_counts, true_count, decks_remaining), BET_OBS_DIM

    def test_output_shape(self):
        obs, dim = self._call()
        assert obs.shape == (dim,)
        assert obs.dtype == np.float32

    def test_composition_sums_to_one(self):
        obs, _ = self._call()
        assert abs(obs[:10].sum() - 1.0) < 1e-5

    def test_composition_fractions_correct(self):
        rank_counts = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 10], dtype=np.int32)
        obs, _ = self._call(rank_counts)
        assert abs(obs[0] - 0.5) < 1e-5
        assert abs(obs[9] - 0.5) < 1e-5
        assert obs[1:9].sum() < 1e-5

    def test_true_count_normalized(self):
        obs_pos, _ = self._call(true_count=5.0)
        assert abs(obs_pos[10] - 1.0) < 1e-5

        obs_neg, _ = self._call(true_count=-5.0)
        assert abs(obs_neg[10] - (-1.0)) < 1e-5

    def test_true_count_clipped(self):
        obs_high, _ = self._call(true_count=10.0)
        assert abs(obs_high[10] - 1.0) < 1e-5

        obs_low, _ = self._call(true_count=-10.0)
        assert abs(obs_low[10] - (-1.0)) < 1e-5

    def test_decks_remaining_normalized(self):
        obs, _ = self._call(decks_remaining=6.0)
        assert abs(obs[11] - 1.0) < 1e-5

        obs_half, _ = self._call(decks_remaining=3.0)
        assert abs(obs_half[11] - 0.5) < 1e-5

    def test_empty_shoe_fallback(self):
        from env.bet_encoding import _FULL_SHOE_COUNTS, _FULL_SHOE_TOTAL
        zero_counts = np.zeros(10, dtype=np.int32)
        obs, _ = self._call(zero_counts)
        expected = _FULL_SHOE_COUNTS / _FULL_SHOE_TOTAL
        np.testing.assert_allclose(obs[:10], expected, atol=1e-6)

    def test_full_shoe_composition_matches_expected(self):
        """Full-shoe rank fractions: A-9 ≈ 7.69%, 10-group ≈ 30.77%."""
        rank_counts = np.array([24, 24, 24, 24, 24, 24, 24, 24, 24, 96], dtype=np.int32)
        obs, _ = self._call(rank_counts)
        np.testing.assert_allclose(obs[:9],  np.full(9,  24 / 312), atol=1e-5)
        np.testing.assert_allclose(obs[9],   96 / 312,               atol=1e-5)


# ---------------------------------------------------------------------------
# BetReplayBuffer tests
# ---------------------------------------------------------------------------

class TestBetReplayBuffer:
    def _make(self, capacity=100, obs_dim=12):
        from agent.bet_agent import BetReplayBuffer
        return BetReplayBuffer(capacity, obs_dim)

    def test_len_increases_on_add(self):
        buf = self._make()
        assert len(buf) == 0
        buf.add(np.zeros(12, np.float32), 0, 1.0)
        assert len(buf) == 1

    def test_len_caps_at_capacity(self):
        buf = self._make(capacity=10)
        for i in range(20):
            buf.add(np.zeros(12, np.float32), 0, 0.0)
        assert len(buf) == 10

    def test_sample_shapes(self):
        buf = self._make()
        for _ in range(50):
            buf.add(np.random.randn(12).astype(np.float32), 1, 0.5)
        batch = buf.sample(16)
        assert batch["obs"].shape     == (16, 12)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)

    def test_circular_overwrite(self):
        """After capacity overflow the oldest entries are overwritten."""
        buf = self._make(capacity=5, obs_dim=12)
        for i in range(10):
            obs = np.full(12, float(i), dtype=np.float32)
            buf.add(obs, 0, float(i))
        # Buffer should hold the most recent 5 entries (indices 5..9).
        stored = set(buf.obs[:, 0].tolist())
        assert stored.issubset({5.0, 6.0, 7.0, 8.0, 9.0})

    def test_sample_raises_if_empty(self):
        buf = self._make()
        with pytest.raises(Exception):
            buf.sample(1)


# ---------------------------------------------------------------------------
# BetAgent tests
# ---------------------------------------------------------------------------

class _BetAgentFixture:
    _CFG = {
        "bet_obs_dim":        12,
        "bet_hidden":         32,
        "bet_actions":        5,
        "bet_lr":             1e-3,
        "bet_batch_size":     32,
        "bet_replay_cap":     1_000,
        "bet_eps_start":      1.0,
        "bet_eps_end":        0.05,
        "bet_eps_decay_hands": 10_000,
    }

    def _make(self):
        from agent.bet_agent import BetAgent
        return BetAgent(self._CFG, torch.device("cpu"))

    def _obs(self):
        return np.random.randn(12).astype(np.float32)

    def _fill(self, agent, n=50):
        for _ in range(n):
            agent.add_transition(self._obs(), int(np.random.randint(5)), float(np.random.randn()))


class TestBetAgentEpsilon(_BetAgentFixture):
    def test_epsilon_starts_at_eps_start(self):
        agent = self._make()
        assert abs(agent.epsilon - self._CFG["bet_eps_start"]) < 1e-6

    def test_epsilon_decreases_with_hands(self):
        agent = self._make()
        eps0 = agent.epsilon
        self._fill(agent, n=100)
        assert agent.epsilon < eps0

    def test_epsilon_floors_at_eps_end(self):
        agent = self._make()
        # Simulate far more hands than the decay horizon.
        self._fill(agent, n=self._CFG["bet_eps_decay_hands"] * 2)
        assert abs(agent.epsilon - self._CFG["bet_eps_end"]) < 1e-6

    def test_epsilon_linear_midpoint(self):
        agent = self._make()
        self._fill(agent, n=self._CFG["bet_eps_decay_hands"] // 2)
        expected = (self._CFG["bet_eps_start"] + self._CFG["bet_eps_end"]) / 2
        assert abs(agent.epsilon - expected) < 1e-5


class TestBetAgentActions(_BetAgentFixture):
    def test_select_action_range(self):
        agent = self._make()
        for _ in range(20):
            a = agent.select_action(self._obs())
            assert 0 <= a < 5

    def test_select_action_is_deterministic(self):
        agent = self._make()
        obs = self._obs()
        a1 = agent.select_action(obs)
        a2 = agent.select_action(obs)
        assert a1 == a2

    def test_select_actions_batch_shape(self):
        agent = self._make()
        obs_batch = np.random.randn(16, 12).astype(np.float32)
        actions = agent.select_actions_batch(obs_batch, greedy=False)
        assert actions.shape == (16,)

    def test_select_actions_batch_greedy_shape(self):
        agent = self._make()
        obs_batch = np.random.randn(16, 12).astype(np.float32)
        actions = agent.select_actions_batch(obs_batch, greedy=True)
        assert actions.shape == (16,)

    def test_select_actions_batch_valid_range(self):
        agent = self._make()
        obs_batch = np.random.randn(32, 12).astype(np.float32)
        for greedy in (True, False):
            actions = agent.select_actions_batch(obs_batch, greedy=greedy)
            assert actions.min() >= 0
            assert actions.max() < 5

    def test_greedy_batch_is_deterministic(self):
        agent = self._make()
        obs_batch = np.random.randn(16, 12).astype(np.float32)
        a1 = agent.select_actions_batch(obs_batch, greedy=True)
        a2 = agent.select_actions_batch(obs_batch, greedy=True)
        np.testing.assert_array_equal(a1, a2)

    def test_eps1_exploration_is_uniform(self):
        """At eps=1 all actions should appear across a large batch."""
        from agent.bet_agent import BetAgent
        cfg = dict(self._CFG)
        cfg["bet_eps_start"] = 1.0
        cfg["bet_eps_end"]   = 1.0   # keep at 1 regardless of hands
        agent = BetAgent(cfg, torch.device("cpu"))
        obs_batch = np.random.randn(1000, 12).astype(np.float32)
        actions = agent.select_actions_batch(obs_batch, greedy=False)
        unique = set(actions.tolist())
        assert len(unique) == 5, f"Expected all 5 actions, got {unique}"


class TestBetAgentTraining(_BetAgentFixture):
    def test_train_step_returns_none_when_buffer_small(self):
        agent = self._make()
        self._fill(agent, n=10)  # less than batch_size=32
        assert agent.train_step() is None

    def test_train_step_returns_float_when_ready(self):
        agent = self._make()
        self._fill(agent, n=50)
        loss = agent.train_step()
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_decreases_over_many_steps(self):
        """Loss should trend downward over 200 gradient steps on a fixed dataset."""
        agent = self._make()
        self._fill(agent, n=200)
        losses = [agent.train_step() for _ in range(200)]
        losses = [l for l in losses if l is not None]
        assert losses[-1] < losses[0] * 2, "Loss did not decrease at all"

    def test_train_step_increments_counter(self):
        agent = self._make()
        self._fill(agent, n=50)
        before = agent._train_steps
        agent.train_step()
        assert agent._train_steps == before + 1

    def test_add_transition_increments_hands_seen(self):
        agent = self._make()
        assert agent._hands_seen == 0
        agent.add_transition(self._obs(), 0, 0.5)
        assert agent._hands_seen == 1


class TestBetAgentCheckpoint(_BetAgentFixture):
    def test_state_dict_roundtrip(self):
        agent = self._make()
        self._fill(agent, n=50)
        agent.train_step()
        sd = agent.state_dict()

        from agent.bet_agent import BetAgent
        agent2 = BetAgent(self._CFG, torch.device("cpu"))
        agent2.load_state_dict(sd)

        obs = self._obs()
        a1 = agent.select_action(obs)
        a2 = agent2.select_action(obs)
        assert a1 == a2

    def test_state_dict_has_expected_keys(self):
        agent = self._make()
        sd = agent.state_dict()
        assert "net"         in sd
        assert "optimizer"   in sd
        assert "train_steps" in sd
        assert "hands_seen"  in sd

    def test_train_steps_preserved(self):
        agent = self._make()
        self._fill(agent, n=50)
        for _ in range(3):
            agent.train_step()
        sd = agent.state_dict()

        from agent.bet_agent import BetAgent
        agent2 = BetAgent(self._CFG, torch.device("cpu"))
        agent2.load_state_dict(sd)
        assert agent2._train_steps == 3

    def test_hands_seen_preserved(self):
        agent = self._make()
        self._fill(agent, n=42)
        sd = agent.state_dict()

        from agent.bet_agent import BetAgent
        agent2 = BetAgent(self._CFG, torch.device("cpu"))
        agent2.load_state_dict(sd)
        assert agent2._hands_seen == 42
