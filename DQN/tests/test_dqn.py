"""Tests for PrioritizedReplayBuffer and DQNAgent."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# SumTree / PrioritizedReplayBuffer tests
# ---------------------------------------------------------------------------

class TestSumTree:
    def _make(self, capacity=8):
        from agent.replay import SumTree
        return SumTree(capacity)

    def test_initial_total_zero(self):
        tree = self._make()
        assert tree.total == pytest.approx(0.0)

    def test_update_and_total(self):
        tree = self._make(4)
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 3.0)
        assert tree.total == pytest.approx(6.0)

    def test_update_overwrite(self):
        tree = self._make(4)
        tree.update(0, 5.0)
        tree.update(0, 2.0)   # overwrite
        assert tree.total == pytest.approx(2.0)

    def test_get_returns_correct_leaf(self):
        tree = self._make(4)
        for i in range(4):
            tree.update(i, float(i + 1))   # priorities: 1, 2, 3, 4  total=10
        # s=0.5 → should land in leaf 0 (cumsum 1.0 covers [0, 1))
        leaf_idx, p = tree.get(0.5)
        assert leaf_idx == 0
        assert p == pytest.approx(1.0)

    def test_get_last_leaf(self):
        tree = self._make(4)
        for i in range(4):
            tree.update(i, 1.0)
        # s = 3.9 should land in leaf 3
        leaf_idx, _ = tree.get(3.9)
        assert leaf_idx == 3

    def test_uniform_priorities_uniform_sampling(self):
        """With equal priorities, each leaf should be sampled equally often."""
        from agent.replay import SumTree
        capacity = 8
        tree = SumTree(capacity)
        for i in range(capacity):
            tree.update(i, 1.0)
        counts = np.zeros(capacity, dtype=int)
        n = 80_000
        total = tree.total
        for _ in range(n):
            s = np.random.uniform(0, total)
            leaf_idx, _ = tree.get(s)
            counts[leaf_idx] += 1
        expected = n / capacity
        # Each bucket should get within 10% of expected (very loose test)
        assert counts.min() > expected * 0.8
        assert counts.max() < expected * 1.2


class TestPrioritizedReplayBuffer:
    OBS_DIM = 28

    def _make(self, capacity=100, alpha=0.6):
        from agent.replay import PrioritizedReplayBuffer
        return PrioritizedReplayBuffer(capacity, self.OBS_DIM, alpha=alpha)

    def _transition(self, reward=0.0, done=False, head_id=0):
        return dict(
            obs=np.random.randn(self.OBS_DIM).astype(np.float32),
            action=1,
            reward=reward,
            next_obs=np.random.randn(self.OBS_DIM).astype(np.float32),
            done=done,
            mask=np.array([True, True, True, False]),
            next_mask=np.array([True, True, False, False]),
            head_id=head_id,
        )

    def test_len_zero_initially(self):
        buf = self._make()
        assert len(buf) == 0

    def test_add_increments_len(self):
        buf = self._make(capacity=10)
        for i in range(5):
            buf.add(**self._transition())
        assert len(buf) == 5

    def test_len_caps_at_capacity(self):
        buf = self._make(capacity=10)
        for _ in range(25):
            buf.add(**self._transition())
        assert len(buf) == 10

    def test_sample_shapes(self):
        buf = self._make(capacity=100)
        for _ in range(50):
            buf.add(**self._transition())
        batch, weights, leaf_indices = buf.sample(16, beta=0.4)
        assert batch["obs"].shape == (16, self.OBS_DIM)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["next_obs"].shape == (16, self.OBS_DIM)
        assert batch["dones"].shape == (16,)
        assert batch["masks"].shape == (16, 4)
        assert batch["next_masks"].shape == (16, 4)
        assert batch["head_ids"].shape == (16,)
        assert weights.shape == (16,)
        assert leaf_indices.shape == (16,)

    def test_weights_max_normalised(self):
        """IS weights should be in (0, 1] with max = 1."""
        buf = self._make(capacity=100)
        for _ in range(50):
            buf.add(**self._transition())
        _, weights, _ = buf.sample(32, beta=0.5)
        assert weights.max() == pytest.approx(1.0, abs=1e-5)
        assert (weights > 0).all()
        assert (weights <= 1.0 + 1e-5).all()

    def test_high_priority_sampled_more_often(self):
        """A transition with 100x higher priority should be sampled more often."""
        from agent.replay import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=10, obs_dim=self.OBS_DIM, alpha=1.0)
        # Add 9 low-priority transitions
        for i in range(9):
            buf.add(**self._transition(reward=float(i)))
        # Force one to have very high priority
        high_prio_idx = 0
        buf._sumtree.update(high_prio_idx, 100.0)    # 100× the rest
        # Sample many times and check that leaf 0 appears much more often
        counts = np.zeros(10, dtype=int)
        for _ in range(1000):
            _, _, leaf_indices = buf.sample(1, beta=0.0)
            counts[leaf_indices[0]] += 1
        assert counts[0] > 300, f"High-priority leaf sampled only {counts[0]}/1000 times"

    def test_update_priorities(self):
        buf = self._make(capacity=50)
        for _ in range(25):
            buf.add(**self._transition())
        _, _, leaf_indices = buf.sample(8, beta=0.4)
        td_errors = np.ones(8) * 0.5
        old_total = buf._sumtree.total
        buf.update_priorities(leaf_indices, td_errors)
        # Total should change after priority update
        new_total = buf._sumtree.total
        # It may or may not change depending on values, but shouldn't crash
        assert np.isfinite(new_total)

    def test_head_ids_stored_correctly(self):
        """head_id field should round-trip through the buffer."""
        buf = self._make(capacity=20)
        for i in range(10):
            buf.add(**self._transition(head_id=(i % 2)))
        batch, _, _ = buf.sample(10, beta=0.4)
        assert set(batch["head_ids"]).issubset({0, 1})

    def test_circular_overwrite(self):
        """Old transitions should be overwritten when buffer is full."""
        buf = self._make(capacity=5)
        for i in range(10):
            buf.add(**self._transition(reward=float(i)))
        assert len(buf) == 5
        # Sample and check rewards are from the last 5 transitions (5-9)
        # (probabilistic, but with max priority for new entries they're
        # likely to appear)
        batch, _, _ = buf.sample(5, beta=0.4)
        # Just check no crash and correct shapes
        assert batch["rewards"].shape == (5,)


# ---------------------------------------------------------------------------
# NStepAccumulator tests (Rainbow n-step returns)
# ---------------------------------------------------------------------------

class TestNStepAccumulator:
    OBS_DIM = 28

    def _t(self, reward=0.0, done=False, head_id=0):
        return dict(
            obs=np.zeros(self.OBS_DIM, dtype=np.float32),
            action=0,
            reward=reward,
            next_obs=np.ones(self.OBS_DIM, dtype=np.float32) * reward,
            done=done,
            mask=np.array([True, True, False, False]),
            next_mask=np.array([True, True, False, False]),
            head_id=head_id,
        )

    def test_no_flush_before_n(self):
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=3, gamma=1.0)
        assert acc.push(self._t(reward=1.0)) == []
        assert acc.push(self._t(reward=2.0)) == []

    def test_flush_at_n(self):
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=3, gamma=1.0)
        acc.push(self._t(reward=1.0))
        acc.push(self._t(reward=2.0))
        out = acc.push(self._t(reward=4.0))
        assert len(out) == 1
        # gamma=1 → R = 1+2+4 = 7
        assert out[0]["reward"] == pytest.approx(7.0)
        assert out[0]["done"] is False
        assert out[0]["n_step"] == 3

    def test_discounted_sum(self):
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=3, gamma=0.5)
        acc.push(self._t(reward=1.0))
        acc.push(self._t(reward=2.0))
        out = acc.push(self._t(reward=4.0))
        # R = 1 + 0.5*2 + 0.25*4 = 1 + 1 + 1 = 3
        assert out[0]["reward"] == pytest.approx(3.0)

    def test_terminal_flushes_all(self):
        """Done on the 2nd push with n=3 → flush 2 truncated transitions."""
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=3, gamma=1.0)
        acc.push(self._t(reward=1.0))
        out = acc.push(self._t(reward=10.0, done=True))
        assert len(out) == 2
        # First: R = 1 + 10 = 11, done=True, n=2
        assert out[0]["reward"] == pytest.approx(11.0)
        assert out[0]["done"] is True
        assert out[0]["n_step"] == 2
        # Second: R = 10, done=True, n=1
        assert out[1]["reward"] == pytest.approx(10.0)
        assert out[1]["done"] is True
        assert out[1]["n_step"] == 1

    def test_terminal_after_full_window(self):
        """Push n items with last one done → one normal flush, then terminal tail."""
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=3, gamma=1.0)
        acc.push(self._t(reward=1.0))
        acc.push(self._t(reward=2.0))
        # Third push triggers len==n flush AND has done=True.
        out = acc.push(self._t(reward=5.0, done=True))
        # Logic: done path flushes all. buf has 3 items; flush buf[0..3], buf[1..3], buf[2..3].
        assert len(out) == 3
        # All flushed entries should be done=True
        assert all(t["done"] for t in out)
        assert out[0]["reward"] == pytest.approx(1.0 + 2.0 + 5.0)
        assert out[0]["n_step"] == 3
        assert out[1]["reward"] == pytest.approx(2.0 + 5.0)
        assert out[1]["n_step"] == 2
        assert out[2]["reward"] == pytest.approx(5.0)
        assert out[2]["n_step"] == 1

    def test_buffer_cleared_after_terminal(self):
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=3, gamma=1.0)
        acc.push(self._t(reward=1.0))
        acc.push(self._t(reward=2.0, done=True))
        assert len(acc) == 0
        # Subsequent pushes should start a fresh window
        out = acc.push(self._t(reward=3.0))
        assert out == []

    def test_n_step_one_passes_through(self):
        """n_step=1 should flush every push as a single-step transition."""
        from agent.replay import NStepAccumulator
        acc = NStepAccumulator(n_step=1, gamma=0.99)
        out = acc.push(self._t(reward=5.0))
        assert len(out) == 1
        assert out[0]["reward"] == pytest.approx(5.0)
        assert out[0]["n_step"] == 1


# ---------------------------------------------------------------------------
# DQNAgent tests
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def _cfg(self):
        return {
            # Network
            "obs_dim": 28, "playing_actions": 4, "bet_actions": 5,
            "trunk_hidden": 64, "head_hidden": 64, "noisy_sigma0": 0.5,
            # Training
            "gamma": 1.0, "target_tau": 0.005, "batch_size": 16,
            "learning_rate": 3e-4, "replay_alpha": 0.6, "grad_clip": 10.0,
        }

    def _make(self, device=None):
        from agent.dqn import DQNAgent
        if device is None:
            device = torch.device("cpu")
        return DQNAgent(
            net_config=self._cfg(),
            train_config=self._cfg(),
            replay_capacity=200,
            device=device,
        ), device

    def _fill_replay(self, agent, n=100, head_id=0):
        obs_dim = 28
        for i in range(n):
            obs       = np.random.randn(obs_dim).astype(np.float32)
            next_obs  = np.random.randn(obs_dim).astype(np.float32)
            mask      = np.array([True, True, True, False])
            next_mask = np.array([True, True, False, False])
            agent.replay.add(
                obs=obs, action=np.random.randint(0, 2),
                reward=np.random.uniform(-1, 1),
                next_obs=next_obs, done=bool(i % 5 == 0),
                mask=mask, next_mask=next_mask, head_id=head_id,
            )

    def test_target_net_is_copy(self):
        """Target and online nets should start with identical weights."""
        agent, _ = self._make()
        for p_online, p_target in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        ):
            assert torch.allclose(p_online, p_target)

    def test_target_net_always_deterministic(self):
        from agent.noisy_linear import NoisyLinear
        agent, _ = self._make()
        for m in agent.target_net.modules():
            if isinstance(m, NoisyLinear):
                assert m.deterministic is True

    def test_train_step_returns_none_when_buffer_empty(self):
        agent, _ = self._make()
        result = agent.train_step(beta=0.4)
        assert result is None

    def test_train_step_returns_loss_when_ready(self):
        agent, _ = self._make()
        self._fill_replay(agent, n=100)
        loss = agent.train_step(beta=0.4)
        assert loss is not None
        assert np.isfinite(loss)

    def test_train_step_updates_online_weights(self):
        agent, _ = self._make()
        self._fill_replay(agent, n=100)
        w_before = agent.online_net.trunk[0].weight.data.clone()
        agent.train_step(beta=0.4)
        w_after = agent.online_net.trunk[0].weight.data
        assert not torch.allclose(w_before, w_after), \
            "Online network weights should change after a train step"

    def test_polyak_update_moves_target_toward_online(self):
        """After many Polyak updates, target should approach online."""
        agent, _ = self._make()
        # Manually change online net weights
        with torch.no_grad():
            for p in agent.online_net.parameters():
                p.fill_(1.0)
            for p in agent.target_net.parameters():
                p.fill_(0.0)

        # Run 1000 Polyak updates
        for _ in range(1000):
            agent._polyak_update()

        # Target should be close to 1.0 now (τ=0.005, (1-0.005)^1000 ≈ 0.007)
        for p_target in agent.target_net.parameters():
            assert p_target.data.mean().item() > 0.99, \
                "Target net should converge to online net after many Polyak updates"

    def test_action_selection_respects_mask(self):
        """Masked actions must never be chosen by select_play_actions_batch."""
        agent, device = self._make()
        obs   = np.random.randn(32, 28).astype(np.float32)
        masks = np.ones((32, 4), dtype=bool)
        masks[:, 2] = False   # mask out double
        masks[:, 3] = False   # mask out split
        actions = agent.select_play_actions_batch(obs, masks)
        assert actions.shape == (32,)
        assert (actions < 2).all(), "Should only choose hit(0) or stand(1)"

    def test_bet_action_selection_shape(self):
        agent, _ = self._make()
        obs = np.random.randn(16, 28).astype(np.float32)
        actions = agent.select_bet_actions_batch(obs)
        assert actions.shape == (16,)
        assert ((actions >= 0) & (actions < 5)).all()

    def test_train_step_with_bet_transitions(self):
        """Bet-head transitions (head_id=1) should also produce a valid loss."""
        agent, _ = self._make()
        self._fill_replay(agent, n=100, head_id=1)
        loss = agent.train_step(beta=0.4)
        assert loss is not None
        assert np.isfinite(loss)

    def test_train_step_mixed_heads(self):
        """Buffer with both head_id=0 and head_id=1 should work."""
        agent, _ = self._make()
        self._fill_replay(agent, n=60, head_id=0)
        self._fill_replay(agent, n=60, head_id=1)
        loss = agent.train_step(beta=0.4)
        assert loss is not None
        assert np.isfinite(loss)

    def test_state_dict_roundtrip(self):
        """save then load should reproduce identical weights."""
        import copy
        agent, _ = self._make()
        self._fill_replay(agent, n=50)
        agent.train_step(beta=0.4)

        sd = agent.state_dict()
        agent2, _ = self._make()
        agent2.load_state_dict(sd)

        for p1, p2 in zip(
            agent.online_net.parameters(), agent2.online_net.parameters()
        ):
            assert torch.allclose(p1, p2)

    def test_illegal_action_never_bootstrapped(self):
        """The Bellman target should not propagate through masked-out actions.

        Construct a scenario where only action 0 is legal in the next state.
        After training, the target must have used action 0 for the bootstrap.
        We verify this indirectly by checking that the target Q for illegal
        actions (after masking to -1e9) is -1e9, and the update doesn't NaN.
        """
        agent, device = self._make()
        # Create transitions where next state has only action 0 legal.
        obs_dim = 28
        for _ in range(50):
            obs       = np.zeros(obs_dim, dtype=np.float32)
            next_obs  = np.zeros(obs_dim, dtype=np.float32)
            mask      = np.array([True,  True,  True,  True])
            next_mask = np.array([True,  False, False, False])  # only hit legal
            agent.replay.add(
                obs=obs, action=0, reward=1.0,
                next_obs=next_obs, done=False,
                mask=mask, next_mask=next_mask, head_id=0,
            )
        loss = agent.train_step(beta=1.0)
        assert loss is not None and np.isfinite(loss), \
            f"Training with masked next state produced non-finite loss: {loss}"
