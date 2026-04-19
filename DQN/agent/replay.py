"""Prioritized Experience Replay with a SumTree.

Implements the PER algorithm (Schaul et al., 2016) used by the DQN agent.
Priorities are stored in a SumTree for O(log N) add and sample operations.
Importance-sampling weights correct for the non-uniform sampling distribution.

Hyperparameters (from configs/default.yaml):
  alpha = 0.6   — how much prioritization is used (0 = uniform)
  beta         — IS exponent, annealed from 0.4 → 1.0 over training

Each stored transition carries a ``head_id`` field (0 = playing head,
1 = bet-sizing head) so the training step knows which loss to compute.

All transitions are stored in pre-allocated NumPy arrays for efficiency.
A circular write pointer wraps around after ``capacity`` transitions.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------

class SumTree:
    """Binary SumTree for O(log N) priority-proportional sampling.

    The tree has ``capacity`` leaves.  Internal nodes store the sum of
    their children's priorities.  The root (index 0) stores the total sum.
    Leaf ``i`` is stored at tree index ``i + capacity - 1``.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, leaf_idx: int, change: float) -> None:
        """Bubble a priority change up from a leaf to the root."""
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2        # parent
            self.tree[idx] += change

    def _retrieve(self, idx: int, s: float) -> int:
        """Descend the tree to find the leaf whose cumulative sum contains s."""
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx              # reached a leaf
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        """Sum of all priorities (stored at the root)."""
        return float(self.tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        """Set the priority at a leaf (tree index = leaf_idx + capacity - 1)."""
        tree_idx = leaf_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float]:
        """Return (leaf_idx, priority) for the cumulative-sum target s."""
        tree_idx = self._retrieve(0, s)
        leaf_idx = tree_idx - self.capacity + 1
        return leaf_idx, float(self.tree[tree_idx])


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Circular replay buffer with proportional prioritization.

    Args:
        capacity: Maximum number of transitions.
        obs_dim:  Observation vector length (28).
        alpha:    Prioritization exponent (0 = uniform, 1 = full priority).
        eps:      Small constant added to |TD error| before raising to alpha,
                  ensuring every transition has non-zero probability.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.alpha    = alpha
        self.eps      = eps

        self._sumtree = SumTree(capacity)

        # Pre-allocated transition arrays
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions   = np.zeros(capacity,            dtype=np.int32)
        self.rewards   = np.zeros(capacity,            dtype=np.float32)
        self.next_obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones     = np.zeros(capacity,            dtype=bool)
        self.masks     = np.zeros((capacity, 4),       dtype=bool)
        self.next_masks= np.zeros((capacity, 4),       dtype=bool)
        self.head_ids  = np.zeros(capacity,            dtype=np.int8)

        self._write        = 0           # next write position
        self._n_entries    = 0
        self._max_priority = 1.0         # tracks running max for new entries

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
        head_id: int,
    ) -> None:
        """Add one transition with maximum current priority."""
        i = self._write
        self.obs[i]        = obs
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.next_obs[i]   = next_obs
        self.dones[i]      = done
        self.masks[i]      = mask
        self.next_masks[i] = next_mask
        self.head_ids[i]   = head_id

        # New transitions get max priority so they're sampled at least once.
        priority = self._max_priority ** self.alpha
        self._sumtree.update(i, priority)

        self._write    = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Sample a batch with stratified sampling over priority segments.

        Args:
            batch_size: Number of transitions to sample.
            beta:       IS correction exponent (higher → stronger correction).

        Returns:
            batch:      Dict of numpy arrays (obs, actions, rewards, …).
            weights:    IS weights, shape (batch_size,), max-normalised.
            leaf_indices: Leaf indices in the SumTree for priority updates.
        """
        assert self._n_entries >= batch_size, (
            f"Buffer has only {self._n_entries} entries, need {batch_size}"
        )

        leaf_indices = np.empty(batch_size, dtype=np.int64)
        priorities   = np.empty(batch_size, dtype=np.float64)

        # Stratified sampling: divide [0, total] into batch_size equal segments.
        total    = self._sumtree.total
        segment  = total / batch_size

        for k in range(batch_size):
            s = np.random.uniform(segment * k, segment * (k + 1))
            s = min(s, total - 1e-12)         # numerical safety at boundary
            leaf_idx, p = self._sumtree.get(s)
            # Guard against empty leaves (priority = 0); fall back to uniform.
            if p <= 0.0 or leaf_idx >= self._n_entries:
                leaf_idx = np.random.randint(0, self._n_entries)
                p = max(float(self._sumtree.tree[leaf_idx + self._sumtree.capacity - 1]),
                        self.eps ** self.alpha)
            leaf_indices[k] = leaf_idx
            priorities[k]   = p

        # IS weights: w_i ∝ (1 / (N * P(i)))^beta, normalised by max.
        sampling_probs = priorities / total
        weights = (self._n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()

        batch = {
            "obs":        self.obs[leaf_indices],
            "actions":    self.actions[leaf_indices],
            "rewards":    self.rewards[leaf_indices],
            "next_obs":   self.next_obs[leaf_indices],
            "dones":      self.dones[leaf_indices],
            "masks":      self.masks[leaf_indices],
            "next_masks": self.next_masks[leaf_indices],
            "head_ids":   self.head_ids[leaf_indices],
        }
        return batch, weights.astype(np.float32), leaf_indices

    # ------------------------------------------------------------------
    # Priority update (called after each training step)
    # ------------------------------------------------------------------

    def update_priorities(
        self,
        leaf_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities for a sampled batch after computing TD errors."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(leaf_indices, priorities):
            self._sumtree.update(int(idx), float(p))
        if len(priorities) > 0:
            self._max_priority = max(self._max_priority, float(priorities.max()))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_entries

    @property
    def is_ready(self) -> bool:
        """True once the buffer has enough entries to sample a full batch."""
        return self._n_entries > 0
