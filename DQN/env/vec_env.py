"""Vectorized wrapper over multiple BlackjackEnv instances.

Runs ``num_envs`` independent environments in a Python loop.  This is
intentionally simple for Milestone 1; NumPy/multiprocessing optimisation
can be added later once correctness is established.

Important: this env does NOT auto-reset on done=True.  The training loop
is responsible for calling reset() on each environment that finishes before
the next step.  See CLAUDE.md for the rationale.
"""

from __future__ import annotations

import numpy as np

from env.blackjack import BlackjackEnv


class VecBlackjackEnv:
    """Vectorised blackjack environment wrapping ``num_envs`` single envs.

    Args:
        num_envs: Number of parallel environments.
        config: Dict loaded from configs/default.yaml (env section).
        seeds: Optional list of per-environment seeds (length num_envs).
    """

    def __init__(
        self,
        num_envs: int,
        config: dict,
        seeds: list[int] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self._envs: list[BlackjackEnv] = [
            BlackjackEnv(config, seed=(seeds[i] if seeds else None))
            for i in range(num_envs)
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Reset all environments.

        Returns:
            obs:   shape (num_envs, 28), float32
            masks: shape (num_envs, 4),  bool
            infos: list of dicts, length num_envs
        """
        results = [env.reset() for env in self._envs]
        obs   = np.stack([r[0] for r in results])
        masks = np.stack([r[1] for r in results])
        infos = [r[2] for r in results]
        return obs, masks, infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments with the given actions.

        Args:
            actions: shape (num_envs,), int — action per environment.

        Returns:
            obs:     shape (num_envs, 28), float32
            masks:   shape (num_envs, 4),  bool
            rewards: shape (num_envs,),    float32
            dones:   shape (num_envs,),    bool
            infos:   list of dicts, length num_envs
        """
        results = [
            env.step(int(a))
            for env, a in zip(self._envs, actions)
        ]
        obs     = np.stack([r[0] for r in results])
        masks   = np.stack([r[1] for r in results])
        rewards = np.array([r[2] for r in results], dtype=np.float32)
        dones   = np.array([r[3] for r in results], dtype=bool)
        infos   = [r[4] for r in results]
        return obs, masks, rewards, dones, infos

    def reset_at(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Reset a single environment and return its (obs, mask, info)."""
        return self._envs[idx].reset()

    @property
    def in_bet_phase(self) -> np.ndarray:
        """Boolean array of shape (num_envs,) — True when env is in bet phase."""
        return np.array([env.in_bet_phase for env in self._envs], dtype=bool)

    def get_rank_counts_batch(
        self, indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Return remaining rank counts for a subset (or all) of environments.

        Args:
            indices: Optional 1-D integer array of env indices.  When None,
                     returns counts for all environments.

        Returns:
            Array of shape (K, 10), dtype int32, where K = len(indices) or
            num_envs when indices is None.
        """
        envs = self._envs if indices is None else [self._envs[i] for i in indices]
        return np.stack([e.get_rank_counts() for e in envs]).astype(np.int32)

    def seed(self, seeds: list[int]) -> None:
        """Re-seed all environments."""
        for env, s in zip(self._envs, seeds):
            env.seed(s)
