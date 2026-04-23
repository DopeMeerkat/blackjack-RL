"""
DeepCount — Training Script
============================
Runs the full three-stage PPO training pipeline with optional parallel
environment collection via gymnasium SyncVectorEnv.

Stages
------
  1  (0 → stage1_end steps)       flat bet, bet-head frozen → learn basic strategy
  2  (stage1_end → stage2_end)    flat bet, bet-head frozen → learn count-aware play
  3  (stage2_end → total_steps)   learned bet, bet-head live → learn bet sizing

Usage
-----
  python train.py                          # 4 envs, 2 M steps (defaults)
  python train.py --n_envs 8               # 8 parallel envs
  python train.py --n_envs 1               # single env (original behaviour)
  python train.py --device cuda            # GPU
  python train.py --help                   # all options

Step counting
-------------
global_step counts total environment steps across all envs.
rollout_steps is the number of steps collected per env per rollout.
The buffer holds rollout_steps * n_envs transitions total.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from env.blackjack_env import (
    BlackjackShoeEnv,
    PHASE_BETTING,
    PHASE_PLAYING,
    HIT, STAND, DOUBLE, SPLIT,
    MIN_BET, MAX_BET,
)
from models.policy_network import DeepCountNet
from training.buffer import RolloutBuffer
from training.ppo import PPOTrainer
from training.curriculum import CurriculumManager, CurriculumConfig
from utils import obs_to_tensor, build_action_mask


# ─────────────────────────────────────────────────────────────────────────────
# Rollout statistics
# ─────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {HIT: "Hit", STAND: "Stand", DOUBLE: "Dbl", SPLIT: "Spl"}

@dataclass
class RolloutStats:
    """
    Behavioural statistics accumulated over one rollout.

    Action counts cover only PLAYING-phase decisions.
    Bets and ROI figures cover only BETTING-phase steps.
    ROI = total_profit / total_wagered (chips won or lost vs chips risked).
    """
    action_counts:  dict[int, int]  = field(default_factory=lambda: {a: 0 for a in range(4)})
    total_wagered:  float           = 0.0   # sum of bets placed at BETTING steps
    total_profit:   float           = 0.0   # sum of raw rewards across all steps
    ep_chips:       list[float]     = field(default_factory=list)

    # Number of BETTING-phase steps seen (to compute mean_bet correctly
    # even when no episodes finish within this rollout window).
    n_bet_steps: int = 0
    n_play_steps: int = 0

    def record_step(
        self,
        phase:    int,
        action:   int,
        bet:      float,
        reward:   float,
    ):
        self.total_profit += reward
        if phase == PHASE_PLAYING:
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
        elif phase == PHASE_BETTING:
            self.total_wagered += bet
            self.n_bet_steps   += 1

    def mean_bet(self) -> float:
        if self.n_bet_steps == 0:
            return 0.0
        return self.total_wagered / self.n_bet_steps

    def roi(self) -> float:
        """Total profit / total wagered, as a percentage."""
        if self.total_wagered <= 0:
            return 0.0
        return 100.0 * self.total_profit / self.total_wagered

    def action_fracs(self) -> dict[int, float]:
        total = sum(self.action_counts.values())
        if total == 0:
            return {a: 0.0 for a in range(4)}
        return {a: c / total for a, c in self.action_counts.items()}

    def mean_ep_chips(self) -> float:
        return float(np.mean(self.ep_chips)) if self.ep_chips else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_obs_to_tensor(obs: dict, device: torch.device) -> dict:
    """
    Convert a batched obs dict (B, ...) from SyncVectorEnv to tensors.
    Unlike obs_to_tensor, does NOT unsqueeze — the batch dim is already present.
    """
    return {
        "shoe_history":  torch.tensor(obs["shoe_history"],  dtype=torch.long, device=device),
        "hand":          torch.tensor(obs["hand"],          dtype=torch.long, device=device),
        "hand_len":      torch.tensor(obs["hand_len"],      dtype=torch.long, device=device),
        "dealer_upcard": torch.tensor(obs["dealer_upcard"], dtype=torch.long, device=device),
        "phase":         torch.tensor(obs["phase"],         dtype=torch.long, device=device),
    }


def single_obs_from_batch(obs: dict, i: int) -> dict:
    """Extract the i-th env's obs from a batched obs dict."""
    return {k: v[i] for k, v in obs.items()}


def build_action_masks_vec(envs: SyncVectorEnv, mask_complex: bool = False) -> np.ndarray:
    """Build a (n_envs, 4) boolean action mask from each sub-env."""
    return np.stack([build_action_mask(e, mask_complex=mask_complex) for e in envs.envs])


def get_current_bets(envs: SyncVectorEnv, fallback_bets: np.ndarray) -> np.ndarray:
    """Current placed bet per sub-env; fallback when no bet placed yet."""
    return np.array([
        float(e._current_bet) if e._current_bet > 0 else float(fallback_bets[i])
        for i, e in enumerate(envs.envs)
    ], dtype=np.float32)

def get_true_counts(envs: SyncVectorEnv) -> np.ndarray:
    """
    Read the true count directly from each sub-env's visible history.
 
    Do NOT use extract_info_field for this value. Gymnasium's SyncVectorEnv
    stacks info dicts in a way that is unreliable for auto-reset envs:
    the top-level info for a terminated env reflects the NEW episode's first
    step (true_count ≈ 0), not the terminal step. Reading directly from the
    sub-env objects is always correct and immune to this stacking behaviour.
    """
    from env.card_utils import true_count as _true_count
    return np.array([
        _true_count(e._visible_history) for e in envs.envs
    ], dtype=np.float32)


def extract_info_field(
    info: dict,
    key: str,
    terminated: np.ndarray,
    default: float = 0.0,
) -> np.ndarray:
    """
    Extract a scalar info field from SyncVectorEnv output.
    For terminated (auto-reset) envs the final-step info is in
    info["final_info"][i]; for live envs it is in info[key][i].
    """
    n = len(terminated)
    values = np.full(n, default, dtype=np.float32)
    top = info.get(key)
    final_infos = info.get("final_info", [None] * n)
    for i in range(n):
        if terminated[i] and final_infos[i] is not None:
            values[i] = float(final_infos[i].get(key, default))
        elif top is not None:
            values[i] = float(top[i])
    return values


# ─────────────────────────────────────────────────────────────────────────────
# Rollout collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rollout(
    envs:        SyncVectorEnv,
    policy:      DeepCountNet,
    buffer:      RolloutBuffer,
    curriculum:  CurriculumManager,
    global_step: int,
    obs:         dict,
    device:      torch.device,
) -> tuple[dict, RolloutStats]:
    """
    Collect exactly n_steps steps from every env, filling the buffer.

    Returns:
        obs   : first obs of the next rollout
        stats : RolloutStats with action fractions, bet sizes, ROI, chips
    """
    buffer.reset()
    n_envs       = envs.num_envs
    flat_bet     = curriculum.flat_bet(global_step)
    # Stage 1: mask DOUBLE and SPLIT so the value function learns a clean
    # HIT/STAND baseline (all rewards ±1.0) before the ±2.0 DOUBLE signal
    # is introduced in Stage 2.  Without this, V(s) for 2-card hands gets
    # inflated by DOUBLE's ±2.0 outcomes, making A(HIT) = γV(s') − V(s)
    # systematically negative and causing excessive standing.
    mask_complex = curriculum.get_stage(global_step) == 1
    for env in envs.envs:
        env.flat_bet = flat_bet
    stats    = RolloutStats()

    for _ in range(buffer.n_steps):
        obs_t   = batch_obs_to_tensor(obs, device)
        masks   = build_action_masks_vec(envs, mask_complex=mask_complex)
        masks_t = torch.tensor(masks, dtype=torch.bool, device=device)
        phases  = obs["phase"]                               # (n_envs,) int8

        with torch.no_grad():
            actions, log_probs, quantiles, bet_quantiles = \
                policy.get_action_and_logprob(
                    obs_t, action_mask=masks_t, flat_bet=flat_bet
                )
            values_norm = quantiles.mean(dim=-1).cpu().numpy()
            values_raw  = bet_quantiles.mean(dim=-1).cpu().numpy()

        # get_action_and_logprob already returns the flat_bet as the action
        # and its log_prob when flat_bet is provided, so no correction needed.
        bet_arr  = actions["bet"].cpu().numpy()              # (n_envs, 1)
        play_arr = actions["play"].cpu().numpy()             # (n_envs,)
 
        vec_action = {"bet": bet_arr, "play": play_arr}

        # Read true counts BEFORE stepping so they match obs["shoe_history"].
        # After envs.step() the env's _visible_history has grown (new cards
        # drawn), and terminated envs have already been auto-reset to an empty
        # history, making post-step values stale for the aux count head target.
        true_counts  = get_true_counts(envs)

        # Capture bets BEFORE the step so DOUBLE doesn't inflate the scale.
        # After a DOUBLE, e._current_bet is 2×B; normalising by 2B would make
        # a doubled win/loss look identical (±1.0) to a regular win/loss.
        # Using the pre-step bet (B) lets a doubled outcome correctly register
        # as ±2.0 in the normalised stream, so the value function can learn
        # that doubling on a bad hand costs twice as many chips.
        bets_before = np.array([
            float(e._current_bet) if e._current_bet > 0 else float(bet_arr[i, 0])
            for i, e in enumerate(envs.envs)
        ], dtype=np.float32)

        next_obs, rewards, terminated, truncated, info = envs.step(vec_action)
        dones = terminated | truncated

        # For BETTING-phase steps the bet was just placed, so we need the
        # post-step value.  For PLAYING-phase steps we always want bets_before.
        bets_after   = get_current_bets(envs, bet_arr[:, 0])
        current_bets = np.where(phases == PHASE_PLAYING, bets_before, bets_after)

        # ── Record per-env statistics ─────────────────────────────────────
        log_probs_np = log_probs.cpu().numpy()
        for i in range(n_envs):
            stats.record_step(
                phase  = int(phases[i]),
                action = int(play_arr[i]),
                bet    = float(bet_arr[i, 0]),
                reward = float(rewards[i]),
            )
            buffer.add(
                obs         = single_obs_from_batch(obs, i),
                action      = {"bet": bet_arr[i], "play": int(play_arr[i])},
                log_prob    = float(log_probs_np[i]),
                reward      = float(rewards[i]),
                value_norm  = float(values_norm[i]),
                value_raw   = float(values_raw[i]),
                done        = bool(dones[i]),
                true_count  = float(true_counts[i]),
                action_mask = masks[i],
                current_bet = float(current_bets[i]),
            )
        # Collect completed episode chip totals
        for i in range(n_envs):
            if dones[i]:
                ep_chips = extract_info_field(info, "episode_chips", terminated, default=0.0)[i]
                stats.ep_chips.append(ep_chips)
        obs = next_obs

    # Bootstrap last values per env for GAE
    obs_t = batch_obs_to_tensor(obs, device)
    with torch.no_grad():
        last_vn, last_vr = policy.get_values(obs_t)
    buffer.compute_returns_and_advantages(
        last_values_norm = last_vn.cpu().numpy(),
        last_values_raw  = last_vr.cpu().numpy(),
    )

    return obs, stats


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the DeepCount blackjack agent.")

    p.add_argument("--total_steps",   type=int,   default=2_000_000)
    p.add_argument("--rollout_steps", type=int,   default=2048,
                   help="Steps collected per env per rollout. "
                        "Total buffer size = rollout_steps * n_envs.")
    p.add_argument("--n_envs",        type=int,   default=4,
                   help="Number of parallel environments (SyncVectorEnv).")

    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--n_epochs",      type=int,   default=10)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--aux_lr",        type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae_lambda",    type=float, default=0.95)
    p.add_argument("--clip_eps",      type=float, default=0.2)
    p.add_argument("--ent_coef",      type=float, default=0.02)
    p.add_argument("--vf_bet_coef",   type=float, default=0.2)
    p.add_argument("--vf_play_coef",   type=float, default=0.7)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--target_kl",     type=float, default=0.02)

    p.add_argument("--d_shoe",        type=int,   default=64)
    p.add_argument("--d_hand",        type=int,   default=32)
    p.add_argument("--d_fused",       type=int,   default=128)

    p.add_argument("--stage1_end",    type=int,   default=200_000)
    p.add_argument("--stage2_end",    type=int,   default=1_000_000)
    p.add_argument("--aux_anneal",    type=int,   default=500_000)
    p.add_argument("--flat_bet",      type=float, default=10.0)
    p.add_argument("--no_bet_curriculum", action="store_true",
                   help="Disable the flat-bet / frozen-bet-head curriculum. "
                        "The bet head trains freely from step 0 alongside play decisions.")

    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--log_dir",       type=str,   default="runs/deepcount")
    p.add_argument("--save_dir",      type=str,   default="checkpoints")
    p.add_argument("--save_every",    type=int,   default=100_000)
    p.add_argument("--log_every",     type=int,   default=10)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    except Exception:
        print("[INFO] TensorBoard unavailable — logging to stdout only.")

    curriculum = CurriculumManager(CurriculumConfig(
        stage1_end         = args.stage1_end,
        stage2_end         = args.stage2_end,
        aux_anneal_steps   = args.aux_anneal,
        flat_bet           = args.flat_bet,
        use_bet_curriculum = not args.no_bet_curriculum,
    ))

    def make_env(rank: int):
        def _init():
            return BlackjackShoeEnv(seed=args.seed + rank)
        return _init

    envs = SyncVectorEnv([make_env(i) for i in range(args.n_envs)])
    obs, _ = envs.reset()

    policy = DeepCountNet(
        d_shoe  = args.d_shoe,
        d_hand  = args.d_hand,
        d_fused = args.d_fused,
    ).to(device)

    n_params         = sum(p.numel() for p in policy.parameters())
    steps_per_update = args.rollout_steps * args.n_envs
    print(f"\n{'━'*60}")
    print(f"  DeepCount — Training")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Device       : {device}")
    print(f"  Envs         : {args.n_envs}")
    print(f"  Steps/update : {steps_per_update:,}  ({args.rollout_steps} × {args.n_envs})")
    print(f"  Total steps  : {args.total_steps:,}")
    print(f"{'━'*60}\n")

    trainer = PPOTrainer(
        policy        = policy,
        lr            = args.lr,
        aux_lr        = args.aux_lr,
        clip_eps      = args.clip_eps,
        n_epochs      = args.n_epochs,
        vf_bet_coef   = args.vf_bet_coef,
        vf_play_coef  = args.vf_play_coef,
        ent_coef      = args.ent_coef,
        max_grad_norm = args.max_grad_norm,
        target_kl     = args.target_kl,
        device        = device,
    )

    buffer = RolloutBuffer(
        n_steps    = args.rollout_steps,
        n_envs     = args.n_envs,
        gamma      = args.gamma,
        gae_lambda = args.gae_lambda,
        device     = device,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    next_save    = args.save_every
    global_step  = 0
    update_count = 0
    t_start      = time.time()

    try:
        while global_step < args.total_steps:

            policy.eval()
            obs, stats = collect_rollout(
                envs, policy, buffer, curriculum, global_step, obs, device
            )
            global_step  += steps_per_update
            policy.train()

            aux_lambda = curriculum.aux_loss_weight(global_step)
            freeze_bet = curriculum.bet_head_frozen(global_step)
            metrics    = trainer.update(
                buffer,
                batch_size      = args.batch_size,
                aux_lambda      = aux_lambda,
                freeze_bet_head = freeze_bet,
            )
            update_count += 1

            fracs      = stats.action_fracs()
            mean_chips = stats.mean_ep_chips()
            roi        = stats.roi()
            mean_bet   = stats.mean_bet()

            # ── TensorBoard ───────────────────────────────────────────────────
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar("rollout/episode_chips", mean_chips, global_step)
                writer.add_scalar("rollout/roi_pct",       roi,        global_step)
                writer.add_scalar("rollout/mean_bet",      mean_bet,   global_step)
                for a, name in ACTION_NAMES.items():
                    writer.add_scalar(f"actions/pct_{name}", fracs[a] * 100, global_step)
                writer.add_scalar("curriculum/stage",      curriculum.get_stage(global_step), global_step)
                writer.add_scalar("curriculum/aux_lambda", aux_lambda,    global_step)
                writer.add_scalar("curriculum/bet_frozen", int(freeze_bet), global_step)

            # ── Stdout ────────────────────────────────────────────────────────
            if update_count % args.log_every == 0:
                elapsed = time.time() - t_start
                sps     = global_step / max(elapsed, 1e-6)
                action_str = "  ".join(
                    f"{ACTION_NAMES[a]}:{fracs[a]*100:4.1f}%" for a in range(4)
                )
                print(
                    f"step {global_step:>8,} | "
                    f"stage {curriculum.get_stage(global_step)} | "
                    f"chips {mean_chips:>+7.1f} | "
                    f"ROI {roi:>+6.2f}% | "
                    f"bet {mean_bet:>6.1f} | "
                    f"{action_str} | "
                    f"π {metrics['loss_policy']:>+6.4f} | "
                    f"aux {metrics['loss_aux']:>5.4f} | "
                    f"KL {metrics['approx_kl']:>5.4f} | "
                    f"{sps:,.0f} sps"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step >= next_save:
                ckpt = save_dir / f"deepcount_{global_step:08d}.pt"
                torch.save({
                    "step":          global_step,
                    "policy":        policy.state_dict(),
                    "optimizer":     trainer.optimizer.state_dict(),
                    "aux_optimizer": trainer.aux_optimizer.state_dict(),
                    "args":          vars(args),
                }, ckpt)
                print(f"  ✓ checkpoint → {ckpt}")
                next_save += args.save_every

        final = save_dir / "deepcount_final.pt"
        torch.save({
            "step":          global_step,
            "policy":        policy.state_dict(),
            "optimizer":     trainer.optimizer.state_dict(),
            "aux_optimizer": trainer.aux_optimizer.state_dict(),
            "args":          vars(args),
        }, final)
        elapsed = time.time() - t_start
        print(f"\n  ✓ final model → {final}  ({elapsed/60:.1f} min)")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        envs.close()
        if writer is not None:
            writer.close()
        del policy, trainer, buffer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[INFO] GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
