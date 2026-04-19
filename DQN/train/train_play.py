"""Training script for the playing-policy DQN (Milestones 2 and 3).

Milestone 2: count feature zeroed (--zero-count), flat 1x bet, 10M hands.
Milestone 3: count feature enabled, flat 1x bet, 50M hands.

Usage:
  cd blackjack-RL/DQN
  python train/train_play.py                           # Milestone 2 defaults
  python train/train_play.py --use-count               # Milestone 3
  python train/train_play.py --total-hands 5000000     # quick test

Outputs:
  outputs/checkpoints/<exp_name>/          checkpoints every 1M hands + final
  outputs/runs/<exp_name>/                 TensorBoard logs
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow running from the DQN/ root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.dqn import DQNAgent
from env.vec_env import VecBlackjackEnv

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Merge all sections into a flat dict for the agent.
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    return cfg


def net_config(cfg: dict) -> dict:
    """Keys expected by BlackjackNet."""
    return {k: cfg[k] for k in (
        "obs_dim", "playing_actions", "bet_actions",
        "trunk_hidden", "trunk_layers",
        "head_hidden", "noisy_sigma0", "dueling",
    ) if k in cfg}


def train_config(cfg: dict) -> dict:
    """Keys expected by DQNAgent."""
    return {k: cfg[k] for k in (
        "gamma", "target_tau", "batch_size", "learning_rate",
        "replay_alpha", "grad_clip", "n_step",
    ) if k in cfg}


# ---------------------------------------------------------------------------
# Beta annealing schedule
# ---------------------------------------------------------------------------

def beta_schedule(step: int, total_steps: int,
                  beta_start: float, beta_end: float) -> float:
    """Linear annealing of the IS beta from beta_start to beta_end."""
    frac = min(step / max(total_steps, 1), 1.0)
    return beta_start + frac * (beta_end - beta_start)


# ---------------------------------------------------------------------------
# Evaluation helper (called periodically during training)
# ---------------------------------------------------------------------------

_ACTION_NAMES = ["hit", "stand", "double", "split"]


def quick_eval(
    agent: DQNAgent,
    cfg: dict,
    n_hands: int,
    zero_count: bool,
    device: torch.device,
    num_envs: int = 64,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Run n_hands hands with the deterministic policy using VecBlackjackEnv.

    Returns:
        ev:           Mean reward per hand.
        action_counts: (4,) int array — counts of hit/stand/double/split
                       decisions taken across all play-phase steps.
    """
    agent.online_net.set_deterministic(True)
    vec_env = VecBlackjackEnv(
        num_envs, cfg, seeds=[seed + i for i in range(num_envs)]
    )
    obs, masks, infos = vec_env.reset()

    rewards = []
    action_counts = np.zeros(4, dtype=np.int64)

    while len(rewards) < n_hands:
        actions = np.zeros(num_envs, dtype=np.int32)

        play_idx = np.array(
            [i for i in range(num_envs) if infos[i]["phase"] != "bet"]
        )
        if len(play_idx) > 0:
            batch_obs = obs[play_idx].copy()
            if zero_count:
                batch_obs[:, 25] = 0.0
            obs_t  = torch.tensor(batch_obs,      dtype=torch.float32, device=device)
            mask_t = torch.tensor(masks[play_idx], dtype=torch.bool,    device=device)
            with torch.no_grad():
                play_q, _ = agent.online_net(obs_t)
                play_q = play_q.clone()
                play_q[~mask_t] = -1e9
            play_actions = play_q.argmax(dim=1).cpu().numpy().astype(np.int32)
            actions[play_idx] = play_actions
            for a in play_actions:
                action_counts[a] += 1

        obs, masks, rews, dones, infos = vec_env.step(actions)

        for i in range(num_envs):
            if dones[i]:
                rewards.append(float(rews[i]))
                if len(rewards) >= n_hands:
                    break
                obs_i, mask_i, info_i = vec_env.reset_at(i)
                obs[i] = obs_i
                masks[i] = mask_i
                infos[i] = info_i

    agent.online_net.set_deterministic(False)
    arr = np.array(rewards[:n_hands], dtype=np.float64)
    return float(arr.mean()), action_counts


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> DQNAgent:
    cfg = load_config(args.config)

    # Command-line overrides
    zero_count  = not args.use_count
    total_hands = args.total_hands or (
        cfg["total_hands_no_count"] if zero_count else cfg["total_hands_with_count"]
    )
    exp_name    = args.exp_name or (
        "play_no_count" if zero_count else "play_with_count"
    )

    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")
    print(f"Experiment: {exp_name}")
    print(f"Total hands: {total_hands:,}")
    print(f"Zero count:  {zero_count}")

    # Directories
    ckpt_dir = Path("outputs/checkpoints") / exp_name
    tb_dir   = Path("outputs/runs")        / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if _TB_AVAILABLE:
        writer = SummaryWriter(str(tb_dir))

    # Agent
    ncfg  = net_config(cfg)
    tcfg  = train_config(cfg)
    agent = DQNAgent(
        net_config=ncfg,
        train_config=tcfg,
        replay_capacity=cfg.get("replay_buffer_size", 1_000_000),
        device=device,
    )

    # Vectorised env
    num_envs = cfg.get("num_envs", 64)
    vec_env  = VecBlackjackEnv(
        num_envs=num_envs,
        config=cfg,
        seeds=list(range(num_envs)),
    )

    # Training hyperparameters
    train_every  = cfg.get("train_every_n_steps", 4)   # gradient updates per N global steps
    beta_start   = cfg.get("replay_beta_start", 0.4)
    eval_every   = cfg.get("eval_every_n_hands", 1_000_000)
    eval_hands   = cfg.get("eval_hands", 1_000_000)
    ckpt_every   = cfg.get("eval_every_n_hands", 1_000_000)
    # Don't start gradient updates until the buffer has enough diverse experience.
    # Starting too early (e.g. 512 transitions) biases early Q-values toward
    # whatever NoisyNet happened to explore in the first handful of hands.
    warmup_transitions = cfg.get("warmup_transitions", 50_000)

    # State
    obs_arr, masks_arr, infos = vec_env.reset()
    global_step  = 0
    hands_played = 0
    last_eval    = 0
    last_ckpt    = 0
    t_start      = time.time()

    print(f"\nStarting training…  (batch_size={agent.batch_size}, "
          f"buffer={agent.replay.capacity:,})")

    while hands_played < total_hands:
        in_bet = vec_env.in_bet_phase  # (N,) bool

        # --- Action selection ---
        actions = np.zeros(num_envs, dtype=np.int32)

        # Bet phase envs: flat 1x bet in M2/M3
        actions[in_bet] = 0

        # Play phase envs: batched network inference
        play_envs = np.where(~in_bet)[0]
        if len(play_envs) > 0:
            batch_obs = obs_arr[play_envs].copy()
            if zero_count:
                batch_obs[:, 25] = 0.0
            batch_masks = masks_arr[play_envs]

            actions[play_envs] = agent.select_play_actions_batch(
                batch_obs, batch_masks
            )

        # --- Step all envs ---
        next_obs, next_masks, rewards, dones, infos = vec_env.step(actions)

        # --- Store playing transitions (via n-step accumulator) ---
        for i in play_envs:
            ob      = obs_arr[i].copy()
            next_ob = next_obs[i].copy()
            if zero_count:
                ob[25]      = 0.0
                next_ob[25] = 0.0

            agent.add_play_transition(
                env_id=int(i),
                obs=ob,
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_obs=next_ob,
                done=bool(dones[i]),
                mask=masks_arr[i].copy(),
                next_mask=next_masks[i].copy(),
            )

        # --- Handle done envs (both BJ-immediate and normal) ---
        for i in range(num_envs):
            if dones[i]:
                hands_played += 1
                reset_obs, reset_mask, _ = vec_env.reset_at(i)
                next_obs[i]  = reset_obs
                next_masks[i]= reset_mask

        obs_arr   = next_obs
        masks_arr = next_masks
        global_step += 1

        # --- Gradient update ---
        if global_step % train_every == 0 and len(agent.replay) >= warmup_transitions:
            # Beta annealed over total training hands
            beta = beta_schedule(hands_played, total_hands, beta_start, 1.0)
            loss = agent.train_step(beta)

            if writer and loss is not None and global_step % 1000 == 0:
                writer.add_scalar("train/loss", loss, hands_played)
                writer.add_scalar("train/beta", beta, hands_played)
                writer.add_scalar("train/replay_size", len(agent.replay), hands_played)

        # --- Periodic eval ---
        if hands_played - last_eval >= eval_every:
            ev, action_counts = quick_eval(agent, cfg, eval_hands, zero_count, device)
            elapsed = time.time() - t_start
            rate = hands_played / elapsed
            total_actions = action_counts.sum()
            action_pcts = action_counts / max(total_actions, 1)
            action_str = "  ".join(
                f"{name}={action_pcts[i]*100:.1f}%"
                for i, name in enumerate(_ACTION_NAMES)
            )
            print(
                f"  hands={hands_played:>10,}  "
                f"EV={ev*100:+.3f}%  "
                f"replay={len(agent.replay):,}  "
                f"rate={rate/1000:.1f}k/s\n"
                f"    actions: {action_str}"
            )
            if writer:
                writer.add_scalar("eval/ev", ev, hands_played)
                for i, name in enumerate(_ACTION_NAMES):
                    writer.add_scalar(f"eval/action_pct_{name}", action_pcts[i], hands_played)
            last_eval = hands_played

        # --- Periodic checkpoint ---
        if hands_played - last_ckpt >= ckpt_every:
            ckpt_path = ckpt_dir / f"step_{hands_played:010d}.pt"
            torch.save({
                "agent":       agent.state_dict(),
                "hands_played": hands_played,
                "config":      cfg,
            }, ckpt_path)
            last_ckpt = hands_played

    # --- Final checkpoint ---
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "agent":        agent.state_dict(),
        "hands_played": hands_played,
        "config":       cfg,
    }, final_path)
    print(f"\nTraining complete.  Final checkpoint: {final_path}")

    if writer:
        writer.close()

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the blackjack playing policy")
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--use-count",    action="store_true",
                   help="Enable count feature in obs[25] (Milestone 3). "
                        "Default: count zeroed (Milestone 2).")
    p.add_argument("--total-hands",  type=int, default=None,
                   help="Override total training hands.")
    p.add_argument("--exp-name",     default=None,
                   help="Experiment name for checkpoints/logs.")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
