"""Joint training script for playing policy + bet-sizing head (Milestone 4).

Trains both heads simultaneously with a shared trunk.  Bet-sizing transitions
are stored in the same replay buffer as playing transitions, distinguished by
head_id (0 = play, 1 = bet).  See blackjack_rl_design.md §7, §10, §15.4.

Bet-head transitions are one-step bandits: the bet observation is recorded at
hand start, and the terminal hand reward (summed across sub-hands, scaled by
the chosen multiplier) is the target.  No bootstrapping into a next state.

Usage:
  cd blackjack-RL/DQN
  python train/train_joint.py                             # defaults from config
  python train/train_joint.py --total-hands 10000000      # quick test
  python train/train_joint.py --exp-name joint_run2

Outputs:
  outputs/checkpoints/<exp_name>/          checkpoints every 1M hands + final
  outputs/runs/<exp_name>/                 TensorBoard logs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.dqn import DQNAgent
from env.vec_env import VecBlackjackEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config loading (shared with train_play.py)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    return cfg


def net_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in (
        "obs_dim", "playing_actions", "bet_actions",
        "trunk_hidden", "trunk_layers",
        "head_hidden", "noisy_sigma0", "dueling",
        "n_atoms", "v_min", "v_max",
    ) if k in cfg}


def train_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in (
        "gamma", "target_tau", "batch_size", "learning_rate",
        "replay_alpha", "grad_clip", "n_step",
    ) if k in cfg}


# ---------------------------------------------------------------------------
# Beta annealing
# ---------------------------------------------------------------------------

def beta_schedule(step: int, total_steps: int,
                  beta_start: float, beta_end: float) -> float:
    frac = min(step / max(total_steps, 1), 1.0)
    return beta_start + frac * (beta_end - beta_start)


# ---------------------------------------------------------------------------
# Quick evaluation (deterministic, single-env)
# ---------------------------------------------------------------------------

_ACTION_NAMES = ["hit", "stand", "double", "split"]
_BET_MULTIPLIERS = [1, 2, 4, 8, 12]


def quick_eval(
    agent: DQNAgent,
    cfg: dict,
    n_hands: int,
    device: torch.device,
) -> dict:
    """Run n_hands hands with the deterministic joint policy.

    Returns dict with ev, action_counts, bet_distribution, tc_bet_pairs.
    """
    from env.blackjack import BlackjackEnv

    agent.online_net.set_deterministic(True)
    env = BlackjackEnv(cfg, seed=12345)

    total_reward = 0.0
    hands = 0
    action_counts = np.zeros(4, dtype=np.int64)
    bet_counts = np.zeros(5, dtype=np.int64)
    tc_bet_pairs = []  # (true_count, bet_multiplier) for correlation

    obs, mask, info = env.reset()

    while hands < n_hands:
        if info["phase"] == "bet":
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, bet_q = agent.online_net(obs_t)
            bet_idx = int(bet_q.argmax(dim=1).item())
            action = bet_idx
            bet_counts[bet_idx] += 1
            tc_bet_pairs.append((info["true_count"], _BET_MULTIPLIERS[bet_idx]))
        else:
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask[None], dtype=torch.bool, device=device)
            with torch.no_grad():
                play_q, _ = agent.online_net(obs_t)
                play_q = play_q.clone()
                play_q[~mask_t] = -1e9
            action = int(play_q.argmax(dim=1).item())
            action_counts[action] += 1

        obs, mask, reward, done, info = env.step(action)
        if done:
            total_reward += reward
            hands += 1
            if hands < n_hands:
                obs, mask, info = env.reset()

    agent.online_net.set_deterministic(False)

    ev = total_reward / n_hands

    # Compute TC-bet correlation
    tc_corr = 0.0
    if len(tc_bet_pairs) > 1:
        tc_arr = np.array([p[0] for p in tc_bet_pairs])
        bet_arr = np.array([p[1] for p in tc_bet_pairs], dtype=np.float64)
        if tc_arr.std() > 0 and bet_arr.std() > 0:
            tc_corr = float(np.corrcoef(tc_arr, bet_arr)[0, 1])

    return {
        "ev": ev,
        "action_counts": action_counts,
        "bet_counts": bet_counts,
        "tc_bet_corr": tc_corr,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> DQNAgent:
    cfg = load_config(args.config)

    total_hands = args.total_hands or cfg.get("total_hands", 200_000_000)
    exp_name = args.exp_name or "joint"

    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )
    print(f"Device: {device}")
    print(f"Experiment: {exp_name}")
    print(f"Total hands: {total_hands:,}")

    # Directories
    ckpt_dir = Path("outputs/checkpoints") / exp_name
    tb_dir = Path("outputs/runs") / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if _TB_AVAILABLE:
        writer = SummaryWriter(str(tb_dir))

    # Agent
    ncfg = net_config(cfg)
    tcfg = train_config(cfg)
    agent = DQNAgent(
        net_config=ncfg,
        train_config=tcfg,
        replay_capacity=cfg.get("replay_buffer_size", 1_000_000),
        device=device,
    )

    # Optionally load a playing-policy checkpoint to warm-start the trunk
    if args.play_checkpoint:
        print(f"Loading playing-policy checkpoint: {args.play_checkpoint}")
        ckpt = torch.load(args.play_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        print("  Loaded. Trunk and playing head warm-started.")

    # Vectorised env
    num_envs = cfg.get("num_envs", 64)
    bet_multipliers = cfg.get("bet_multipliers", [1, 2, 4, 8, 12])
    vec_env = VecBlackjackEnv(
        num_envs=num_envs,
        config=cfg,
        seeds=list(range(num_envs)),
    )

    # Training hyperparameters
    train_every = cfg.get("train_every_n_steps", 4)
    beta_start = cfg.get("replay_beta_start", 0.4)
    eval_every = cfg.get("eval_every_n_hands", 1_000_000)
    eval_hands = min(cfg.get("eval_hands", 1_000_000), 100_000)
    ckpt_every = cfg.get("eval_every_n_hands", 1_000_000)
    warmup_transitions = cfg.get("warmup_transitions", 50_000)

    # Per-env tracking for bet-head transitions
    # We record the bet observation and action at hand start, then use the
    # terminal reward as the bet-head target when the hand finishes.
    bet_obs = np.zeros((num_envs, cfg.get("obs_dim", 28)), dtype=np.float32)
    bet_actions = np.zeros(num_envs, dtype=np.int32)
    hand_reward_accum = np.zeros(num_envs, dtype=np.float32)

    # State
    obs_arr, masks_arr, infos = vec_env.reset()
    global_step = 0
    hands_played = 0
    last_eval = 0
    last_ckpt = 0
    t_start = time.time()

    print(f"\nStarting joint training...  (batch_size={agent.batch_size}, "
          f"buffer={agent.replay.capacity:,})")

    while hands_played < total_hands:
        in_bet = vec_env.in_bet_phase  # (N,) bool

        # --- Action selection ---
        actions = np.zeros(num_envs, dtype=np.int32)

        # Bet phase envs: use the bet head
        bet_envs = np.where(in_bet)[0]
        if len(bet_envs) > 0:
            batch_obs = obs_arr[bet_envs]
            actions[bet_envs] = agent.select_bet_actions_batch(batch_obs)

            # Record bet observations for later replay storage
            for i in bet_envs:
                bet_obs[i] = obs_arr[i].copy()
                bet_actions[i] = actions[i]
                hand_reward_accum[i] = 0.0

        # Play phase envs: use the playing head
        play_envs = np.where(~in_bet)[0]
        if len(play_envs) > 0:
            batch_obs = obs_arr[play_envs]
            batch_masks = masks_arr[play_envs]
            actions[play_envs] = agent.select_play_actions_batch(
                batch_obs, batch_masks
            )

        # --- Step all envs ---
        next_obs, next_masks, rewards, dones, infos = vec_env.step(actions)

        # --- Store playing transitions (via n-step accumulator) ---
        for i in play_envs:
            agent.add_play_transition(
                env_id=int(i),
                obs=obs_arr[i].copy(),
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_obs=next_obs[i].copy(),
                done=bool(dones[i]),
                mask=masks_arr[i].copy(),
                next_mask=next_masks[i].copy(),
            )

        # --- Handle done envs ---
        for i in range(num_envs):
            if dones[i]:
                hands_played += 1
                hand_reward_accum[i] += rewards[i]

                # Store the bet-head transition (one-step bandit, always done)
                # The reward is the total hand outcome (already scaled by
                # the bet multiplier inside the environment).
                dummy_mask = np.array([True, True, True, True], dtype=bool)
                agent.add_bet_transition(
                    obs=bet_obs[i],
                    action=int(bet_actions[i]),
                    reward=float(hand_reward_accum[i]),
                    next_obs=np.zeros(agent.obs_dim, dtype=np.float32),
                    mask=dummy_mask,
                    next_mask=dummy_mask,
                )

                reset_obs, reset_mask, _ = vec_env.reset_at(i)
                next_obs[i] = reset_obs
                next_masks[i] = reset_mask
            else:
                # Accumulate intermediate rewards (normally 0 until terminal)
                hand_reward_accum[i] += rewards[i]

        obs_arr = next_obs
        masks_arr = next_masks
        global_step += 1

        # --- Gradient update ---
        if global_step % train_every == 0 and len(agent.replay) >= warmup_transitions:
            beta = beta_schedule(hands_played, total_hands, beta_start, 1.0)
            loss = agent.train_step(beta)

            if writer and loss is not None and global_step % 1000 == 0:
                writer.add_scalar("train/loss", loss, hands_played)
                writer.add_scalar("train/beta", beta, hands_played)
                writer.add_scalar("train/replay_size", len(agent.replay), hands_played)

        # --- Periodic eval ---
        if hands_played - last_eval >= eval_every:
            stats = quick_eval(agent, cfg, eval_hands, device)
            elapsed = time.time() - t_start
            rate = hands_played / elapsed

            total_actions = stats["action_counts"].sum()
            action_pcts = stats["action_counts"] / max(total_actions, 1)
            action_str = "  ".join(
                f"{name}={action_pcts[i]*100:.1f}%"
                for i, name in enumerate(_ACTION_NAMES)
            )

            total_bets = stats["bet_counts"].sum()
            bet_pcts = stats["bet_counts"] / max(total_bets, 1)
            bet_str = "  ".join(
                f"{m}x={bet_pcts[i]*100:.1f}%"
                for i, m in enumerate(_BET_MULTIPLIERS)
            )

            print(
                f"  hands={hands_played:>10,}  "
                f"EV={stats['ev']*100:+.3f}%  "
                f"TC-bet corr={stats['tc_bet_corr']:.3f}  "
                f"rate={rate/1000:.1f}k/s\n"
                f"    actions: {action_str}\n"
                f"    bets:    {bet_str}"
            )

            if writer:
                writer.add_scalar("eval/ev", stats["ev"], hands_played)
                writer.add_scalar("eval/tc_bet_corr", stats["tc_bet_corr"], hands_played)
                for i, name in enumerate(_ACTION_NAMES):
                    writer.add_scalar(f"eval/action_pct_{name}", action_pcts[i], hands_played)
                for i, m in enumerate(_BET_MULTIPLIERS):
                    writer.add_scalar(f"eval/bet_pct_{m}x", bet_pcts[i], hands_played)

            last_eval = hands_played

        # --- Periodic checkpoint ---
        if hands_played - last_ckpt >= ckpt_every:
            ckpt_path = ckpt_dir / f"step_{hands_played:010d}.pt"
            torch.save({
                "agent": agent.state_dict(),
                "hands_played": hands_played,
                "config": cfg,
            }, ckpt_path)
            last_ckpt = hands_played

    # --- Final checkpoint ---
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "agent": agent.state_dict(),
        "hands_played": hands_played,
        "config": cfg,
    }, final_path)
    print(f"\nTraining complete.  Final checkpoint: {final_path}")

    if writer:
        writer.close()

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint training: playing + bet-sizing")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--total-hands", type=int, default=None,
                   help="Override total training hands (default: from config).")
    p.add_argument("--exp-name", default=None,
                   help="Experiment name for checkpoints/logs.")
    p.add_argument("--play-checkpoint", default=None,
                   help="Optional: warm-start from a playing-policy checkpoint.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
