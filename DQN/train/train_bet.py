"""Stage B: train bet-sizing agent against a frozen play oracle.

Per-hand loop (vectorised over num_envs):
  1. Bet phase  — BetAgent observes shoe/count features, selects multiplier.
  2. Play phase — frozen play oracle (DQNAgent, deterministic) drives play.
  3. done=True  — scaled hand reward → BetAgent replay → gradient step.

The play oracle is never updated.  Its NoisyNets are set to deterministic
mode so action selection is greedy and reproducible.

Usage:
  python train/train_bet.py --play-oracle outputs/checkpoints/play_oracle/final.pt
  python train/train_bet.py --play-oracle ... --exp-name my_bet --total-hands 5000000

Outputs:
  outputs/checkpoints/<exp_name>/     bet-agent checkpoints every eval interval
  outputs/runs/<exp_name>/            TensorBoard logs
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.bet_agent import BetAgent
from agent.dqn import DQNAgent
from env.bet_encoding import BET_OBS_DIM, encode_bet_obs
from env.vec_env import VecBlackjackEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    return cfg


def _net_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in (
        "obs_dim", "playing_actions",
        "trunk_hidden", "trunk_layers",
        "head_hidden", "noisy_sigma0",
        "dueling", "n_atoms", "v_min", "v_max",
    ) if k in cfg}


def _train_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in (
        "gamma", "target_tau", "batch_size", "learning_rate",
        "replay_alpha", "grad_clip", "n_step",
    ) if k in cfg}


# ---------------------------------------------------------------------------
# Oracle loader
# ---------------------------------------------------------------------------

def load_play_oracle(
    checkpoint_path: str,
    fallback_cfg: dict,
    device: torch.device,
) -> DQNAgent:
    """Load a DQNAgent from checkpoint, freeze it, and set deterministic mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", fallback_cfg)
    agent = DQNAgent(
        net_config=_net_config(cfg),
        train_config=_train_config(cfg),
        replay_capacity=1_000,   # not used; oracle is never trained
        device=device,
    )
    agent.load_state_dict(ckpt["agent"])
    agent.online_net.set_deterministic(True)
    for p in agent.online_net.parameters():
        p.requires_grad_(False)
    return agent


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )

    cfg       = load_config(args.config)
    exp_name  = args.exp_name
    total_hands = args.total_hands or cfg.get("total_hands", 10_000_000)
    num_envs  = cfg.get("num_envs", 64)

    print(f"Device:       {device}")
    print(f"Experiment:   {exp_name}")
    print(f"Total hands:  {total_hands:,}")
    print(f"Num envs:     {num_envs}")

    ckpt_dir = Path("outputs/checkpoints") / exp_name
    tb_dir   = Path("outputs/runs")        / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if _TB_AVAILABLE:
        writer = SummaryWriter(str(tb_dir))

    # --- Agents ---
    print(f"Loading play oracle: {args.play_oracle}")
    play_agent = load_play_oracle(args.play_oracle, cfg, device)
    print("  Play oracle loaded and frozen.")

    if args.checkpoint:
        ckpt_path = f"outputs/checkpoints/{exp_name}/{args.checkpoint}"
        print(f"Resuming bet agent from: {ckpt_path}")
        resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        bet_agent = BetAgent(resume_ckpt.get("config", cfg), device)
        bet_agent.load_state_dict(resume_ckpt["bet_agent"])
        hands_played = resume_ckpt.get("hands_played", 0)
        print(f"  Resumed at hands_played={hands_played:,}")
    else:
        bet_agent    = BetAgent(cfg, device)
        hands_played = 0

    # --- Environment ---
    vec_env = VecBlackjackEnv(
        num_envs=num_envs,
        config=cfg,
        seeds=list(range(num_envs)),
    )
    obs_arr, masks_arr, infos = vec_env.reset()

    # --- Per-env pending-bet state ---
    bet_obs_dim = int(cfg.get("bet_obs_dim", BET_OBS_DIM))
    pending_bet_obs    = np.zeros((num_envs, bet_obs_dim), dtype=np.float32)
    pending_bet_action = np.zeros(num_envs, dtype=np.int32)
    has_pending_bet    = np.zeros(num_envs, dtype=bool)

    eval_every = cfg.get("eval_every_n_hands", 1_000_000)
    ckpt_every = cfg.get("eval_every_n_hands", 1_000_000)
    last_eval  = hands_played
    last_ckpt  = hands_played
    t_start    = time.time()

    # Rolling stats windows
    recent_rewards: list[float] = []
    recent_losses:  list[float] = []
    _WINDOW = 100_000

    _interrupted = [False]

    def _handle_sigint(signum, frame):
        _interrupted[0] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[Interrupted] Finishing step, saving…")

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"\nStarting bet-agent training…  "
          f"(batch_size={bet_agent.batch_size}, "
          f"buf_capacity={bet_agent.replay.capacity:,})")

    while hands_played < total_hands:
        in_bet = vec_env.in_bet_phase   # (num_envs,) bool
        actions = np.zeros(num_envs, dtype=np.int32)

        # --- Bet phase: record obs and pick multiplier ---
        bet_envs = np.where(in_bet)[0]
        if len(bet_envs) > 0:
            rank_counts = vec_env.get_rank_counts_batch(bet_envs)   # (K, 10)
            bet_obs_batch = np.stack([
                encode_bet_obs(
                    rank_counts[j],
                    float(infos[i]["true_count"]),
                    float(infos[i]["decks_remaining"]),
                )
                for j, i in enumerate(bet_envs)
            ])
            bet_acts = bet_agent.select_actions_batch(bet_obs_batch, greedy=False)
            for j, i in enumerate(bet_envs):
                pending_bet_obs[i]    = bet_obs_batch[j]
                pending_bet_action[i] = int(bet_acts[j])
                has_pending_bet[i]    = True
                actions[i]            = int(bet_acts[j])

        # --- Play phase: oracle selects deterministically ---
        play_envs = np.where(~in_bet)[0]
        if len(play_envs) > 0:
            actions[play_envs] = play_agent.select_play_actions_batch(
                obs_arr[play_envs], masks_arr[play_envs]
            )

        # --- Step ---
        next_obs, next_masks, rewards, dones, infos = vec_env.step(actions)

        # --- Handle terminal hands ---
        for i in range(num_envs):
            if dones[i]:
                hands_played += 1
                hand_rew = float(rewards[i])
                recent_rewards.append(hand_rew)
                if len(recent_rewards) > _WINDOW:
                    recent_rewards.pop(0)

                if has_pending_bet[i]:
                    bet_agent.add_transition(
                        pending_bet_obs[i],
                        int(pending_bet_action[i]),
                        hand_rew,
                    )
                    has_pending_bet[i] = False
                    loss = bet_agent.train_step()
                    if loss is not None:
                        recent_losses.append(loss)
                        if len(recent_losses) > _WINDOW:
                            recent_losses.pop(0)

                reset_obs, reset_mask, reset_info = vec_env.reset_at(i)
                next_obs[i]   = reset_obs
                next_masks[i] = reset_mask
                infos[i]      = reset_info

        obs_arr   = next_obs
        masks_arr = next_masks

        # --- Periodic logging ---
        if hands_played - last_eval >= eval_every and hands_played > last_eval:
            elapsed  = time.time() - t_start
            rate     = hands_played / elapsed
            mean_ev  = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
            mean_loss = float(np.mean(recent_losses[-10_000:])) if recent_losses else float("nan")
            print(
                f"  hands={hands_played:>10,}  "
                f"bet_EV={mean_ev*100:+.3f}%  "
                f"bet_loss={mean_loss:.5f}  "
                f"buf={len(bet_agent.replay):,}  "
                f"rate={rate/1000:.1f}k/s"
            )
            if writer:
                writer.add_scalar("train/bet_ev",       mean_ev,   hands_played)
                writer.add_scalar("train/bet_loss",     mean_loss, hands_played)
                writer.add_scalar("train/replay_size",  len(bet_agent.replay), hands_played)
            last_eval = hands_played

        # --- Periodic checkpoint ---
        if hands_played - last_ckpt >= ckpt_every and hands_played > last_ckpt:
            step_path = ckpt_dir / f"step_{hands_played:010d}.pt"
            torch.save({
                "bet_agent":    bet_agent.state_dict(),
                "hands_played": hands_played,
                "config":       cfg,
            }, step_path)
            last_ckpt = hands_played

        # --- Ctrl+C checkpoint ---
        if _interrupted[0]:
            interrupt_path = ckpt_dir / f"interrupt_{hands_played:010d}.pt"
            torch.save({
                "bet_agent":    bet_agent.state_dict(),
                "hands_played": hands_played,
                "config":       cfg,
            }, interrupt_path)
            print(f"[Interrupted] Saved: {interrupt_path}")
            if writer:
                writer.close()
            return

    # --- Final checkpoint ---
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "bet_agent":    bet_agent.state_dict(),
        "hands_played": hands_played,
        "config":       cfg,
    }, final_path)
    print(f"\nTraining complete.  Final checkpoint: {final_path}")
    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage B: train bet-sizing agent against frozen play oracle"
    )
    p.add_argument("--play-oracle", required=True,
                   help="Path to frozen play-oracle checkpoint (.pt).")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--total-hands", type=int, default=None,
                   help="Override total training hands (default: from config).")
    p.add_argument("--exp-name",    default="bet_stage",
                   help="Experiment name for checkpoints/logs.")
    p.add_argument("--checkpoint",  default=None,
                   help="Resume bet-agent from this filename inside <exp_name>/.")
    p.add_argument("--cpu",         action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
