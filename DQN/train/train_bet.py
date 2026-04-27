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
from collections import deque
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
# Oracle helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _oracle_act(oracle: DQNAgent, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Greedy oracle action selection — skips reset_noise (oracle is deterministic)."""
    obs_t  = torch.tensor(obs,   dtype=torch.float32, device=oracle.device)
    mask_t = torch.tensor(masks, dtype=torch.bool,    device=oracle.device)
    play_q = oracle.online_net(obs_t).clone()
    play_q[~mask_t] = -1e9
    return play_q.argmax(dim=1).cpu().numpy().astype(np.int32)


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
    bet_multipliers_list = [float(m) for m in cfg.get("bet_multipliers", [1, 2, 4, 8, 12])]
    n_bet_actions = len(bet_multipliers_list)

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
    pending_bet_mult   = np.zeros(num_envs, dtype=np.float32)
    has_pending_bet    = np.zeros(num_envs, dtype=bool)

    eval_every  = cfg.get("eval_every_n_hands", 1_000_000)
    ckpt_every  = cfg.get("eval_every_n_hands", 1_000_000)
    train_every = int(cfg.get("train_every_n_steps", 4))
    last_eval   = hands_played
    last_ckpt   = hands_played
    t_start     = time.time()
    global_step = 0

    # Rolling stats — deque auto-drops oldest element, O(1) append/pop.
    recent_rewards     = deque(maxlen=100_000)
    recent_multipliers = deque(maxlen=100_000)
    recent_bet_actions = deque(maxlen=100_000)
    recent_losses      = deque(maxlen=10_000)
    _pending_train = 0   # accumulated hands; flush one train_step per train_every

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
                act = int(bet_acts[j])
                pending_bet_obs[i]    = bet_obs_batch[j]
                pending_bet_action[i] = act
                pending_bet_mult[i]   = bet_multipliers_list[act]
                has_pending_bet[i]    = True
                actions[i]            = act

        # --- Play phase: oracle selects deterministically (no reset_noise) ---
        play_envs = np.where(~in_bet)[0]
        if len(play_envs) > 0:
            actions[play_envs] = _oracle_act(
                play_agent, obs_arr[play_envs], masks_arr[play_envs]
            )

        # --- Step ---
        next_obs, next_masks, rewards, dones, infos = vec_env.step(actions)

        # --- Handle terminal hands ---
        for i in range(num_envs):
            if dones[i]:
                hands_played += 1
                hand_rew = float(rewards[i])
                recent_rewards.append(hand_rew)

                if has_pending_bet[i]:
                    mult = float(pending_bet_mult[i])
                    act  = int(pending_bet_action[i])
                    recent_multipliers.append(mult)
                    recent_bet_actions.append(act)
                    # Normalise by the bet multiplier so Q-values estimate the
                    # per-unit outcome (≈ ±1 std) rather than the scaled reward
                    # (std up to ≈13 for 12× bets). Action selection re-applies
                    # the multiplier via BetAgent._weighted().
                    norm_rew = hand_rew / max(mult, 1e-8)
                    bet_agent.add_transition(pending_bet_obs[i], act, norm_rew)
                    has_pending_bet[i] = False
                    _pending_train += 1

                reset_obs, reset_mask, reset_info = vec_env.reset_at(i)
                next_obs[i]   = reset_obs
                next_masks[i] = reset_mask
                infos[i]      = reset_info

        # --- Gradient updates: one step per train_every completed hands ---
        while _pending_train >= train_every:
            loss = bet_agent.train_step()
            _pending_train -= train_every
            if loss is not None:
                recent_losses.append(loss)

        obs_arr   = next_obs
        masks_arr = next_masks
        global_step += 1

        # --- Per-step TB logging (mirrors train_curriculum every 1 000 steps) ---
        if writer and global_step % 1_000 == 0:
            if recent_losses:
                writer.add_scalar("train/bet_loss",
                                  float(np.mean(recent_losses)), hands_played)
            writer.add_scalar("train/replay_size",
                              len(bet_agent.replay), hands_played)
            if recent_multipliers:
                writer.add_scalar("train/avg_multiplier",
                                  float(np.mean(recent_multipliers)),
                                  hands_played)

        # --- Periodic eval ---
        if hands_played - last_eval >= eval_every and hands_played > last_eval:
            elapsed   = time.time() - t_start
            rate      = hands_played / elapsed

            # EV per unit wagered: rewards are already scaled by bet_multiplier,
            # so we divide by the average multiplier to recover per-unit return.
            mean_ev_hand = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
            mean_mult    = float(np.mean(recent_multipliers)) if recent_multipliers else 1.0
            ev_per_unit  = mean_ev_hand / max(mean_mult, 1e-8)
            mean_loss    = float(np.mean(recent_losses)) if recent_losses else float("nan")

            # Bet-action distribution over the recent window
            act_counts = np.bincount(
                np.array(recent_bet_actions, dtype=np.int32), minlength=n_bet_actions
            ) if recent_bet_actions else np.zeros(n_bet_actions, dtype=np.int64)
            act_pcts   = act_counts / max(act_counts.sum(), 1)
            bet_str = "  ".join(
                f"{bet_multipliers_list[a]:.0f}x={act_pcts[a]*100:.1f}%"
                for a in range(n_bet_actions)
            )

            print(
                f"  hands={hands_played:>10,}  "
                f"EV/unit={ev_per_unit*100:+.3f}%  "
                f"EV/hand={mean_ev_hand*100:+.3f}%  "
                f"avg_mult={mean_mult:.2f}x  "
                f"loss={mean_loss:.5f}  "
                f"buf={len(bet_agent.replay):,}  "
                f"rate={rate/1000:.1f}k/s\n"
                f"    bets: {bet_str}"
            )

            if writer:
                writer.add_scalar("eval/ev_per_unit",   ev_per_unit,   hands_played)
                writer.add_scalar("eval/ev_per_hand",   mean_ev_hand,  hands_played)
                writer.add_scalar("eval/avg_multiplier", mean_mult,    hands_played)
                writer.add_scalar("eval/bet_loss",       mean_loss,    hands_played)
                for a in range(n_bet_actions):
                    writer.add_scalar(
                        f"eval/bet_action_pct_{a}", float(act_pcts[a]), hands_played
                    )

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
