"""Evaluate a trained BetAgent against a frozen play oracle.

Outputs (printed to stdout):
  1. Bet distribution per true-count bucket — how often each multiplier is
     chosen at each TC level.
  2. Average bet multiplier per true-count bucket (TC-bet correlation).
  3. EV comparison: bet-agent bets vs flat 1× baseline, both using the same
     play oracle and the same shoe seeds.

Interpretation guide
--------------------
A well-trained bet agent should spread bets: small at negative / neutral TC,
large at TC ≥ +2 where player has the edge.  "EV per unit wagered" measures
the return on each chip risked; it should exceed the flat-1× baseline if the
agent has learned the correct TC → bet relationship.

Usage:
  python eval/bet_agent_eval.py \
    --play-oracle outputs/checkpoints/play_oracle/final.pt \
    --bet-agent   outputs/checkpoints/bet_stage/final.pt \
    [--eval-hands 500000] [--num-envs 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.bet_agent import BetAgent
from agent.dqn import DQNAgent
from env.bet_encoding import BET_OBS_DIM, encode_bet_obs
from env.vec_env import VecBlackjackEnv


# ---------------------------------------------------------------------------
# TC bucket helpers
# ---------------------------------------------------------------------------

_TC_BUCKET_EDGES = [-3, -2, -1, 0, 1, 2, 3]   # 7 buckets: ≤-3 … ≥+3
_TC_BUCKET_LABELS = ["≤-3", " -2", " -1", "  0", " +1", " +2", "≥+3"]
_N_TC_BUCKETS = len(_TC_BUCKET_LABELS)


def tc_to_bucket(tc: float) -> int:
    """Map a true count to one of 7 display buckets."""
    if tc < -2:
        return 0
    elif tc < -1:
        return 1
    elif tc < 0:
        return 2
    elif tc < 1:
        return 3
    elif tc < 2:
        return 4
    elif tc < 3:
        return 5
    else:
        return 6


# ---------------------------------------------------------------------------
# Config / loader helpers
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


def load_play_oracle(checkpoint_path: str, fallback_cfg: dict,
                     device: torch.device) -> DQNAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", fallback_cfg)
    agent = DQNAgent(
        net_config=_net_config(cfg),
        train_config=_train_config(cfg),
        replay_capacity=1_000,
        device=device,
    )
    agent.load_state_dict(ckpt["agent"])
    agent.online_net.set_deterministic(True)
    for p in agent.online_net.parameters():
        p.requires_grad_(False)
    return agent


def load_bet_agent(checkpoint_path: str, fallback_cfg: dict,
                   device: torch.device) -> BetAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", fallback_cfg)
    agent = BetAgent(cfg, device)
    agent.load_state_dict(ckpt["bet_agent"])
    return agent


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_eval(
    play_agent: DQNAgent,
    bet_agent: BetAgent | None,     # None → flat 1× baseline
    cfg: dict,
    n_hands: int,
    num_envs: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Run n_hands hands and collect per-hand statistics.

    Returns a dict with keys:
      rewards          (n_hands,) float — scaled hand reward (already * multiplier)
      bet_actions      (n_hands,) int   — bet action chosen (0 = 1×)
      tc_at_bet        (n_hands,) float — true count at bet time
      bet_multipliers  (n_hands,) float — actual multiplier values
    """
    bet_multipliers_list: list[float] = [
        float(m) for m in cfg.get("bet_multipliers", [1, 2, 4, 8, 12])
    ]
    bet_obs_dim = int(cfg.get("bet_obs_dim", BET_OBS_DIM))

    vec_env = VecBlackjackEnv(
        num_envs, cfg,
        seeds=[seed + i for i in range(num_envs)],
    )
    obs_arr, masks_arr, infos = vec_env.reset()

    pending_bet_obs    = np.zeros((num_envs, bet_obs_dim), dtype=np.float32)
    pending_bet_action = np.zeros(num_envs, dtype=np.int32)
    pending_tc         = np.zeros(num_envs, dtype=np.float32)
    has_pending_bet    = np.zeros(num_envs, dtype=bool)

    rewards:         list[float] = []
    bet_actions_out: list[int]   = []
    tc_at_bet_out:   list[float] = []

    while len(rewards) < n_hands:
        in_bet  = vec_env.in_bet_phase
        actions = np.zeros(num_envs, dtype=np.int32)

        # Bet phase
        bet_envs = np.where(in_bet)[0]
        if len(bet_envs) > 0:
            rank_counts = vec_env.get_rank_counts_batch(bet_envs)
            bet_obs_batch = np.stack([
                encode_bet_obs(
                    rank_counts[j],
                    float(infos[i]["true_count"]),
                    float(infos[i]["decks_remaining"]),
                )
                for j, i in enumerate(bet_envs)
            ])
            if bet_agent is not None:
                bet_acts = bet_agent.select_actions_batch(bet_obs_batch, greedy=True)
            else:
                bet_acts = np.zeros(len(bet_envs), dtype=np.int32)  # flat 1×

            for j, i in enumerate(bet_envs):
                pending_bet_obs[i]    = bet_obs_batch[j]
                pending_bet_action[i] = int(bet_acts[j])
                pending_tc[i]         = float(infos[i]["true_count"])
                has_pending_bet[i]    = True
                actions[i]            = int(bet_acts[j])

        # Play phase — oracle
        play_envs = np.where(~in_bet)[0]
        if len(play_envs) > 0:
            actions[play_envs] = play_agent.select_play_actions_batch(
                obs_arr[play_envs], masks_arr[play_envs]
            )

        next_obs, next_masks, env_rewards, dones, infos = vec_env.step(actions)

        for i in range(num_envs):
            if dones[i]:
                if has_pending_bet[i]:
                    rewards.append(float(env_rewards[i]))
                    bet_actions_out.append(int(pending_bet_action[i]))
                    tc_at_bet_out.append(float(pending_tc[i]))
                    has_pending_bet[i] = False

                if len(rewards) >= n_hands:
                    break
                reset_obs, reset_mask, reset_info = vec_env.reset_at(i)
                next_obs[i]   = reset_obs
                next_masks[i] = reset_mask
                infos[i]      = reset_info

        obs_arr   = next_obs
        masks_arr = next_masks

    rewards_arr  = np.array(rewards[:n_hands],         dtype=np.float64)
    bet_acts_arr = np.array(bet_actions_out[:n_hands],  dtype=np.int32)
    tc_arr       = np.array(tc_at_bet_out[:n_hands],    dtype=np.float64)
    mult_arr     = np.array(
        [bet_multipliers_list[a] for a in bet_acts_arr], dtype=np.float64
    )

    return {
        "rewards":         rewards_arr,
        "bet_actions":     bet_acts_arr,
        "tc_at_bet":       tc_arr,
        "bet_multipliers": mult_arr,
    }


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

def print_bet_distribution(
    stats: dict,
    bet_multipliers_list: list[float],
    title: str = "Bet distribution",
) -> None:
    n_actions = len(bet_multipliers_list)
    bet_acts  = stats["bet_actions"]
    tc_arr    = stats["tc_at_bet"]

    # Header
    mult_labels = [f"{m:4.0f}x" for m in bet_multipliers_list]
    header_cols = "  ".join(mult_labels)
    print(f"\n{'─'*60}")
    print(f"{title}")
    print(f"{'─'*60}")
    print(f"{'TC':>5}  {header_cols}  {'avg mult':>9}  {'  count':>8}")
    print(f"{'─'*60}")

    for b in range(_N_TC_BUCKETS):
        mask = np.array([tc_to_bucket(tc) == b for tc in tc_arr])
        n    = mask.sum()
        if n == 0:
            continue
        acts_in_bucket = bet_acts[mask]
        counts = np.bincount(acts_in_bucket, minlength=n_actions)
        pcts   = counts / n * 100
        mults  = np.array([bet_multipliers_list[a] for a in acts_in_bucket])
        avg_m  = float(mults.mean())
        pct_str = "  ".join(f"{p:4.0f}%" for p in pcts)
        print(f"{_TC_BUCKET_LABELS[b]:>5}  {pct_str}  {avg_m:9.2f}  {n:8,}")

    print(f"{'─'*60}")
    overall_mults = np.array(
        [bet_multipliers_list[a] for a in bet_acts], dtype=np.float64
    )
    print(f"{'All':>5}  {'':>{5*n_actions+2*(n_actions-1)}}  "
          f"{overall_mults.mean():9.2f}  {len(bet_acts):8,}")
    print(f"{'─'*60}")


def print_ev_comparison(
    agent_stats: dict,
    flat_stats:  dict,
) -> None:
    rew_agent = agent_stats["rewards"]
    rew_flat  = flat_stats["rewards"]
    mult_arr  = agent_stats["bet_multipliers"]

    ev_flat             = float(rew_flat.mean())
    ev_agent_raw        = float(rew_agent.mean())    # mean scaled reward per hand
    avg_mult            = float(mult_arr.mean())
    ev_agent_per_unit   = ev_agent_raw / avg_mult    # per unit wagered

    # 95 % CI via CLT
    def _ci(arr: np.ndarray) -> float:
        return 1.96 * float(arr.std()) / np.sqrt(len(arr))

    ci_flat  = _ci(rew_flat)
    ci_agent = _ci(rew_agent)

    print(f"\n{'─'*60}")
    print("EV comparison (per hand in units of base bet)")
    print(f"{'─'*60}")
    print(f"  Flat 1× baseline:          {ev_flat*100:+.4f}%  "
          f"(±{ci_flat*100:.4f}%, n={len(rew_flat):,})")
    print(f"  Bet agent raw EV:          {ev_agent_raw*100:+.4f}%  "
          f"(±{ci_agent*100:.4f}%, n={len(rew_agent):,})")
    print(f"  Bet agent avg multiplier:  {avg_mult:.3f}×")
    print(f"  Bet agent EV per unit:     {ev_agent_per_unit*100:+.4f}%")
    delta = ev_agent_per_unit - ev_flat
    print(f"  Per-unit delta vs flat:    {delta*100:+.4f}%")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )
    cfg = load_config(args.config)
    bet_multipliers_list = [float(m) for m in cfg.get("bet_multipliers", [1, 2, 4, 8, 12])]

    print(f"Device:        {device}")
    print(f"Eval hands:    {args.eval_hands:,}")
    print(f"Num envs:      {args.num_envs}")

    print(f"\nLoading play oracle: {args.play_oracle}")
    play_agent = load_play_oracle(args.play_oracle, cfg, device)

    print(f"Loading bet agent:   {args.bet_agent}")
    bet_agent = load_bet_agent(args.bet_agent, cfg, device)

    # Run both evaluations with the same starting seeds so shoe states match.
    seed = args.seed

    print(f"\nRunning bet-agent evaluation  ({args.eval_hands:,} hands)…")
    agent_stats = run_eval(
        play_agent, bet_agent, cfg, args.eval_hands, args.num_envs, device, seed
    )

    print(f"Running flat-1× baseline      ({args.eval_hands:,} hands)…")
    flat_stats = run_eval(
        play_agent, None, cfg, args.eval_hands, args.num_envs, device, seed
    )

    print_bet_distribution(agent_stats, bet_multipliers_list,
                           title="Bet distribution by true-count bucket")
    print_ev_comparison(agent_stats, flat_stats)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate bet agent: TC correlation, distribution, EV vs flat 1×"
    )
    p.add_argument("--play-oracle", required=True,
                   help="Path to frozen play-oracle checkpoint (.pt).")
    p.add_argument("--bet-agent",   required=True,
                   help="Path to trained bet-agent checkpoint (.pt).")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--eval-hands",  type=int, default=500_000)
    p.add_argument("--num-envs",    type=int, default=64)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--cpu",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
