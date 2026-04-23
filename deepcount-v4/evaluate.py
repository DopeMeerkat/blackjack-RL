"""
DeepCount — Evaluation & Interpretability Script
==================================================
Three evaluation modes.

1. basic_strategy
   Run the trained agent on many PLAYING-phase decisions and compare
   against published hard-total basic strategy.
   Reports overall accuracy and a per-(player_total, dealer_upcard) table.

2. count_probe
   Roll out full shoes and correlate the ShoeEncoder's auxiliary count
   prediction with the ground-truth Hi-Lo true count.
   Reports Pearson r and R² — a high R² means the network learned to count.

3. bet_sizing
   Record the agent's bet at each BETTING phase, binned by true count.
   A rational counter bets more at positive true counts.

Usage
-----
  python evaluate.py --mode all --checkpoint checkpoints/deepcount_final.pt
  python evaluate.py --mode basic_strategy --checkpoint path/to/ckpt.pt
  python evaluate.py --mode count_probe    --checkpoint path/to/ckpt.pt
  python evaluate.py --mode bet_sizing     --checkpoint path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from env.blackjack_env import (
    BlackjackShoeEnv,
    PHASE_BETTING,
    PHASE_PLAYING,
    HIT, STAND, DOUBLE, SPLIT,
    MIN_BET, MAX_BET,
)
from env.card_utils import hand_value
from models.policy_network import DeepCountNet
from utils import obs_to_tensor, build_action_mask


# ─────────────────────────────────────────────────────────────────────────────
# Basic strategy reference table (hard totals, 6-deck S17)
# ─────────────────────────────────────────────────────────────────────────────

def _build_basic_strategy() -> dict[tuple[int, int], int]:
    """
    Returns a mapping (player_hard_total, dealer_upcard_value) → action.
    dealer_upcard_value: 2-10 for pip cards, 11 for Ace.
    Covers hard totals 4-21 only (soft hands and pairs are omitted).
    """
    bs: dict[tuple[int, int], int] = {}
    for dealer in range(2, 12):          # 2-10, Ace=11
        for player in range(4, 22):
            if player <= 8:
                a = HIT
            elif player == 9:
                a = DOUBLE if 3 <= dealer <= 6 else HIT
            elif player == 10:
                a = DOUBLE if dealer <= 9 else HIT
            elif player == 11:
                a = DOUBLE if dealer <= 10 else HIT
            elif player == 12:
                a = STAND if 4 <= dealer <= 6 else HIT
            elif 13 <= player <= 16:
                a = STAND if dealer <= 6 else HIT
            else:                        # 17+
                a = STAND
            bs[(player, dealer)] = a
    return bs

BASIC_STRATEGY = _build_basic_strategy()
ACTION_NAME = {HIT: "H", STAND: "S", DOUBLE: "D", SPLIT: "P"}


# ─────────────────────────────────────────────────────────────────────────────
# Load policy
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(checkpoint: str, device: torch.device) -> DeepCountNet:
    policy = DeepCountNet().to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        step = ckpt.get("step", "?")
        print(f"  Loaded checkpoint '{checkpoint}'  (step {step})")
    else:
        print("  No checkpoint provided — evaluating random initialisation.")
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic Strategy Recovery
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_basic_strategy(
    policy: DeepCountNet,
    device: torch.device,
    n_play_decisions: int = 50_000,
    seed: int = 0,
):
    """
    Step through the environment, evaluate only PLAYING-phase decisions,
    and compare with hard-total basic strategy.

    Returns overall accuracy (float, 0-100).
    """
    env = BlackjackShoeEnv(seed=seed)
    obs, info = env.reset()

    match_count = 0
    total_count = 0
    # per-(player_total, dealer_up) stats
    cell_stats: dict[tuple[int, int], dict] = defaultdict(lambda: {"match": 0, "total": 0})

    while total_count < n_play_decisions:
        phase = int(obs["phase"])

        if phase == PHASE_BETTING:
            # Flat bet; we only care about play decisions
            action = {"bet": np.array([10.0 / MAX_BET], dtype=np.float32), "play": 0}
            obs, _, terminated, _, info = env.step(action)
            if terminated:
                obs, info = env.reset()
            continue

        # PLAYING phase
        hl   = int(obs["hand_len"])
        hand = list(obs["hand"][:hl])
        p_total, soft = hand_value(hand)
        d_val = int(obs["dealer_upcard"])
        if d_val == 1:
            d_val = 11  # Ace as 11 for BS lookup

        bs_action = BASIC_STRATEGY.get((p_total, d_val))

        if bs_action is not None and not soft:
            obs_t  = obs_to_tensor(obs, device)
            amask  = build_action_mask(env)
            amask_t = torch.tensor(amask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action_dict, _, _ = policy.get_action_and_logprob(
                    obs_t, action_mask=amask_t, deterministic=True
                )
            agent_action = int(action_dict["play"].item())

            hit = (agent_action == bs_action)
            if hit:
                match_count += 1
            total_count += 1
            cell_stats[(p_total, d_val)]["total"] += 1
            cell_stats[(p_total, d_val)]["match"] += int(hit)

            action = {"bet": np.array([0.0], dtype=np.float32), "play": agent_action}
        else:
            action = {"bet": np.array([0.0], dtype=np.float32), "play": STAND}

        obs, _, terminated, _, info = env.step(action)
        if terminated:
            obs, info = env.reset()

    accuracy = 100.0 * match_count / max(total_count, 1)

    print(f"\n── Basic Strategy Recovery ──────────────────────────────────")
    print(f"  Decisions evaluated : {total_count:,}")
    print(f"  Overall accuracy    : {accuracy:.1f}%")
    print(f"\n  Sample cells (player_total vs dealer_up):")
    print(f"  {'P\\D':>4}", end="")
    for d in range(2, 12):
        print(f"  {d:>3}", end="")
    print()
    for p in [8, 10, 12, 16, 17]:
        print(f"  {p:>4}", end="")
        for d in range(2, 12):
            st = cell_stats.get((p, d), {"match": 0, "total": 0})
            if st["total"] > 0:
                acc = 100 * st["match"] / st["total"]
                print(f"  {acc:>3.0f}", end="")
            else:
                print(f"   --", end="")
        print()

    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 2. Count Probe (ShoeEncoder interpretability)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_count_probe(
    policy: DeepCountNet,
    device: torch.device,
    n_shoes: int = 50,
    seed: int = 42,
):
    """
    Correlate the ShoeEncoder auxiliary count prediction with ground truth.
    High R² means the Transformer has learned to card-count.

    Returns R² (float, 0-1).
    """
    env = BlackjackShoeEnv(seed=seed)
    pred_list: list[float] = []
    true_list: list[float] = []

    for _ in range(n_shoes):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            obs_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                out = policy.forward(obs_t)
                pred_tc = float(out["count_pred"].item())

            gt_tc = float(info["true_count"])
            pred_list.append(pred_tc)
            true_list.append(gt_tc)

            # Dummy actions to advance the shoe
            if obs["phase"] == PHASE_BETTING:
                action = {"bet": np.array([0.02], dtype=np.float32), "play": 0}
            else:
                action = {"bet": np.array([0.02], dtype=np.float32), "play": STAND}

            obs, _, terminated, _, info = env.step(action)

    pred = np.array(pred_list)
    true = np.array(true_list)

    r = float(np.corrcoef(pred, true)[0, 1])
    r2 = r ** 2
    mae = float(np.mean(np.abs(pred - true)))

    print(f"\n── Count Probe ──────────────────────────────────────────────")
    print(f"  Shoes evaluated     : {n_shoes}")
    print(f"  Observations        : {len(pred):,}")
    print(f"  Pearson r           : {r:+.4f}")
    print(f"  R²                  : {r2:.4f}")
    print(f"  Mean absolute error : {mae:.4f} (true count units)")

    return r2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bet Sizing vs. True Count
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bet_sizing(
    policy: DeepCountNet,
    device: torch.device,
    n_shoes: int = 100,
    seed: int = 7,
):
    """
    Record the agent's bet at each BETTING phase, binned by true count.
    A rational card counter bets more at high positive true counts.

    Returns dict mapping TC bin → list of bets.
    """
    env = BlackjackShoeEnv(seed=seed)   # no flat_bet — agent chooses
    bets_by_bin: dict[int, list[float]] = defaultdict(list)

    for _ in range(n_shoes):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            if obs["phase"] == PHASE_BETTING:
                tc  = float(info["true_count"])
                obs_t  = obs_to_tensor(obs, device)
                amask_t = torch.ones(1, 5, dtype=torch.bool, device=device)

                with torch.no_grad():
                    action_dict, _, _ = policy.get_action_and_logprob(
                        obs_t, action_mask=amask_t, deterministic=True
                    )
                bet = float(action_dict["bet"].item())
                bin_key = int(np.clip(round(tc), -6, 6))
                bets_by_bin[bin_key].append(bet)

                action = {
                    "bet":  action_dict["bet"][0].cpu().numpy(),
                    "play": 0,
                }
            else:
                action = {
                    "bet":  np.array([0.02], dtype=np.float32),
                    "play": STAND,
                }

            obs, _, terminated, _, info = env.step(action)

    print(f"\n── Bet Sizing by True Count ─────────────────────────────────")
    print(f"  Shoes evaluated : {n_shoes}")
    print(f"\n  {'TC':>4}  {'Mean Bet':>10}  {'Std':>8}  {'N':>6}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*6}")
    for b in range(-6, 7):
        bets = bets_by_bin.get(b, [])
        if bets:
            print(
                f"  {b:>+4}  "
                f"{np.mean(bets):>10.1f}  "
                f"{np.std(bets):>8.1f}  "
                f"{len(bets):>6}"
            )

    return dict(bets_by_bin)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained DeepCount agent.")
    p.add_argument("--mode",
                   choices=["basic_strategy", "count_probe", "bet_sizing", "all"],
                   default="all")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to a .pt checkpoint (omit to use random weights)")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--n_shoes",    type=int, default=50,
                   help="Shoes for count_probe / bet_sizing")
    p.add_argument("--n_decisions",type=int, default=50_000,
                   help="Play decisions for basic_strategy eval")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"\n{'━'*60}")
    print(f"  DeepCount — Evaluation  (mode={args.mode})")
    print(f"{'━'*60}")

    policy = load_policy(args.checkpoint, device)

    if args.mode in ("basic_strategy", "all"):
        evaluate_basic_strategy(policy, device,
                                n_play_decisions=args.n_decisions,
                                seed=args.seed)
    if args.mode in ("count_probe", "all"):
        evaluate_count_probe(policy, device,
                             n_shoes=args.n_shoes,
                             seed=args.seed)
    if args.mode in ("bet_sizing", "all"):
        evaluate_bet_sizing(policy, device,
                            n_shoes=args.n_shoes,
                            seed=args.seed)

    print()


if __name__ == "__main__":
    main()
