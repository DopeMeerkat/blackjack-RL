"""Evaluate the joint (playing + bet-sizing) policy for Milestone 4.

Acceptance tests (blackjack_rl_design.md §13, Milestone 4):
  After joint training for 100M hands, evaluated over 1M hands:
  1. Pearson correlation between true count and chosen bet multiplier >= 0.6
  2. Joint policy EV per shoe is positive
  3. Joint policy EV per shoe exceeds the flat-betting variant of the same
     playing policy by a statistically significant margin (paired-difference
     t-test on identical shoe seeds, p < 0.01)

Usage:
  python eval/eval_betting.py --checkpoint outputs/checkpoints/joint/final.pt
  python eval/eval_betting.py --checkpoint ... --eval-hands 2000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.network import BlackjackNet
from env.blackjack import BlackjackEnv

_BET_MULTIPLIERS = [1, 2, 4, 8, 12]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_net(checkpoint_path: str, device: torch.device) -> tuple[BlackjackNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    ncfg = {}
    for v in cfg.values() if isinstance(cfg, dict) else [cfg]:
        if isinstance(v, dict):
            ncfg.update(v)

    net = BlackjackNet(ncfg).to(device)
    net.load_state_dict(ckpt["agent"]["online_net"])
    return net, cfg


# ---------------------------------------------------------------------------
# Single-shoe rollout
# ---------------------------------------------------------------------------

def play_shoe(
    net: BlackjackNet,
    cfg: dict,
    device: torch.device,
    seed: int,
    use_bet_head: bool,
) -> tuple[float, list[tuple[float, int]]]:
    """Play one full shoe and return (total_reward, [(tc, bet_mult), ...]).

    When use_bet_head=False, all bets are flat 1x (for the paired comparison).
    """
    env = BlackjackEnv(cfg, seed=seed)
    obs, mask, info = env.reset()

    shoe_reward = 0.0
    tc_bet_pairs = []

    while True:
        if info["phase"] == "bet":
            if use_bet_head:
                obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    _, bet_q = net(obs_t)
                bet_idx = int(bet_q.argmax(dim=1).item())
            else:
                bet_idx = 0  # flat 1x

            tc_bet_pairs.append((info["true_count"], _BET_MULTIPLIERS[bet_idx]))
            action = bet_idx
        else:
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask[None], dtype=torch.bool, device=device)
            with torch.no_grad():
                play_q, _ = net(obs_t)
                play_q = play_q.clone()
                play_q[~mask_t] = -1e9
            action = int(play_q.argmax(dim=1).item())

        obs, mask, reward, done, info = env.step(action)
        if done:
            shoe_reward += reward
            # Check if shoe is nearly depleted (env reshuffles at start of
            # next hand, but we track cards_in_shoe to decide when to stop)
            if env.cards_in_shoe / 52.0 < cfg.get("reshuffle_threshold", 1.5):
                break
            obs, mask, info = env.reset()

    return shoe_reward, tc_bet_pairs


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: BlackjackNet,
    cfg: dict,
    device: torch.device,
    n_shoes: int,
    base_seed: int,
) -> dict:
    """Run paired evaluation: joint policy vs flat-bet variant on same seeds.

    Returns a dict with all metrics needed for the Milestone 4 report.
    """
    net.set_deterministic(True)

    joint_shoe_rewards = []
    flat_shoe_rewards = []
    all_tc_bet_pairs = []

    for i in range(n_shoes):
        seed = base_seed + i

        # Joint policy (bet head active)
        jr, tc_bets = play_shoe(net, cfg, device, seed, use_bet_head=True)
        joint_shoe_rewards.append(jr)
        all_tc_bet_pairs.extend(tc_bets)

        # Flat-bet variant (same playing policy, 1x bet always)
        fr, _ = play_shoe(net, cfg, device, seed, use_bet_head=False)
        flat_shoe_rewards.append(fr)

    net.set_deterministic(False)

    joint_arr = np.array(joint_shoe_rewards)
    flat_arr = np.array(flat_shoe_rewards)

    # --- Metric 1: TC-bet Pearson correlation ---
    tc_arr = np.array([p[0] for p in all_tc_bet_pairs])
    bet_arr = np.array([p[1] for p in all_tc_bet_pairs], dtype=np.float64)
    if tc_arr.std() > 0 and bet_arr.std() > 0:
        tc_bet_corr = float(np.corrcoef(tc_arr, bet_arr)[0, 1])
    else:
        tc_bet_corr = 0.0

    # --- Metric 2: Joint EV per shoe ---
    joint_ev_shoe = float(joint_arr.mean())
    joint_ev_stderr = float(joint_arr.std() / np.sqrt(len(joint_arr)))

    # --- Metric 3: Paired t-test (joint vs flat) ---
    diff = joint_arr - flat_arr
    t_stat, p_value = sp_stats.ttest_1samp(diff, 0.0)

    # Per-hand EV
    n_hands = len(all_tc_bet_pairs)
    joint_ev_hand = float(joint_arr.sum()) / n_hands if n_hands > 0 else 0.0
    flat_ev_hand = float(flat_arr.sum()) / n_hands if n_hands > 0 else 0.0

    # Bet distribution
    bet_counts = np.zeros(5, dtype=np.int64)
    for _, mult in all_tc_bet_pairs:
        idx = _BET_MULTIPLIERS.index(mult)
        bet_counts[idx] += 1

    # Bet distribution by TC bucket
    tc_buckets = {}
    for tc, mult in all_tc_bet_pairs:
        bucket = _tc_bucket(tc)
        if bucket not in tc_buckets:
            tc_buckets[bucket] = np.zeros(5, dtype=np.int64)
        tc_buckets[bucket][_BET_MULTIPLIERS.index(mult)] += 1

    return {
        "n_shoes": n_shoes,
        "n_hands": n_hands,
        "tc_bet_corr": tc_bet_corr,
        "joint_ev_shoe": joint_ev_shoe,
        "joint_ev_shoe_stderr": joint_ev_stderr,
        "flat_ev_shoe": float(flat_arr.mean()),
        "flat_ev_shoe_stderr": float(flat_arr.std() / np.sqrt(len(flat_arr))),
        "joint_ev_hand": joint_ev_hand,
        "flat_ev_hand": flat_ev_hand,
        "diff_mean": float(diff.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "bet_counts": bet_counts,
        "tc_buckets": tc_buckets,
    }


def _tc_bucket(tc: float) -> str:
    if tc <= -3:
        return "TC<=-3"
    elif tc <= -1:
        return "-3<TC<=-1"
    elif tc < 1:
        return "-1<TC<1"
    elif tc < 3:
        return "1<=TC<3"
    else:
        return "TC>=3"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict) -> None:
    print("\n" + "=" * 70)
    print("Milestone 4 — Joint Policy Bet-Sizing Evaluation")
    print("=" * 70)
    print(f"  Shoes evaluated:   {results['n_shoes']:,}")
    print(f"  Hands evaluated:   {results['n_hands']:,}")

    # Test 1: TC-bet correlation
    corr = results["tc_bet_corr"]
    pass1 = corr >= 0.6
    print(f"\n  Test 1: TC-bet Pearson correlation")
    print(f"    Correlation:     {corr:.4f}  (threshold: >= 0.6)")
    print(f"    Result:          {'PASS' if pass1 else 'FAIL'}")

    # Test 2: Positive EV per shoe
    jev = results["joint_ev_shoe"]
    jev_se = results["joint_ev_shoe_stderr"]
    pass2 = jev > 0
    print(f"\n  Test 2: Positive joint EV per shoe")
    print(f"    Joint EV/shoe:   {jev:+.4f} +/- {jev_se:.4f}")
    print(f"    Joint EV/hand:   {results['joint_ev_hand']*100:+.4f}%")
    print(f"    Result:          {'PASS' if pass2 else 'FAIL'}")

    # Test 3: Joint > flat (paired t-test, p < 0.01)
    pass3 = results["p_value"] < 0.01 and results["diff_mean"] > 0
    print(f"\n  Test 3: Joint > flat-bet (paired t-test)")
    print(f"    Flat EV/shoe:    {results['flat_ev_shoe']:+.4f} +/- {results['flat_ev_shoe_stderr']:.4f}")
    print(f"    Flat EV/hand:    {results['flat_ev_hand']*100:+.4f}%")
    print(f"    Diff (J-F)/shoe: {results['diff_mean']:+.4f}")
    print(f"    t-statistic:     {results['t_stat']:.3f}")
    print(f"    p-value:         {results['p_value']:.6f}  (threshold: < 0.01)")
    print(f"    Result:          {'PASS' if pass3 else 'FAIL'}")

    # Bet distribution
    total_bets = results["bet_counts"].sum()
    print(f"\n  Bet distribution (overall):")
    for i, m in enumerate(_BET_MULTIPLIERS):
        pct = results["bet_counts"][i] / max(total_bets, 1)
        bar = "#" * int(pct * 40)
        print(f"    {m:2d}x: {pct*100:5.1f}%  {bar}")

    # Bet distribution by TC bucket
    print(f"\n  Bet distribution by true-count bucket:")
    for bucket in ["TC<=-3", "-3<TC<=-1", "-1<TC<1", "1<=TC<3", "TC>=3"]:
        if bucket in results["tc_buckets"]:
            bc = results["tc_buckets"][bucket]
            total = bc.sum()
            if total > 0:
                pcts = bc / total
                line = "  ".join(f"{m}x:{pcts[i]*100:4.0f}%" for i, m in enumerate(_BET_MULTIPLIERS))
                print(f"    {bucket:>12s} (n={total:>6d}): {line}")

    # Summary
    print("\n" + "-" * 70)
    all_pass = pass1 and pass2 and pass3
    print(f"  MILESTONE 4 ACCEPTANCE: {'PASS' if all_pass else 'FAIL'}")
    print(f"    Test 1 (correlation >= 0.6): {'PASS' if pass1 else 'FAIL'}")
    print(f"    Test 2 (positive shoe EV):   {'PASS' if pass2 else 'FAIL'}")
    print(f"    Test 3 (joint > flat, p<.01):{'PASS' if pass3 else 'FAIL'}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    net, cfg = load_net(args.checkpoint, device)
    net.eval()

    # Estimate shoes needed for the requested number of hands.
    # A 6-deck shoe at 75% penetration yields roughly 60-80 hands.
    hands_per_shoe = 70
    n_shoes = max(args.eval_hands // hands_per_shoe, 100)

    print(f"Evaluating over {n_shoes:,} shoes (~{n_shoes * hands_per_shoe:,} hands)...")
    results = evaluate(net, cfg, device, n_shoes, base_seed=args.seed)
    print_report(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate joint playing + bet-sizing policy")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a Milestone 4 checkpoint .pt file")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--eval-hands", type=int, default=1_000_000,
                   help="Approximate number of hands to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
