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
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.network import BlackjackNet
from agent.dqn import DQNAgent
from env.vec_env import VecBlackjackEnv
from train.train_curriculum import net_config, train_config

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
# Vectorised paired rollout
# ---------------------------------------------------------------------------

def _compute_actions_batched(
    net: BlackjackNet,
    obs_j: np.ndarray,
    mask_j: np.ndarray,
    obs_f: np.ndarray,
    mask_f: np.ndarray,
    bet_phase_j: np.ndarray,
    bet_phase_f: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """One network forward over both joint+flat batches; return action arrays.

    Joint and flat envs see different observations (bet multiplier in obs[27]),
    so play decisions can't be shared. But concatenating both into a single
    (2N, 28) forward amortises GPU launch overhead — the dominant cost in the
    pre-vectorised version.
    """
    n = obs_j.shape[0]
    all_obs   = np.concatenate([obs_j,  obs_f],  axis=0)
    all_masks = np.concatenate([mask_j, mask_f], axis=0)

    obs_t  = torch.from_numpy(all_obs).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(all_masks).to(device=device, dtype=torch.bool)

    with torch.no_grad():
        play_q, bet_q = net(obs_t)
        play_q = play_q.masked_fill(~mask_t, -1e9)
        play_arg = play_q.argmax(dim=1).cpu().numpy().astype(np.int32)
        bet_arg  = bet_q.argmax(dim=1).cpu().numpy().astype(np.int32)

    play_j, play_f = play_arg[:n], play_arg[n:]
    bet_j          = bet_arg[:n]   # only joint uses learned bets

    actions_j = np.where(bet_phase_j, bet_j, play_j).astype(np.int32)
    actions_f = np.where(bet_phase_f, np.int32(0), play_f).astype(np.int32)
    return actions_j, actions_f


def play_round(
    net: BlackjackNet,
    cfg: dict,
    device: torch.device,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, int]]]:
    """Play one shoe in each of len(seeds) parallel envs, paired joint+flat.

    Returns (joint_shoe_rewards, flat_shoe_rewards, tc_bet_pairs).  Each reward
    array has shape (len(seeds),); entry i is the total reward for env i's
    single shoe under the corresponding policy.  Joint and flat use identical
    seeds so the paired t-test on shoe rewards remains valid.
    """
    n = len(seeds)
    reshuffle_threshold = cfg.get("reshuffle_threshold", 1.5)

    joint_env = VecBlackjackEnv(n, cfg, seeds=seeds)
    flat_env  = VecBlackjackEnv(n, cfg, seeds=seeds)

    obs_j, mask_j, info_j = joint_env.reset()
    obs_f, mask_f, info_f = flat_env.reset()

    joint_rewards = np.zeros(n, dtype=np.float64)
    flat_rewards  = np.zeros(n, dtype=np.float64)
    joint_active  = np.ones(n, dtype=bool)
    flat_active   = np.ones(n, dtype=bool)
    tc_bet_pairs: list[tuple[float, int]] = []

    while joint_active.any() or flat_active.any():
        bet_phase_j = np.array([i["phase"] == "bet" for i in info_j], dtype=bool)
        bet_phase_f = np.array([i["phase"] == "bet" for i in info_f], dtype=bool)

        actions_j, actions_f = _compute_actions_batched(
            net, obs_j, mask_j, obs_f, mask_f,
            bet_phase_j, bet_phase_f, device,
        )

        # Record (true_count, chosen_bet) for the joint env's bet decisions
        # (active envs only).
        record_idx = np.where(joint_active & bet_phase_j)[0]
        for i in record_idx:
            tc_bet_pairs.append((
                info_j[i]["true_count"],
                _BET_MULTIPLIERS[int(actions_j[i])],
            ))

        obs_j, mask_j, rews_j, dones_j, info_j = joint_env.step(actions_j)
        obs_f, mask_f, rews_f, dones_f, info_f = flat_env.step(actions_f)

        joint_rewards += rews_j * joint_active
        flat_rewards  += rews_f * flat_active

        # When a hand finishes, either close out the shoe (mark inactive so its
        # rewards stop being recorded) or reset to start the next hand.  The
        # zombie envs continue stepping in lockstep but their results are
        # masked out — extra GPU work is bounded by the variance in shoe length.
        for i in np.where(joint_active & dones_j)[0]:
            if joint_env._envs[i].cards_in_shoe / 52.0 < reshuffle_threshold:
                joint_active[i] = False
            else:
                o, m, inf = joint_env.reset_at(i)
                obs_j[i], mask_j[i], info_j[i] = o, m, inf
        for i in np.where(flat_active & dones_f)[0]:
            if flat_env._envs[i].cards_in_shoe / 52.0 < reshuffle_threshold:
                flat_active[i] = False
            else:
                o, m, inf = flat_env.reset_at(i)
                obs_f[i], mask_f[i], info_f[i] = o, m, inf

    return joint_rewards, flat_rewards, tc_bet_pairs


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: BlackjackNet,
    cfg: dict,
    device: torch.device,
    n_shoes: int,
    base_seed: int,
    num_envs: int = 1024,
) -> dict:
    """Run paired evaluation: joint policy vs flat-bet variant on same seeds.

    Shoes are processed in rounds of up to ``num_envs`` parallel envs to keep
    memory bounded while amortising GPU launches over a large batch.
    """
    net.set_deterministic(True)

    joint_shoe_rewards: list[float] = []
    flat_shoe_rewards: list[float] = []
    all_tc_bet_pairs: list[tuple[float, int]] = []

    n_done = 0
    while n_done < n_shoes:
        n_in_round = min(num_envs, n_shoes - n_done)
        seeds = [base_seed + n_done + i for i in range(n_in_round)]
        jr, fr, tcb = play_round(net, cfg, device, seeds)
        joint_shoe_rewards.extend(jr.tolist())
        flat_shoe_rewards.extend(fr.tolist())
        all_tc_bet_pairs.extend(tcb)
        n_done += n_in_round
        print(f"  ... {n_done:,}/{n_shoes:,} shoes done")

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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print("  Using config from checkpoint.")

    agent = DQNAgent(
        net_config=net_config(cfg),
        train_config=train_config(cfg),
        replay_capacity=cfg.get("replay_buffer_size", 1_000_000),
        device=device,
    )
    agent.load_state_dict(ckpt["agent"])
    print("  Loaded agent weights from checkpoint.")

    agent.online_net.eval()
    agent.online_net.set_deterministic(True)

    # Estimate shoes needed for the requested number of hands.
    # A 6-deck shoe at 75% penetration yields roughly 60-80 hands.
    hands_per_shoe = 70
    n_shoes = max(args.eval_hands // hands_per_shoe, 100)

    print(f"Evaluating over {n_shoes:,} shoes (~{n_shoes * hands_per_shoe:,} hands), "
          f"{args.num_envs} parallel envs per round...")
    results = evaluate(
        agent.online_net, cfg, device, n_shoes,
        base_seed=args.seed, num_envs=args.num_envs,
    )
    print_report(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate joint playing + bet-sizing policy")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a Milestone 4 checkpoint .pt file")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--eval-hands", type=int, default=1_000_000,
                   help="Approximate number of hands to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=1024,
                   help="Parallel envs per round (larger = better GPU utilisation)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
