"""Evaluate a count-aware playing policy.

  1. EV per hand under flat 1x bets vs basic-strategy EV, averaged over 640
     environment seeds (10 batches × 64 envs).  Count features are left intact
     so learned deviations can improve over basic strategy.
  2. Action agreement against (basic strategy + Illustrious-18 deviations).
     Non-deviation cells are tested once at TC=0; deviation cells are tested
     at TCs that straddle each deviation threshold.

Usage:
  python eval/compare_with_count.py --checkpoint outputs/checkpoints/play_with_count/final.pt
  python eval/compare_with_count.py --checkpoint ... --eval-hands 1000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.dqn import DQNAgent
from agent.network import BlackjackNet
from env.vec_env import VecBlackjackEnv
from env.encoding import encode_state
from train.train_curriculum import net_config, train_config

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

HIT    = 0
STAND  = 1
DOUBLE = 2
SPLIT  = 3

ACTION_NAMES = {HIT: "H", STAND: "S", DOUBLE: "D", SPLIT: "P"}


# ---------------------------------------------------------------------------
# Basic strategy oracle (S17, DAS — same table as compare_basic_strategy.py)
# ---------------------------------------------------------------------------

def basic_strategy_action(
    player_sum: int,
    usable_ace: bool,
    dealer_upcard_value: int,
    can_double: bool,
    can_split: bool,
    pair_value: int | None,
) -> int:
    """Return the standard basic-strategy action for S17+DAS (no surrender)."""
    d = dealer_upcard_value

    if can_split and pair_value is not None:
        pv = pair_value
        if pv == 1:   return SPLIT
        if pv == 10:  return STAND
        if pv == 9:   return SPLIT if d not in (7, 10, 1) else STAND
        if pv == 8:   return SPLIT
        if pv == 7:   return SPLIT if d <= 7 else HIT
        if pv == 6:   return SPLIT if 2 <= d <= 6 else HIT
        if pv == 4:   return SPLIT if d in (5, 6) else HIT
        if pv == 3:   return SPLIT if 2 <= d <= 7 else HIT
        if pv == 2:   return SPLIT if 2 <= d <= 7 else HIT
        

    if usable_ace:
        s = player_sum
        if s == 20:  return STAND
        if s == 19:
            if d == 6:
                return DOUBLE if can_double else STAND
            else:
                return STAND
        if s == 18:
            if d in (7, 8):  return STAND
            if 2 <= d <= 6:     return DOUBLE if can_double else STAND
            return HIT
        if s == 17:  return DOUBLE if (3 <= d <= 6 and can_double) else HIT
        if s in (15, 16): return DOUBLE if (4 <= d <= 6 and can_double) else HIT
        if s in (13, 14): return DOUBLE if (5 <= d <= 6 and can_double) else HIT
        return HIT

    h = player_sum
    if h >= 17:  return STAND
    if h >= 13:  return STAND if 2 <= d <= 6 else HIT
    if h == 12:  return STAND if 4 <= d <= 6 else HIT
    if h == 11:  return DOUBLE if can_double else HIT
    if h == 10:  return DOUBLE if (can_double and d not in (10, 1)) else HIT
    if h == 9:   return DOUBLE if (can_double and 3 <= d <= 6) else HIT
    return HIT


def _dealer_upcard_val(rank: int) -> int:
    if rank == 1:    return 1
    if rank >= 10:   return 10
    return rank


# ---------------------------------------------------------------------------
# Hi-Lo deviations (Illustrious 18 subset — hard hands only, no surrender)
#
# direction "gte": deviate when TC >= threshold
# direction "lt":  deviate when TC <  threshold
# ---------------------------------------------------------------------------

DEVIATIONS: dict[tuple[int, bool, int], list[tuple[str, int, int, bool]]] = {
    # Stand-instead-of-hit deviations
    (16, False, 10): [("gte", 0, STAND, False,)],
    (15, False, 10): [("gte", 4, STAND, False)],
    (12, False, 3):  [("gte", 2, STAND, False)],
    (12, False, 2):  [("gte", 3, STAND, False)],
    (16, False, 9): [("gte", 5, STAND, False)],
    # Hit-instead-of-stand deviations (negative count)
    (13, False, 2):  [("lt", -1, HIT, False)],
    (12, False, 4):  [("lt", 0,  HIT, False)],
    (12, False, 5):  [("lt", -2,  HIT, False)],
    (12, False, 6):  [("lt", -1,  HIT, False)],
    (13, False, 3):  [("lt", -2,  HIT, False)],
    # Double-instead-of-hit deviations (require can_double)
    (10, False, 10): [("gte", 4, DOUBLE, True)],
    (11, False, 1):  [("gte", 1, DOUBLE, True)],
    (9,  False, 2):  [("gte", 1, DOUBLE, True)],
    (10, False, 1):  [("gte", 4, DOUBLE, True)],
    (9, False, 7): [("gte", 3, DOUBLE, True)],
}


def deviation_action(
    player_sum: int,
    usable_ace: bool,
    dealer_val: int,
    true_count: float,
    can_double: bool,
) -> int | None:
    """Return the deviation action if one applies at this (cell, TC), else None."""
    rules = DEVIATIONS.get((player_sum, usable_ace, dealer_val))
    if rules is None:
        return None
    for direction, threshold, dev_act, requires_double in rules:
        if requires_double and not can_double:
            continue
        fires = (
            (direction == "gte" and true_count >= threshold) or
            (direction == "lt"  and true_count <  threshold)
        )
        if fires:
            return dev_act
    return None


def expected_action(
    player_sum: int,
    usable_ace: bool,
    dealer_val: int,
    can_double: bool,
    can_split: bool,
    pair_value: int | None,
    true_count: float,
) -> tuple[int, bool]:
    """Return (expected_action, is_deviation).

    is_deviation is True when a deviation rule fires for this (cell, TC).
    """
    dev = deviation_action(player_sum, usable_ace, dealer_val, true_count, can_double)
    if dev is not None:
        return dev, True
    bs = basic_strategy_action(
        player_sum=player_sum,
        usable_ace=usable_ace,
        dealer_upcard_value=dealer_val,
        can_double=can_double,
        can_split=can_split,
        pair_value=pair_value,
    )
    return bs, False


# ---------------------------------------------------------------------------
# EV evaluation (flat 1x bets, 640 seeds)
# ---------------------------------------------------------------------------

def evaluate_ev(
    policy_fn,
    cfg: dict,
    n_hands: int,
    seed: int = 0,
    num_envs: int = 64,
    n_seed_batches: int = 10,
) -> tuple[float, float]:
    """Run policy_fn; return (ev, stderr).

    Runs n_seed_batches rounds of num_envs envs (default 10×64 = 640 unique
    seeds).  n_hands are distributed evenly across batches.
    Bet-phase actions are forced to 0 (flat 1x).
    """
    all_rewards: list[float] = []
    hands_per_batch = n_hands // n_seed_batches

    for batch in range(n_seed_batches):
        batch_seeds = [seed + batch * num_envs + i for i in range(num_envs)]
        vec_env = VecBlackjackEnv(num_envs, cfg, seeds=batch_seeds)
        obs, masks, infos = vec_env.reset()

        batch_rewards: list[float] = []
        while len(batch_rewards) < hands_per_batch:
            actions = np.zeros(num_envs, dtype=np.int32)
            play_idx = np.array(
                [i for i in range(num_envs) if infos[i]["phase"] != "bet"]
            )
            if len(play_idx) > 0:
                actions[play_idx] = policy_fn(obs[play_idx], masks[play_idx])

            obs, masks, rews, dones, infos = vec_env.step(actions)

            for i in range(num_envs):
                if dones[i]:
                    batch_rewards.append(float(rews[i]))
                    if len(batch_rewards) >= hands_per_batch:
                        break
                    obs_i, mask_i, info_i = vec_env.reset_at(i)
                    obs[i] = obs_i
                    masks[i] = mask_i
                    infos[i] = info_i

        all_rewards.extend(batch_rewards)

    arr = np.array(all_rewards, dtype=np.float64)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))


_UPCARD_IDX_TO_VAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _decode_obs(obs: np.ndarray, mask: np.ndarray):
    player_sum  = round(float(obs[0]) * 17.0 + 4.0)
    usable_ace  = bool(obs[1] > 0.5)
    upcard_idx  = int(np.argmax(obs[2:12]))
    dealer_val  = _UPCARD_IDX_TO_VAL[upcard_idx]
    is_pair     = bool(obs[12] > 0.5)
    pair_val    = None
    if is_pair:
        pair_idx = int(np.argmax(obs[13:23]))
        pair_val = _UPCARD_IDX_TO_VAL[pair_idx]
    can_double  = bool(mask[2])
    can_split   = bool(mask[3])
    return player_sum, usable_ace, dealer_val, can_double, can_split, pair_val


def make_bs_policy():
    """Batched basic-strategy policy (count-agnostic; used as the EV reference)."""
    def policy_fn(obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        k = len(obs_batch)
        actions = np.empty(k, dtype=np.int32)
        for i in range(k):
            ps, ua, dv, cd, cs, pv = _decode_obs(obs_batch[i], mask_batch[i])
            a = basic_strategy_action(
                player_sum=ps, usable_ace=ua, dealer_upcard_value=dv,
                can_double=cd, can_split=cs, pair_value=pv,
            )
            if not mask_batch[i][a]:
                a = int(np.argmax(mask_batch[i].astype(np.float32)))
            actions[i] = a
        return actions
    return policy_fn


def make_agent_policy(net: BlackjackNet, device: torch.device):
    """Batched agent policy (count feature left intact)."""
    def policy_fn(obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        obs_t  = torch.tensor(obs_batch,  dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_batch, dtype=torch.bool,    device=device)
        with torch.no_grad():
            play_q = net(obs_t).clone()
            play_q[~mask_t] = -1e9
        return play_q.argmax(dim=1).cpu().numpy().astype(np.int32)
    return policy_fn


# ---------------------------------------------------------------------------
# Count-aware action agreement
# ---------------------------------------------------------------------------

_NON_DEV_TC = 0.0


def _deviation_test_tcs(cell_key: tuple) -> list[float]:
    """Return TCs straddling each deviation threshold for this cell.

    For each rule with threshold T: adds T-1 (deviation off) and T (on for
    gte, off for lt — but both sides are always covered).
    """
    rules = DEVIATIONS[cell_key]
    tcs: set[float] = set()
    for _, threshold, _, _ in rules:
        tcs.add(float(threshold - 1))
        tcs.add(float(threshold))
    return sorted(tcs)


def evaluate_count_aware_agreement(
    net: BlackjackNet,
    device: torch.device,
    decks_remaining: float = 3.0,
) -> dict:
    """Enumerate cells and check agent vs (basic strategy + Illustrious-18 deviations).

    Non-deviation cells are evaluated once at TC=0.
    Deviation cells are evaluated at TCs that straddle each deviation threshold.
    """
    net.set_deterministic(True)

    dealer_upcards = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
        (7, 7), (8, 8), (9, 9), (10, 10),
    ]
    hard_totals = list(range(4, 22))
    soft_totals = list(range(12, 21))

    n_total       = 0
    n_match       = 0
    n_bs_total    = 0
    n_bs_match    = 0
    n_dev_total   = 0
    n_dev_match   = 0
    mismatches: list[dict] = []
    dev_results: dict[tuple[int, bool, int, int], dict] = {}

    def _query(obs_np: np.ndarray, mask_np: np.ndarray) -> int:
        obs_t  = torch.tensor(obs_np[None],  dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_np[None], dtype=torch.bool,    device=device)
        with torch.no_grad():
            play_q = net(obs_t).clone()
            play_q[~mask_t] = -1e9
        return int(play_q.argmax(dim=1).item())

    for dealer_rank, dealer_val in dealer_upcards:
        for usable_ace in (False, True):
            totals = soft_totals if usable_ace else hard_totals
            for player_sum in totals:
                can_double  = True
                can_split   = False
                cell_key    = (player_sum, usable_ace, dealer_val)
                is_dev_cell = cell_key in DEVIATIONS

                test_tcs = _deviation_test_tcs(cell_key) if is_dev_cell else [_NON_DEV_TC]

                for tc in test_tcs:
                    obs = encode_state(
                        player_sum=player_sum,
                        usable_ace=usable_ace,
                        dealer_upcard_rank=dealer_rank,
                        is_pair=False,
                        pair_rank=None,
                        can_double=can_double,
                        can_split=can_split,
                        true_count=tc,
                        decks_remaining=decks_remaining,
                    )
                    mask = np.array([True, True, can_double, can_split], dtype=bool)

                    agent_act = _query(obs, mask)

                    exp_act, is_dev = expected_action(
                        player_sum=player_sum,
                        usable_ace=usable_ace,
                        dealer_val=dealer_val,
                        can_double=can_double,
                        can_split=can_split,
                        pair_value=None,
                        true_count=tc,
                    )

                    match = (agent_act == exp_act)
                    n_total += 1
                    n_match += int(match)

                    if is_dev:
                        n_dev_total += 1
                        n_dev_match += int(match)
                        key = (player_sum, usable_ace, dealer_val, exp_act)
                        rec = dev_results.setdefault(key, {"n": 0, "ok": 0, "tcs": []})
                        rec["n"]  += 1
                        rec["ok"] += int(match)
                        rec["tcs"].append((tc, agent_act, match))
                    else:
                        n_bs_total += 1
                        n_bs_match += int(match)

                    if not match:
                        mismatches.append({
                            "player_sum": player_sum,
                            "usable_ace": usable_ace,
                            "dealer_val": dealer_val,
                            "true_count": tc,
                            "agent": agent_act,
                            "expected": exp_act,
                            "is_deviation": is_dev,
                        })

    net.set_deterministic(False)

    return {
        "n_total":      n_total,
        "n_match":      n_match,
        "agreement":    n_match / max(n_total, 1),
        "n_bs_total":   n_bs_total,
        "n_bs_match":   n_bs_match,
        "bs_agreement": n_bs_match / max(n_bs_total, 1),
        "n_dev_total":  n_dev_total,
        "n_dev_match":  n_dev_match,
        "dev_agreement": n_dev_match / max(n_dev_total, 1),
        "mismatches":   mismatches,
        "dev_results":  dev_results,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    ev: float,
    ev_stderr: float,
    bs_ev: float,
    bs_ev_stderr: float,
    agreement: dict,
) -> None:
    print("\n" + "=" * 70)
    print("Count-Aware Play Evaluation")
    print("=" * 70)

    # --- EV ---
    gap = bs_ev - ev  # positive = agent below BS; negative = agent beats BS
    if gap > 0:
        ev_label = f"{gap*100:.4f}% below BS"
        pass_ev  = gap <= 0.002
        ev_verdict = f"{'PASS' if pass_ev else 'FAIL'}, threshold: within 0.20%"
    else:
        ev_label   = f"{-gap*100:.4f}% above BS"
        pass_ev    = True
        ev_verdict = "PASS"
    print(f"  Agent EV:          {ev*100:+.4f}% +/- {ev_stderr*100:.4f}%")
    print(f"  Basic-strategy EV: {bs_ev*100:+.4f}% +/- {bs_ev_stderr*100:.4f}%")
    print(f"  Agent vs BS:       {ev_label}  ({ev_verdict})")

    # --- Agreement ---
    print(f"\n  Action agreement"
          f"  (non-deviation cells @ TC=0;"
          f" deviation cells @ threshold-straddling TCs):")
    print(f"  Overall:           {agreement['agreement']*100:5.1f}%"
          f"   ({agreement['n_match']}/{agreement['n_total']})")
    print(f"  Basic-strategy:    {agreement['bs_agreement']*100:5.1f}%"
          f"   ({agreement['n_bs_match']}/{agreement['n_bs_total']})"
          f"  (target: >= 95%)")
    print(f"  Deviation cells:   {agreement['dev_agreement']*100:5.1f}%"
          f"   ({agreement['n_dev_match']}/{agreement['n_dev_total']})")

    # --- Per-deviation breakdown ---
    print(f"\n  Per-deviation results:")
    print(f"    {'cell':18s}  {'dev':4s}  TCs (agent / result)")
    for (psum, ua, dv, dev_act), rec in sorted(agreement["dev_results"].items()):
        cell    = f"{'soft' if ua else 'hard'} {psum} vs {dv}"
        per_tc  = "  ".join(
            f"{tc:+g}:{ACTION_NAMES[a]}{'OK' if ok else 'X'}"
            for tc, a, ok in rec["tcs"]
        )
        status  = "PASS" if rec["ok"] == rec["n"] else "FAIL"
        print(f"    [{status}] {cell:18s}  {ACTION_NAMES[dev_act]:4s}  {per_tc}")

    # --- Basic-strategy mismatch listing ---
    bs_mismatches = [m for m in agreement["mismatches"] if not m["is_deviation"]]
    if bs_mismatches:
        print(f"\n  Basic-strategy mismatches ({len(bs_mismatches)}):")
        for m in sorted(bs_mismatches,
                        key=lambda x: (x["usable_ace"], x["player_sum"],
                                       x["dealer_val"], x["true_count"])):
            hand_type = "soft" if m["usable_ace"] else "hard"
            print(f"    {hand_type:4} {m['player_sum']:2} vs {m['dealer_val']:2}"
                  f"  agent={ACTION_NAMES[m['agent']]}"
                  f"  expected={ACTION_NAMES[m['expected']]}")

    # --- Summary ---
    pass_bs     = agreement["bs_agreement"] >= 0.95
    n_dev_pass  = sum(1 for r in agreement["dev_results"].values()
                      if r["ok"] == r["n"])
    n_dev_cells = len(agreement["dev_results"])
    pass_dev    = n_dev_pass >= 8

    print("\n" + "-" * 70)
    print(f"  EV >= BS - 0.2%:         {'PASS' if pass_ev else 'FAIL'}")
    print(f"  BS agreement >= 95%:     {'PASS' if pass_bs else 'FAIL'}")
    print(f"  Deviations passed:       {n_dev_pass}/{n_dev_cells}"
          f"  ({'PASS' if pass_dev else 'FAIL'}, target: >= 8)")
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
    print(cfg)

    agent = DQNAgent(
        net_config=net_config(cfg),
        train_config=train_config(cfg),
        replay_capacity=cfg.get("replay_buffer_size", 1_000_000),
        device=device,
    )
    agent.load_state_dict(ckpt["agent"])
    agent.online_net.eval()
    agent.online_net.set_deterministic(True)
    print("  Loaded agent weights from checkpoint.")

    agent_policy = make_agent_policy(agent.online_net, device)
    bs_policy    = make_bs_policy()

    n_batches    = 10
    total_seeds  = n_batches * 64
    print(f"Evaluating agent EV over {args.eval_hands:,} hands"
          f" ({n_batches} batches × 64 envs = {total_seeds} seeds)...")
    ev, ev_stderr = evaluate_ev(
        agent_policy, cfg, args.eval_hands,
        seed=args.seed, n_seed_batches=n_batches,
    )

    print(f"Evaluating basic-strategy EV over {args.eval_hands:,} hands...")
    bs_ev, bs_ev_stderr = evaluate_ev(
        bs_policy, cfg, args.eval_hands,
        seed=args.seed, n_seed_batches=n_batches,
    )

    print("Checking count-aware action agreement...")
    agreement = evaluate_count_aware_agreement(agent.online_net, device)

    print_report(ev, ev_stderr, bs_ev, bs_ev_stderr, agreement)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate count-aware play (basic strategy + deviations)"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--eval-hands", type=int, default=1_000_000)
    p.add_argument("--seed",       type=int, default=99999)
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
