"""Evaluate a count-aware playing policy.

Combines `compare_basic_strategy.py` and `check_deviations.py`:
  1. EV per hand under flat 1x bets vs basic-strategy EV (M2 acceptance test).
  2. Action agreement against a *count-aware* oracle: for each
     (player_sum, usable_ace, dealer_upcard) cell evaluated at several true
     counts, the expected action is the Hi-Lo deviation when one applies and
     the basic-strategy action otherwise.

This is the right correctness test for a Milestone 3 (count-enabled)
checkpoint: the agent must replicate basic strategy *and* execute the common
Hi-Lo deviations as the count moves.

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
        if pv == 8:   return SPLIT
        if pv == 9:   return SPLIT if d not in (7, 10, 1) else STAND
        if pv == 7:   return SPLIT if d <= 7 else HIT
        if pv == 6:   return SPLIT if 2 <= d <= 6 else HIT
        if pv == 4:   return SPLIT if d in (5, 6) else HIT
        if pv == 3:   return SPLIT if 2 <= d <= 7 else HIT
        if pv == 2:   return SPLIT if 2 <= d <= 7 else HIT
        if pv == 10:  return STAND

    if usable_ace:
        s = player_sum
        if s >= 19:  return STAND
        if s == 18:
            if d in (2, 7, 8):  return STAND
            if 3 <= d <= 6:     return DOUBLE if can_double else STAND
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
# Each rule fires when the true count is past the threshold in the given
# direction.  The deviation overrides basic strategy at that cell and TC.
# Format: (player_sum, usable_ace, dealer_val) -> list of (direction, threshold,
#         dev_action, requires_double).
# ---------------------------------------------------------------------------

# direction: "gte" -> deviate when TC >= threshold
#            "lt"  -> deviate when TC <  threshold
DEVIATIONS: dict[tuple[int, bool, int], list[tuple[str, int, int, bool]]] = {
    # Stand-instead-of-hit deviations
    (16, False, 10): [("gte", 0, STAND, False)],
    (15, False, 10): [("gte", 4, STAND, False)],
    (12, False, 3):  [("gte", 2, STAND, False)],
    (12, False, 2):  [("gte", 3, STAND, False)],
    # Double-instead-of-hit deviations (require can_double)
    (11, False, 1):  [("gte", 1, DOUBLE, True)],
    (9,  False, 2):  [("gte", 1, DOUBLE, True)],
    (10, False, 10): [("gte", 4, DOUBLE, True)],
    (10, False, 1):  [("gte", 4, DOUBLE, True)],
    # Hit-instead-of-stand deviations (negative count)
    (12, False, 4):  [("lt", 0,  HIT, False)],
    (13, False, 2):  [("lt", -1, HIT, False)],
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
    """Return (expected_action, is_deviation_cell).

    is_deviation_cell is True when a deviation rule fires for this (cell, TC).
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
# EV evaluation (flat 1x bets, count enabled)
# ---------------------------------------------------------------------------

def evaluate_ev(
    policy_fn,
    cfg: dict,
    n_hands: int,
    seed: int = 0,
    num_envs: int = 64,
) -> tuple[float, float]:
    """Run policy_fn for n_hands hands; return (ev, stderr).

    policy_fn receives batched (obs (K,28), mask (K,4)) and returns (K,) int
    actions.  Bet-phase actions are forced to 0 (flat 1x).
    """
    vec_env = VecBlackjackEnv(
        num_envs, cfg, seeds=[seed + i for i in range(num_envs)]
    )
    obs, masks, infos = vec_env.reset()

    rewards = []
    while len(rewards) < n_hands:
        actions = np.zeros(num_envs, dtype=np.int32)
        play_idx = np.array(
            [i for i in range(num_envs) if infos[i]["phase"] != "bet"]
        )
        if len(play_idx) > 0:
            actions[play_idx] = policy_fn(obs[play_idx], masks[play_idx])

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

    arr = np.array(rewards[:n_hands], dtype=np.float64)
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
    """Batched basic-strategy policy (used as the EV reference)."""
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
            play_q, _ = net(obs_t)
            play_q = play_q.clone()
            play_q[~mask_t] = -1e9
        return play_q.argmax(dim=1).cpu().numpy().astype(np.int32)
    return policy_fn


# ---------------------------------------------------------------------------
# Count-aware action agreement
# ---------------------------------------------------------------------------

# True counts at which each cell is evaluated.  Chosen to land clearly on
# both sides of every deviation threshold (-1, 0, 2, 3, 4) so the expected
# action is unambiguous everywhere.
TEST_TRUE_COUNTS = (-3.0, -1.0, 0.0, 2.0, 4.0)


def evaluate_count_aware_agreement(
    net: BlackjackNet,
    device: torch.device,
    test_tcs: tuple[float, ...] = TEST_TRUE_COUNTS,
    decks_remaining: float = 3.0,
) -> dict:
    """Enumerate (cell × TC) and check agent vs count-aware oracle.

    Returns a dict with totals, per-bucket counts (BS cells vs deviation cells),
    a list of mismatches, and a per-deviation pass/fail summary.
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
    # Per-deviation results: keyed by (player_sum, usable_ace, dealer_val, dev_act)
    dev_results: dict[tuple[int, bool, int, int], dict] = {}

    bet_multiplier = 1.0

    def _query(obs_np: np.ndarray, mask_np: np.ndarray) -> int:
        obs_t  = torch.tensor(obs_np[None],  dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_np[None], dtype=torch.bool,    device=device)
        with torch.no_grad():
            play_q, _ = net(obs_t)
            play_q = play_q.clone()
            play_q[~mask_t] = -1e9
        return int(play_q.argmax(dim=1).item())

    for tc in test_tcs:
        for dealer_rank, dealer_val in dealer_upcards:
            for usable_ace in (False, True):
                totals = soft_totals if usable_ace else hard_totals
                for player_sum in totals:
                    can_double = True
                    can_split  = False

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
                        bet_multiplier=bet_multiplier,
                        bet_phase=False,
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
        "n_total": n_total,
        "n_match": n_match,
        "agreement": n_match / max(n_total, 1),
        "n_bs_total": n_bs_total,
        "n_bs_match": n_bs_match,
        "bs_agreement": n_bs_match / max(n_bs_total, 1),
        "n_dev_total": n_dev_total,
        "n_dev_match": n_dev_match,
        "dev_agreement": n_dev_match / max(n_dev_total, 1),
        "mismatches": mismatches,
        "dev_results": dev_results,
        "test_tcs": test_tcs,
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
    diff = abs(ev - bs_ev)
    pass_ev = diff <= 0.002
    print(f"  EV per hand:       {ev*100:+.4f}% +/- {ev_stderr*100:.4f}%")
    print(f"  Basic-strategy EV: {bs_ev*100:+.4f}% +/- {bs_ev_stderr*100:.4f}%")
    print(f"  |Delta EV|:        {diff*100:.4f}%   "
          f"({'PASS' if pass_ev else 'FAIL'}, target: <= 0.20%)")

    # --- Agreement ---
    tcs_str = ", ".join(f"{tc:+g}" for tc in agreement["test_tcs"])
    print(f"\n  Action agreement evaluated at TCs: [{tcs_str}]")
    print(f"  Overall:           {agreement['agreement']*100:5.1f}%   "
          f"({agreement['n_match']}/{agreement['n_total']})")
    print(f"  Basic-strategy:    {agreement['bs_agreement']*100:5.1f}%   "
          f"({agreement['n_bs_match']}/{agreement['n_bs_total']})  "
          f"(target: >= 95%)")
    print(f"  Deviation cells:   {agreement['dev_agreement']*100:5.1f}%   "
          f"({agreement['n_dev_match']}/{agreement['n_dev_total']})")

    # --- Per-deviation breakdown ---
    print(f"\n  Per-deviation results:")
    print(f"    {'cell':16s}  {'dev':4s}  {'TCs (agent / pass)':40s}")
    for (psum, ua, dv, dev_act), rec in sorted(agreement["dev_results"].items()):
        cell = f"{'soft' if ua else 'hard'} {psum} vs {dv}"
        per_tc = "  ".join(
            f"{tc:+g}:{ACTION_NAMES[a]}{'OK' if ok else 'X'}"
            for tc, a, ok in rec["tcs"]
        )
        all_ok = rec["ok"] == rec["n"]
        status = "PASS" if all_ok else "FAIL"
        print(f"    [{status}] {cell:16s}  {ACTION_NAMES[dev_act]:4s}  {per_tc}")

    # --- Mismatch listing (basic-strategy cells only; deviation issues already
    #     surfaced above) ---
    bs_mismatches = [m for m in agreement["mismatches"] if not m["is_deviation"]]
    if bs_mismatches:
        print(f"\n  Basic-strategy mismatches ({len(bs_mismatches)}):")
        for m in sorted(bs_mismatches,
                        key=lambda x: (x["usable_ace"], x["player_sum"],
                                       x["dealer_val"], x["true_count"])):
            hand_type = "soft" if m["usable_ace"] else "hard"
            print(f"    {hand_type:4} {m['player_sum']:2} vs {m['dealer_val']:2} "
                  f"@ TC={m['true_count']:+g}: "
                  f"agent={ACTION_NAMES[m['agent']]}  "
                  f"expected={ACTION_NAMES[m['expected']]}")

    # --- Summary ---
    pass_bs    = agreement["bs_agreement"] >= 0.95
    n_dev_pass = sum(1 for r in agreement["dev_results"].values()
                     if r["ok"] == r["n"])
    n_dev_cells = len(agreement["dev_results"])
    pass_dev = n_dev_pass >= 8

    print("\n" + "-" * 70)
    print(f"  EV within 0.2%:          {'PASS' if pass_ev else 'FAIL'}")
    print(f"  BS agreement >= 95%:     {'PASS' if pass_bs else 'FAIL'}")
    print(f"  Deviations passed:       {n_dev_pass}/{n_dev_cells}  "
          f"({'PASS' if pass_dev else 'FAIL'}, target: >= 8)")
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

    print(f"Evaluating agent EV over {args.eval_hands:,} hands...")
    ev, ev_stderr = evaluate_ev(agent_policy, cfg, args.eval_hands, seed=args.seed)

    print(f"Evaluating basic-strategy EV over {args.eval_hands:,} hands...")
    bs_ev, bs_ev_stderr = evaluate_ev(bs_policy, cfg, args.eval_hands, seed=args.seed)

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
