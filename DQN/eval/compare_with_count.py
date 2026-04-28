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
from collections import Counter
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
# Basic strategy oracle (S17 — same table as compare_basic_strategy.py)
# ---------------------------------------------------------------------------

def basic_strategy_action(
    player_sum: int,
    usable_ace: bool,
    dealer_upcard_value: int,
    can_double: bool,
    can_split: bool,
    pair_value: int | None,
) -> int:
    """Return the standard basic-strategy action for S17 (no surrender)."""
    d = dealer_upcard_value

    if can_split and pair_value is not None:
        pv = pair_value
        if pv == 1:   return SPLIT
        if pv == 10:  return STAND
        if pv == 9:   return SPLIT if d not in (7, 10, 1) else STAND
        if pv == 8:   return SPLIT
        if pv == 7:   return SPLIT if d <= 7 else HIT
        if pv == 6:   return SPLIT if 3 <= d <= 6 else HIT
        if pv == 3:   return SPLIT if 4 <= d <= 7 else HIT
        if pv == 2:   return SPLIT if 4 <= d <= 7 else HIT
        

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
    if h == 11:  return DOUBLE if (can_double and d != 1) else HIT
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

DEVIATIONS: dict[tuple[int, bool, int], list[tuple[int, str, int, int, bool]]] = {
    # Stand-instead-of-hit deviations
    (16, False, 10): [(2, "gte", 0, STAND, False,)],
    (15, False, 10): [(3, "gte", 4, STAND, False)],
    (12, False, 3):  [(7, "gte", 2, STAND, False)],
    (12, False, 2):  [(8, "gte", 3, STAND, False)],
    (16, False, 9): [(13, "gte", 5, STAND, False)],
    # Hit-instead-of-stand deviations (negative count)
    (13, False, 2):  [(14, "lt", -1, HIT, False)],
    (12, False, 4):  [(15, "lt", 0,  HIT, False)],
    (12, False, 5):  [(16, "lt", -2,  HIT, False)],
    (12, False, 6):  [(17, "lt", -1,  HIT, False)],
    (13, False, 3):  [(18, "lt", -2,  HIT, False)],
    # Double-instead-of-hit deviations (require can_double)
    (10, False, 10): [(6, "gte", 4, DOUBLE, True)],
    (11, False, 1):  [(9, "gte", 1, DOUBLE, True)],
    (9,  False, 2):  [(10, "gte", 1, DOUBLE, True)],
    (10, False, 1): [(11, "gte", 4, DOUBLE, True)],
    (9, False, 7): [(12, "gte", 3, DOUBLE, True)],
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
        return 0, None
    for idx, direction, threshold, dev_act, requires_double in rules:
        if requires_double and not can_double:
            continue
        fires = (
            (direction == "gte" and true_count >= threshold) or
            (direction == "lt"  and true_count <  threshold)
        )
        if fires:
            return idx, dev_act
    return 0, None


def expected_action(
    player_sum: int,
    usable_ace: bool,
    dealer_val: int,
    can_double: bool,
    can_split: bool,
    pair_value: int | None,
    true_count: float,
) -> tuple[int, int, bool]:
    """Return (idx of deviation, expected_action, is_deviation).

    is_deviation is True when a deviation rule fires for this (cell, TC).
    """
    idx, dev = deviation_action(player_sum, usable_ace, dealer_val, true_count, can_double)
    if dev is not None:
        return idx, dev, True
    bs = basic_strategy_action(
        player_sum=player_sum,
        usable_ace=usable_ace,
        dealer_upcard_value=dealer_val,
        can_double=can_double,
        can_split=can_split,
        pair_value=pair_value,
    )
    return 0, bs, False


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


def make_oracle_policy():
    """Batched oracle policy using expected_action() (BS + Illustrious-18 deviations)."""
    def policy_fn(obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        k = len(obs_batch)
        actions = np.empty(k, dtype=np.int32)
        idxs = np.empty(k, dtype=np.int32)
        for i in range(k):
            ps, ua, dv, cd, cs, pv = _decode_obs(obs_batch[i], mask_batch[i])
            tc = float(obs_batch[i][25] * 5.0)
            idx, a, _ = expected_action(
                player_sum=ps, usable_ace=ua, dealer_val=dv,
                can_double=cd, can_split=cs, pair_value=pv,
                true_count=tc,
            )
            if not mask_batch[i][a]:
                a = int(np.argmax(mask_batch[i].astype(np.float32)))
            actions[i] = a
            idxs[i] = idx
        return actions
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
    for _, _, threshold, _, _ in rules:
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
    # key = (player_sum, usable_ace, dealer_val); tcs = (tc, agent, oracle, fires, match)
    dev_results: dict[tuple[int, bool, int], dict] = {}

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

                    idx, exp_act, is_dev = expected_action(
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

                    if is_dev_cell:
                        # All tests on deviation cells (firing and non-firing) go here;
                        # they are NOT counted in BS agreement to avoid false penalties.
                        n_dev_total += 1
                        n_dev_match += int(match)
                        rec = dev_results.setdefault(cell_key, {"n": 0, "ok": 0, "tcs": []})
                        rec["n"]  += 1
                        rec["ok"] += int(match)
                        rec["tcs"].append((tc, agent_act, exp_act, is_dev, match))
                    else:
                        # Pure BS cells only.
                        n_bs_total += 1
                        n_bs_match += int(match)

                    if not match:
                        mismatches.append({
                            "player_sum":        player_sum,
                            "usable_ace":        usable_ace,
                            "dealer_val":        dealer_val,
                            "true_count":        tc,
                            "agent":             agent_act,
                            "expected":          exp_act,
                            "is_deviation":      is_dev,
                            "is_deviation_idx":  idx,
                            "is_deviation_cell": is_dev_cell,
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
# Learned deviation analysis (Test 3)
# ---------------------------------------------------------------------------

_TC_SWEEP = list(range(-5, 6))  # integer TCs −5..+5


def count_sweep_bs_cells(
    net: BlackjackNet,
    device: torch.device,
    cells: set[tuple[int, bool, int]],
    decks_remaining: float = 3.0,
) -> dict[tuple[int, bool, int], dict[int, dict]]:
    """Query the agent at each integer TC in [−5, +5] for every cell.

    Uses can_double=True, can_split=False to match Test 2 conditions.
    Returns {cell: {tc: {"agent": int, "oracle": int, "match": bool}}}.
    """
    if not cells:
        return {}

    net.set_deterministic(True)
    mask = np.array([True, True, True, False], dtype=bool)
    result: dict = {}

    for (player_sum, usable_ace, dealer_val) in sorted(cells):
        cell_map: dict[int, dict] = {}
        for tc in _TC_SWEEP:
            obs = encode_state(
                player_sum=player_sum,
                usable_ace=usable_ace,
                dealer_upcard_rank=dealer_val,
                is_pair=False,
                pair_rank=None,
                can_double=True,
                can_split=False,
                true_count=float(tc),
                decks_remaining=decks_remaining,
            )
            obs_t  = torch.tensor(obs[None],  dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask[None], dtype=torch.bool,    device=device)
            with torch.no_grad():
                play_q = net(obs_t).clone()
                play_q[~mask_t] = -1e9
            agent_act = int(play_q.argmax(dim=1).item())

            _, oracle_act, _ = expected_action(
                player_sum=player_sum,
                usable_ace=usable_ace,
                dealer_val=dealer_val,
                can_double=True,
                can_split=False,
                pair_value=None,
                true_count=float(tc),
            )
            cell_map[tc] = {
                "agent":  agent_act,
                "oracle": oracle_act,
                "match":  agent_act == oracle_act,
            }
        result[(player_sum, usable_ace, dealer_val)] = cell_map

    net.set_deterministic(False)
    return result


def classify_cell_sensitivity(tc_map: dict[int, dict]) -> dict:
    """Classify whether the agent is count-sensitive for this cell.

    count-sensitive = agent takes at least 2 distinct actions across TCs.
    Deviant TCs are those where agent != oracle. Returns classification dict
    with tc_lo/tc_hi bounding the deviant range for use in simulation.
    """
    agent_actions = {tc: rec["agent"] for tc, rec in tc_map.items()}
    deviant_tcs   = sorted(tc for tc, rec in tc_map.items() if not rec["match"])
    is_count_sensitive = len(set(agent_actions.values())) >= 2

    if not is_count_sensitive or not deviant_tcs:
        return {
            "is_count_sensitive":           False,
            "deviant_tcs":                  deviant_tcs,
            "deviant_action":               None,
            "oracle_action_at_deviant_tcs": None,
            "tc_lo":                        None,
            "tc_hi":                        None,
        }

    deviant_action = Counter(
        tc_map[tc]["agent"] for tc in deviant_tcs
    ).most_common(1)[0][0]
    oracle_action_at_deviant_tcs = Counter(
        tc_map[tc]["oracle"] for tc in deviant_tcs
    ).most_common(1)[0][0]

    return {
        "is_count_sensitive":           True,
        "deviant_tcs":                  deviant_tcs,
        "deviant_action":               deviant_action,
        "oracle_action_at_deviant_tcs": oracle_action_at_deviant_tcs,
        "tc_lo":                        float(min(deviant_tcs)),
        "tc_hi":                        float(max(deviant_tcs)),
    }


def simulate_action_at_cell(
    target_cell: tuple[int, bool, int],
    forced_action: int,
    tc_lo: float,
    tc_hi: float,
    cfg: dict,
    n_hands: int,
    seed: int = 0,
    num_envs: int = 64,
    n_seed_batches: int = 10,
) -> dict:
    """Empirically estimate EV and win-rate of forced_action at target_cell when TC ∈ [tc_lo, tc_hi].

    Intercepts play-phase decisions at the target cell + TC and forces the
    specified action; uses oracle for all other decisions.  Tracks only hand
    rewards for hands that visited the target cell in the TC range.

    Called twice per count-sensitive cell — once with the agent's deviant action
    and once with the oracle's action — using the same seed for comparability.

    Note: episode reward covers all sub-hands from splits; if the target cell
    is visited during one sub-hand of a split, the full episode reward is
    attributed to that target encounter.
    """
    oracle_policy = make_oracle_policy()

    def _oracle_single(obs_i: np.ndarray, mask_i: np.ndarray) -> int:
        return int(oracle_policy(obs_i[None], mask_i[None])[0])

    target_ps, target_ua, target_dv = target_cell
    all_target_rewards: list[float] = []
    hands_per_batch = n_hands // n_seed_batches

    for batch in range(n_seed_batches):
        batch_seeds = [seed + batch * num_envs + i for i in range(num_envs)]
        vec_env = VecBlackjackEnv(num_envs, cfg, seeds=batch_seeds)
        obs, masks, infos = vec_env.reset()

        hit_target = [False] * num_envs
        hands_done = 0
        batch_target_rewards: list[float] = []

        while hands_done < hands_per_batch:
            actions = np.zeros(num_envs, dtype=np.int32)

            for i in range(num_envs):
                if infos[i]["phase"] == "bet":
                    actions[i] = 0
                else:
                    ps, ua, dv, *_ = _decode_obs(obs[i], masks[i])
                    tc = round(float(obs[i][25] * 5.0))  # round avoids fp jitter

                    if ps == target_ps and ua == target_ua and dv == target_dv \
                            and tc_lo <= tc <= tc_hi:
                        hit_target[i] = True
                        actions[i] = (forced_action if masks[i][forced_action]
                                      else _oracle_single(obs[i], masks[i]))
                    else:
                        actions[i] = _oracle_single(obs[i], masks[i])

            obs, masks, rews, dones, infos = vec_env.step(actions)

            for i in range(num_envs):
                if dones[i]:
                    if hit_target[i]:
                        batch_target_rewards.append(float(rews[i]))
                    hit_target[i] = False
                    hands_done += 1
                    if hands_done >= hands_per_batch:
                        break
                    obs_i, mask_i, info_i = vec_env.reset_at(i)
                    obs[i]   = obs_i
                    masks[i] = mask_i
                    infos[i] = info_i

        all_target_rewards.extend(batch_target_rewards)

    if not all_target_rewards:
        return {"ev": float("nan"), "ev_stderr": float("nan"),
                "win_rate": float("nan"), "n_samples": 0}

    arr = np.array(all_target_rewards, dtype=np.float64)
    return {
        "ev":        float(arr.mean()),
        "ev_stderr": float(arr.std() / np.sqrt(len(arr))),
        "win_rate":  float(np.mean(arr > 0)),
        "n_samples": len(arr),
    }


def evaluate_learned_deviations(
    net: BlackjackNet,
    device: torch.device,
    cfg: dict,
    bs_mismatches: list[dict],
    n_hands: int,
    seed: int = 0,
    n_seed_batches: int = 10,
) -> list[dict]:
    """Analyse BS-cell mismatches from Test 2 for count-sensitivity and EV.

    Steps:
      1. Sweep each unique mismatch cell at integer TCs −5..+5.
      2. Classify each cell as count-sensitive or count-insensitive.
      3. For count-sensitive cells: run two simulations (agent vs oracle action)
         in the deviant TC range and collect EV / win-rate.
    """
    if not bs_mismatches:
        return []

    unique_cells: set[tuple[int, bool, int]] = {
        (m["player_sum"], m["usable_ace"], m["dealer_val"])
        for m in bs_mismatches
    }

    tc_maps = count_sweep_bs_cells(net, device, unique_cells)

    analyses: list[dict] = []
    for cell in sorted(unique_cells):
        tc_map         = tc_maps[cell]
        classification = classify_cell_sensitivity(tc_map)
        record: dict   = {"cell": cell, "tc_map": tc_map, **classification}

        if not classification["is_count_sensitive"]:
            record["agent_action"]  = tc_map[0]["agent"]
            record["oracle_action"] = tc_map[0]["oracle"]
            analyses.append(record)
            continue

        dev_action    = classification["deviant_action"]
        oracle_action = classification["oracle_action_at_deviant_tcs"]
        tc_lo         = classification["tc_lo"]
        tc_hi         = classification["tc_hi"]
        ps, ua, dv    = cell
        hand_type     = "soft" if ua else "hard"
        print(f"  Simulating {hand_type} {ps} vs {dv}:"
              f" agent={ACTION_NAMES[dev_action]}"
              f" vs oracle={ACTION_NAMES[oracle_action]}"
              f" at TC in [{tc_lo:+.0f}, {tc_hi:+.0f}]...")

        record["agent_sim"] = simulate_action_at_cell(
            target_cell=cell, forced_action=dev_action,
            tc_lo=tc_lo, tc_hi=tc_hi,
            cfg=cfg, n_hands=n_hands, seed=seed,
            num_envs=64, n_seed_batches=n_seed_batches,
        )
        record["oracle_sim"] = simulate_action_at_cell(
            target_cell=cell, forced_action=oracle_action,
            tc_lo=tc_lo, tc_hi=tc_hi,
            cfg=cfg, n_hands=n_hands, seed=seed,
            num_envs=64, n_seed_batches=n_seed_batches,
        )
        analyses.append(record)

    return analyses


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

    # --- Evaluation 1: Basic-strategy agreement (pure BS cells, no deviation rules) ---
    print(f"\n  Evaluation 1: Basic-Strategy Agreement")
    print(f"  (cells with no deviation rule, evaluated at TC=0)")
    print(f"  Agreement:  {agreement['bs_agreement']*100:5.1f}%"
          f"   ({agreement['n_bs_match']}/{agreement['n_bs_total']})"
          f"  (target: >= 95%)")

    pure_bs_mismatches = [
        m for m in agreement["mismatches"]
        if not m["is_deviation"] and not m["is_deviation_cell"]
    ]
    if pure_bs_mismatches:
        print(f"  Mismatches ({len(pure_bs_mismatches)}):")
        for m in sorted(pure_bs_mismatches,
                        key=lambda x: (x["usable_ace"], x["player_sum"], x["dealer_val"])):
            hand_type = "soft" if m["usable_ace"] else "hard"
            print(f"    {hand_type} {m['player_sum']:2} vs {m['dealer_val']:2}"
                  f"  agent={ACTION_NAMES[m['agent']]}"
                  f"  expected={ACTION_NAMES[m['expected']]}")

    # --- Evaluation 2: Per-deviation following ---
    print(f"\n  Evaluation 2: Deviation Following")
    print(f"  (each deviation at threshold-straddling TCs; oracle=BS when deviation is off)")
    print(f"  Overall:  {agreement['dev_agreement']*100:5.1f}%"
          f"   ({agreement['n_dev_match']}/{agreement['n_dev_total']} tests)\n")

    for (ps, ua, dv), rec in sorted(agreement["dev_results"].items()):
        cell  = f"{'soft' if ua else 'hard'} {ps} vs {dv}"
        rules = DEVIATIONS.get((ps, ua, dv), [])
        rule_desc = "  ".join(
            f"TC{'≥' if d == 'gte' else '<'}{t:+d}→{ACTION_NAMES[a]}"
            + (" (D req.)" if req else "")
            for idx, d, t, a, req in rules
        )
        tc_results = "  ".join(
            f"TC={tc:+g}:{ACTION_NAMES[exp]}/{ACTION_NAMES[agt]}"
            f"({'on' if fires else 'off'}){'OK' if ok else 'FAIL'}"
            for tc, agt, exp, fires, ok in sorted(rec["tcs"], key=lambda x: x[0])
        )
        status = "PASS" if rec["ok"] == rec["n"] else "FAIL"
        print(f"    [{status}] {cell:18s}  {rule_desc:22s}  {tc_results}")

    # --- Summary ---
    pass_bs     = agreement["bs_agreement"] >= 0.95
    n_dev_pass  = sum(1 for r in agreement["dev_results"].values()
                      if r["ok"] == r["n"])
    n_dev_cells = len(agreement["dev_results"])
    pass_dev    = n_dev_pass >= n_dev_cells - 2  # allow up to 2 deviations failing

    print("\n" + "-" * 70)
    print(f"  EV >= BS - 0.2%:         {'PASS' if pass_ev else 'FAIL'}")
    print(f"  BS agreement >= 95%:     {'PASS' if pass_bs else 'FAIL'}")
    print(f"  Deviations passed:       {n_dev_pass}/{n_dev_cells}"
          f"  ({'PASS' if pass_dev else 'FAIL'}, target: >= {n_dev_cells - 2})")
    print("=" * 70 + "\n")


def print_learned_deviation_report(analyses: list[dict]) -> None:
    print("\nTest 3: Learned Deviation Analysis")
    print("=" * 70)

    if not analyses:
        print("  No BS mismatches — nothing to analyze.")
        print("=" * 70 + "\n")
        return

    count_insensitive = [a for a in analyses if not a["is_count_sensitive"]]
    count_sensitive   = [a for a in analyses if a["is_count_sensitive"]]
    print(f"  BS-cell mismatches from Test 2: {len(analyses)} unique cell(s)\n")

    if count_insensitive:
        print(f"  [Count-insensitive: {len(count_insensitive)} cell(s)]")
        for a in count_insensitive:
            ps, ua, dv = a["cell"]
            hand_type  = "soft" if ua else "hard"
            print(f"    {hand_type} {ps} vs {dv}"
                  f"  ->  agent: {ACTION_NAMES[a['agent_action']]}"
                  f"  oracle: {ACTION_NAMES[a['oracle_action']]}"
                  f"  (consistent across all TCs, likely error)")
        print()

    if count_sensitive:
        print(f"  [Count-sensitive: {len(count_sensitive)} cell(s)]")
        for a in count_sensitive:
            ps, ua, dv = a["cell"]
            hand_type  = "soft" if ua else "hard"
            tc_map     = a["tc_map"]

            print(f"\n  {hand_type} {ps} vs {dv}")
            print("  TC:    " + "  ".join(f"{tc:+3d}" for tc in _TC_SWEEP))
            print("  agent: " + "  ".join(
                f"  {ACTION_NAMES[tc_map[tc]['agent']]}" for tc in _TC_SWEEP
            ))
            print("  oracle:" + "  ".join(
                f"  {ACTION_NAMES[tc_map[tc]['oracle']]}" for tc in _TC_SWEEP
            ))

            dev_action    = a["deviant_action"]
            oracle_action = a["oracle_action_at_deviant_tcs"]
            tc_lo, tc_hi  = a["tc_lo"], a["tc_hi"]
            tc_range_str  = (f"TC = {tc_lo:+.0f}" if tc_lo == tc_hi
                             else f"TC in [{tc_lo:+.0f}, {tc_hi:+.0f}]")
            print(f"  Deviation: agent {ACTION_NAMES[dev_action]} at {tc_range_str};"
                  f" oracle {ACTION_NAMES[oracle_action]} in that range")

            agent_sim  = a["agent_sim"]
            oracle_sim = a["oracle_sim"]

            if agent_sim["n_samples"] == 0:
                print("  Simulation: cell never encountered"
                      " (TC range too extreme or cell extremely rare)")
            else:
                n = agent_sim["n_samples"]
                print(f"  Simulation ({tc_range_str}, {n:,} target hands):")
                print(f"    agent  {ACTION_NAMES[dev_action]}:"
                      f"  EV = {agent_sim['ev']*100:+.3f}%"
                      f" ± {agent_sim['ev_stderr']*100:.3f}%"
                      f"   win rate: {agent_sim['win_rate']*100:.1f}%")
                print(f"    oracle {ACTION_NAMES[oracle_action]}:"
                      f"  EV = {oracle_sim['ev']*100:+.3f}%"
                      f" ± {oracle_sim['ev_stderr']*100:.3f}%"
                      f"   win rate: {oracle_sim['win_rate']*100:.1f}%")
                delta_ev    = agent_sim["ev"] - oracle_sim["ev"]
                combined_se = np.sqrt(
                    agent_sim["ev_stderr"]**2 + oracle_sim["ev_stderr"]**2
                )
                print(f"    delta = {delta_ev*100:+.3f}% ± {combined_se*100:.3f}%")
                verdict = ("PLAUSIBLE LEARNED DEVIATION"
                           if delta_ev >= -2.0 * combined_se
                           else "LIKELY ERROR")
                print(f"    ->  {verdict}")

    print("\n" + "=" * 70 + "\n")


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
    agent.online_net.eval()
    agent.online_net.set_deterministic(True)
    print("  Loaded agent weights from checkpoint.")

    agent_policy = make_agent_policy(agent.online_net, device)
    bs_policy    = make_bs_policy()

    n_batches    = 10
    total_seeds  = n_batches * 64
    print(f"Evaluating agent EV over {args.eval_hands_bs:,} hands"
          f" ({n_batches} batches × 64 envs = {total_seeds} seeds)...")
    ev, ev_stderr = evaluate_ev(
        agent_policy, cfg, args.eval_hands_bs,
        seed=args.seed, n_seed_batches=n_batches,
    )

    print(f"Evaluating basic-strategy EV over {args.eval_hands_bs:,} hands...")
    bs_ev, bs_ev_stderr = evaluate_ev(
        bs_policy, cfg, args.eval_hands_bs,
        seed=args.seed, n_seed_batches=n_batches,
    )

    print("Checking count-aware action agreement...")
    agreement = evaluate_count_aware_agreement(agent.online_net, device)

    print_report(ev, ev_stderr, bs_ev, bs_ev_stderr, agreement)

    bs_mismatches = [
        m for m in agreement["mismatches"]
        if not m["is_deviation"] and not m["is_deviation_cell"]
    ]
    unique_bs_cells = {
        (m["player_sum"], m["usable_ace"], m["dealer_val"]) for m in bs_mismatches
    }
    print(f"Running learned deviation analysis over {args.eval_hands_dev} hands"
          f" ({len(unique_bs_cells)} unique BS-mismatch cell(s))...")
    analyses = evaluate_learned_deviations(
        net=agent.online_net,
        device=device,
        cfg=cfg,
        bs_mismatches=bs_mismatches,
        n_hands=args.eval_hands_dev,
        seed=args.seed,
        n_seed_batches=n_batches,
    )
    print_learned_deviation_report(analyses)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate count-aware play (basic strategy + deviations)"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--eval-hands-bs", type=int, default=10_000_000)
    p.add_argument("--eval-hands-dev", type=int, default=1_000_000)
    p.add_argument("--seed",       type=int, default=99999)
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
