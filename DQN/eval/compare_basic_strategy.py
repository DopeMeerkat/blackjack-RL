"""Evaluate a trained playing policy against the basic-strategy baseline.

Acceptance test for Milestone 2 (blackjack_rl_design.md §13):
  1. Evaluated EV is within 0.2% of published basic-strategy EV for the
     rule set (target: approximately -0.5% to -0.6%).
  2. Agent's argmax action agrees with the basic-strategy chart on at least
     95% of (player_total, dealer_upcard) cells.

Agent EV is measured under flat 1x bets so the comparison against basic
strategy is apples-to-apples.

Usage:
  python eval/compare_basic_strategy.py --checkpoint outputs/checkpoints/curriculum/final.pt
  python eval/compare_basic_strategy.py --checkpoint ... --use-count --eval-hands 1000000
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
# Basic strategy oracle (S17, DAS — same table as in tests/test_env.py)
# ---------------------------------------------------------------------------

HIT    = 0
STAND  = 1
DOUBLE = 2
SPLIT  = 3


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

    # Splits
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

    # Soft hands
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

    # Hard hands
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
# EV evaluation
# ---------------------------------------------------------------------------

def evaluate_ev(
    policy_fn,
    cfg: dict,
    n_hands: int,
    seed: int = 0,
    num_envs: int = 64,
) -> tuple[float, float]:
    """Run *policy_fn* for n_hands hands; return (ev, stderr).

    policy_fn receives batched inputs (obs_batch, mask_batch) with shapes
    (K, 28) and (K, 4) and must return a (K,) int action array.
    Bet-phase actions are always 0 (flat bet).

    Uses VecBlackjackEnv for batched stepping and policy inference.
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
    ev  = float(arr.mean())
    stderr = float(arr.std() / np.sqrt(len(arr)))
    return ev, stderr


# ---------------------------------------------------------------------------
# Action agreement
# ---------------------------------------------------------------------------

def evaluate_action_agreement(
    net: BlackjackNet,
    zero_count: bool,
    device: torch.device,
    cfg: dict,
) -> tuple[float, dict]:
    """Check agent action vs basic strategy over all BS chart cells.

    Enumerates all (player_sum, usable_ace, dealer_upcard_value) combinations,
    constructs a synthetic observation for each, and compares the agent's
    argmax to basic strategy.

    Returns:
        agreement_pct: fraction of cells where agent matches BS.
        detail:        dict mapping (sum, usable, upcard) → (agent, bs, match).
    """
    net.set_deterministic(True)
    detail = {}
    n_correct = 0
    n_total   = 0

    # Dealer upcard values (raw ranks for encoding + value for BS lookup)
    dealer_upcards = [
        (1, 1),   # Ace
        (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
        (7, 7), (8, 8), (9, 9), (10, 10),
    ]

    # Hard hand totals: 4 to 21
    hard_totals = list(range(4, 22))
    # Soft hand totals: 12 to 21 (Ace + 1 to Ace + 10 = 12 to 21)
    # Note: soft 21 is blackjack (never reached in play)
    soft_totals = list(range(12, 21))

    true_count = 0.0          # neutral count for comparison
    decks_remaining = 3.0     # mid-shoe

    for dealer_rank, dealer_val in dealer_upcards:
        # Hard hands (usable_ace=False)
        for player_sum in hard_totals:
            can_double = True
            can_split  = False
            obs = encode_state(
                player_sum=player_sum,
                usable_ace=False,
                dealer_upcard_rank=dealer_rank,
                is_pair=False,
                pair_rank=None,
                can_double=can_double,
                can_split=can_split,
                true_count=true_count,
                decks_remaining=decks_remaining,
            )
            if zero_count:
                obs[25] = 0.0

            obs_t  = torch.tensor(obs[None], dtype=torch.float32, device=device)
            mask_t = torch.ones(1, 4, dtype=torch.bool, device=device)
            mask_t[0, 2] = can_double
            mask_t[0, 3] = can_split

            with torch.no_grad():
                play_q = net(obs_t).clone()
                play_q[~mask_t] = -1e9
            agent_action = int(play_q.argmax(dim=1).item())

            bs_action = basic_strategy_action(
                player_sum=player_sum,
                usable_ace=False,
                dealer_upcard_value=dealer_val,
                can_double=can_double,
                can_split=False,
                pair_value=None,
            )
            # Clamp BS action to legal (can_double=True, no split)
            if bs_action == SPLIT:
                bs_action = HIT

            match = (agent_action == bs_action)
            detail[(player_sum, False, dealer_val)] = (agent_action, bs_action, match)
            n_correct += int(match)
            n_total   += 1

        # Soft hands (usable_ace=True)
        for player_sum in soft_totals:
            can_double = True
            obs = encode_state(
                player_sum=player_sum,
                usable_ace=True,
                dealer_upcard_rank=dealer_rank,
                is_pair=False,
                pair_rank=None,
                can_double=can_double,
                can_split=False,
                true_count=true_count,
                decks_remaining=decks_remaining,
            )
            if zero_count:
                obs[25] = 0.0

            obs_t  = torch.tensor(obs[None], dtype=torch.float32, device=device)
            mask_t = torch.ones(1, 4, dtype=torch.bool, device=device)
            mask_t[0, 2] = can_double
            mask_t[0, 3] = False   # no split

            with torch.no_grad():
                play_q = net(obs_t).clone()
                play_q[~mask_t] = -1e9
            agent_action = int(play_q.argmax(dim=1).item())

            bs_action = basic_strategy_action(
                player_sum=player_sum,
                usable_ace=True,
                dealer_upcard_value=dealer_val,
                can_double=can_double,
                can_split=False,
                pair_value=None,
            )
            if bs_action == SPLIT:
                bs_action = HIT

            match = (agent_action == bs_action)
            detail[(player_sum, True, dealer_val)] = (agent_action, bs_action, match)
            n_correct += int(match)
            n_total   += 1

    net.set_deterministic(False)
    agreement_pct = n_correct / n_total if n_total > 0 else 0.0
    return agreement_pct, detail


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

# Upcard one-hot index (0-9) → blackjack value used by basic_strategy_action
_UPCARD_IDX_TO_VAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _decode_obs(obs: np.ndarray, mask: np.ndarray):
    """Decode a play-phase observation + mask into BS lookup inputs."""
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
    """Return a batched policy_fn that follows basic strategy.

    Signature: policy_fn(obs_batch (K,28), mask_batch (K,4)) -> actions (K,) int32
    """
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


def make_agent_policy(
    net: BlackjackNet,
    zero_count: bool,
    device: torch.device,
):
    """Return a batched policy_fn that follows the agent's deterministic policy.

    Signature: policy_fn(obs_batch (K,28), mask_batch (K,4)) -> actions (K,) int32
    """
    def policy_fn(obs_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        ob = obs_batch
        if zero_count:
            ob = ob.copy()
            ob[:, 25] = 0.0
        obs_t  = torch.tensor(ob,         dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_batch,  dtype=torch.bool,    device=device)
        with torch.no_grad():
            play_q = net(obs_t).clone()
            play_q[~mask_t] = -1e9
        return play_q.argmax(dim=1).cpu().numpy().astype(np.int32)
    return policy_fn


ACTION_NAMES = {HIT: "H", STAND: "S", DOUBLE: "D", SPLIT: "P"}

def print_report(
    ev: float,
    ev_stderr: float,
    bs_ev: float,
    bs_ev_stderr: float,
    agreement_pct: float,
    detail: dict,
    zero_count: bool,
) -> None:
    print("\n" + "=" * 60)
    print("Milestone 2 Evaluation Report")
    print("=" * 60)
    print(f"  Count feature:    {'zeroed (M2)' if zero_count else 'enabled (M3)'}")
    print(f"  EV per hand:      {ev*100:+.4f}% ± {ev_stderr*100:.4f}%")

    diff    = abs(ev - bs_ev)
    pass_ev = diff <= 0.002   # within 0.2%
    print(f"  Basic-strategy EV: {bs_ev*100:+.4f}% ± {bs_ev_stderr*100:.4f}%  (target: within 0.2%)")
    print(f"  Δ EV:             {diff*100:.4f}%  →  {'PASS ✓' if pass_ev else 'FAIL ✗'}")

    pass_agree = agreement_pct >= 0.95
    print(f"  Action agreement: {agreement_pct*100:.1f}%  "
          f"→  {'PASS ✓' if pass_agree else 'FAIL ✗'} (target: ≥ 95%)")

    mismatches = [(k, v) for k, v in detail.items() if not v[2]]
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for (psum, ua, d), (agent_act, bs, _) in sorted(mismatches):
            hand_type = "soft" if ua else "hard"
            print(f"    {hand_type:4} {psum:2} vs dealer {d:2}:  "
                  f"agent={ACTION_NAMES[agent_act]}  bs={ACTION_NAMES[bs]}")
    else:
        print("\n  No mismatches — perfect agreement!")

    print("=" * 60)
    if pass_ev and pass_agree:
        print("  MILESTONE 2 ACCEPTANCE: PASS")
    else:
        print("  MILESTONE 2 ACCEPTANCE: FAIL")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )
    zero_count = not args.use_count

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

    agent_policy = make_agent_policy(agent.online_net, zero_count, device)
    bs_policy = make_bs_policy()

    print(f"Evaluating agent EV over {args.eval_hands:,} hands…")
    ev, ev_stderr = evaluate_ev(agent_policy, cfg, args.eval_hands, seed=args.seed)

    print(f"Evaluating basic-strategy EV over {args.eval_hands:,} hands…")
    bs_ev, bs_ev_stderr = evaluate_ev(bs_policy, cfg, args.eval_hands, seed=args.seed)

    print("Checking action agreement against basic strategy…")
    agreement_pct, detail = evaluate_action_agreement(
        agent.online_net, zero_count, device, cfg
    )

    print_report(ev, ev_stderr, bs_ev, bs_ev_stderr, agreement_pct, detail, zero_count)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare trained policy to basic strategy")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--use-count",  action="store_true",
                   help="Don't zero the count feature (Milestone 3 checkpoint)")
    p.add_argument("--eval-hands", type=int, default=1_000_000)
    p.add_argument("--seed",       type=int, default=99999)
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
