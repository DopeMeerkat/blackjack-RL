"""Check whether the agent plays common Hi-Lo deviations correctly.

Milestone 3 acceptance test (blackjack_rl_design.md §13):
  After training for 50M hands with the count feature enabled, the agent
  correctly plays at least 8 of the 10 most common Hi-Lo deviations from
  the Illustrious 18 list (excluding any that require surrender).

Insurance (TC >= +3) is hard-coded in the environment and not part of the
learned action space, so it is excluded from the deviation list.

Usage:
  python eval/check_deviations.py --checkpoint outputs/checkpoints/play_with_count/final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from agent.network import BlackjackNet
from env.encoding import encode_state

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

HIT = 0
STAND = 1
DOUBLE = 2
SPLIT = 3

_ACTION_NAMES = {HIT: "Hit", STAND: "Stand", DOUBLE: "Double", SPLIT: "Split"}

# ---------------------------------------------------------------------------
# Illustrious 18 deviations (excluding insurance and surrender)
#
# Each deviation specifies:
#   - A game situation (player hand, dealer upcard)
#   - The basic-strategy action (what you'd normally do)
#   - The deviation action (what you should do when the count is right)
#   - The true-count threshold at which the deviation kicks in
#   - Direction: "gte" means deviate when TC >= threshold,
#                "lt"  means deviate when TC < threshold
#
# We test each deviation at a TC that is clearly past the threshold
# (threshold + 2 for "gte", threshold - 2 for "lt") and verify the
# agent selects the deviation action.
#
# The 10 most common deviations (by frequency and EV impact):
# ---------------------------------------------------------------------------

Deviation = dict  # type alias for readability


def _make_deviations() -> list[Deviation]:
    """Return the 10 most impactful Illustrious 18 deviations.

    Ordered roughly by EV impact.  Insurance and surrender are excluded
    per the design doc.
    """
    return [
        # 1. 16 vs 10: normally Hit, Stand when TC >= 0
        {
            "name": "16 vs 10",
            "player_sum": 16,
            "usable_ace": False,
            "dealer_rank": 10,
            "is_pair": False,
            "pair_rank": None,
            "can_double": False,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": STAND,
            "tc_threshold": 0,
            "direction": "gte",
        },
        # 2. 15 vs 10: normally Hit, Stand when TC >= +4
        {
            "name": "15 vs 10",
            "player_sum": 15,
            "usable_ace": False,
            "dealer_rank": 10,
            "is_pair": False,
            "pair_rank": None,
            "can_double": False,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": STAND,
            "tc_threshold": 4,
            "direction": "gte",
        },
        # 3. 12 vs 3: normally Hit, Stand when TC >= +2
        {
            "name": "12 vs 3",
            "player_sum": 12,
            "usable_ace": False,
            "dealer_rank": 3,
            "is_pair": False,
            "pair_rank": None,
            "can_double": False,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": STAND,
            "tc_threshold": 2,
            "direction": "gte",
        },
        # 4. 12 vs 2: normally Hit, Stand when TC >= +3
        {
            "name": "12 vs 2",
            "player_sum": 12,
            "usable_ace": False,
            "dealer_rank": 2,
            "is_pair": False,
            "pair_rank": None,
            "can_double": False,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": STAND,
            "tc_threshold": 3,
            "direction": "gte",
        },
        # 5. 11 vs Ace: normally Hit, Double when TC >= +1
        {
            "name": "11 vs A",
            "player_sum": 11,
            "usable_ace": False,
            "dealer_rank": 1,
            "is_pair": False,
            "pair_rank": None,
            "can_double": True,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": DOUBLE,
            "tc_threshold": 1,
            "direction": "gte",
        },
        # 6. 9 vs 2: normally Hit, Double when TC >= +1
        {
            "name": "9 vs 2",
            "player_sum": 9,
            "usable_ace": False,
            "dealer_rank": 2,
            "is_pair": False,
            "pair_rank": None,
            "can_double": True,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": DOUBLE,
            "tc_threshold": 1,
            "direction": "gte",
        },
        # 7. 10 vs 10: normally Hit, Double when TC >= +4
        {
            "name": "10 vs 10",
            "player_sum": 10,
            "usable_ace": False,
            "dealer_rank": 10,
            "is_pair": False,
            "pair_rank": None,
            "can_double": True,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": DOUBLE,
            "tc_threshold": 4,
            "direction": "gte",
        },
        # 8. 10 vs Ace: normally Hit, Double when TC >= +4
        {
            "name": "10 vs A",
            "player_sum": 10,
            "usable_ace": False,
            "dealer_rank": 1,
            "is_pair": False,
            "pair_rank": None,
            "can_double": True,
            "can_split": False,
            "bs_action": HIT,
            "dev_action": DOUBLE,
            "tc_threshold": 4,
            "direction": "gte",
        },
        # 9. 12 vs 4: normally Stand, Hit when TC < 0
        {
            "name": "12 vs 4 (neg)",
            "player_sum": 12,
            "usable_ace": False,
            "dealer_rank": 4,
            "is_pair": False,
            "pair_rank": None,
            "can_double": False,
            "can_split": False,
            "bs_action": STAND,
            "dev_action": HIT,
            "tc_threshold": 0,
            "direction": "lt",
        },
        # 10. 13 vs 2: normally Stand, Hit when TC < -1
        {
            "name": "13 vs 2 (neg)",
            "player_sum": 13,
            "usable_ace": False,
            "dealer_rank": 2,
            "is_pair": False,
            "pair_rank": None,
            "can_double": False,
            "can_split": False,
            "bs_action": STAND,
            "dev_action": HIT,
            "tc_threshold": -1,
            "direction": "lt",
        },
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_deviation(
    net: BlackjackNet,
    dev: Deviation,
    device: torch.device,
    decks_remaining: float = 3.0,
) -> tuple[bool, int]:
    """Test whether the agent plays a single deviation correctly.

    We set the true count to a value clearly past the threshold and check
    whether the agent selects the deviation action.

    Returns:
        (passed, agent_action)
    """
    # Pick a TC that is clearly past the threshold
    if dev["direction"] == "gte":
        test_tc = dev["tc_threshold"] + 2.0
    else:
        test_tc = dev["tc_threshold"] - 2.0

    # Clamp to the encoder's range
    test_tc = float(np.clip(test_tc, -5.0, 5.0))

    obs = encode_state(
        player_sum=dev["player_sum"],
        usable_ace=dev["usable_ace"],
        dealer_upcard_rank=dev["dealer_rank"],
        is_pair=dev["is_pair"],
        pair_rank=dev["pair_rank"],
        can_double=dev["can_double"],
        can_split=dev["can_split"],
        true_count=test_tc,
        decks_remaining=decks_remaining,
    )

    # Build action mask
    mask = np.array([True, True, dev["can_double"], dev["can_split"]], dtype=bool)

    obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask[None], dtype=torch.bool, device=device)

    with torch.no_grad():
        play_q = net(obs_t).clone()
        play_q[~mask_t] = -1e9

    agent_action = int(play_q.argmax(dim=1).item())
    passed = (agent_action == dev["dev_action"])
    return passed, agent_action


def check_all_deviations(
    net: BlackjackNet,
    device: torch.device,
) -> tuple[int, int, list[dict]]:
    """Check all 10 deviations.

    Returns:
        (n_passed, n_total, results_list)
    """
    net.set_deterministic(True)
    deviations = _make_deviations()
    results = []
    n_passed = 0

    for dev in deviations:
        passed, agent_action = check_deviation(net, dev, device)
        n_passed += int(passed)
        results.append({
            "name": dev["name"],
            "bs_action": _ACTION_NAMES[dev["bs_action"]],
            "dev_action": _ACTION_NAMES[dev["dev_action"]],
            "agent_action": _ACTION_NAMES[agent_action],
            "tc_threshold": dev["tc_threshold"],
            "direction": dev["direction"],
            "passed": passed,
        })

    net.set_deterministic(False)
    return n_passed, len(deviations), results


def print_report(n_passed: int, n_total: int, results: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("Milestone 3 — Illustrious 18 Deviation Check")
    print("=" * 65)

    for r in results:
        dir_str = ">=" if r["direction"] == "gte" else "<"
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{status}]  {r['name']:16s}  "
            f"TC {dir_str} {r['tc_threshold']:+d}  "
            f"BS={r['bs_action']:6s}  Dev={r['dev_action']:6s}  "
            f"Agent={r['agent_action']:6s}"
        )

    print("-" * 65)
    pass_milestone = n_passed >= 8
    print(f"  Passed: {n_passed}/{n_total}  (threshold: 8)")
    print(f"  MILESTONE 3 ACCEPTANCE: {'PASS' if pass_milestone else 'FAIL'}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_net(checkpoint_path: str, device: torch.device) -> BlackjackNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    ncfg = {}
    for v in cfg.values() if isinstance(cfg, dict) else [cfg]:
        if isinstance(v, dict):
            ncfg.update(v)

    net = BlackjackNet(ncfg).to(device)
    net.load_state_dict(ckpt["agent"]["online_net"])
    return net


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    net = load_net(args.checkpoint, device)
    net.eval()

    n_passed, n_total, results = check_all_deviations(net, device)
    print_report(n_passed, n_total, results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check agent's play on Illustrious 18 deviations"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a Milestone 3 checkpoint .pt file")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
