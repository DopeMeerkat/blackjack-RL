"""Curriculum training: stages action-space complexity from hit/stand to full game.

Phases (configured in configs/default.yaml under 'curriculum.phases'):
  1. hit_stand  – hit + stand only, count zeroed, flat 1x bet
  2. doubles    – adds doubling down, count zeroed, flat 1x bet
  3. splits     – full play action space, count zeroed, flat 1x bet
  4. full_game  – all actions, true count enabled, flat 1x bet

Phase advancement requires both:
  (a) phase_hands >= phase.hands
  (b) eval_ev >= phase.min_ev  (skipped when min_ev is null)

Usage:
  cd blackjack-RL/DQN
  python train/train_curriculum.py
  python train/train_curriculum.py --exp-name my_run
  python train/train_curriculum.py --total-hands 1000000 --exp-name smoke_test

Outputs:
  outputs/checkpoints/<exp_name>/     checkpoints every eval interval + final
  outputs/runs/<exp_name>/            TensorBoard logs
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agent.dqn import DQNAgent
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
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    return cfg


def net_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in (
        "obs_dim", "playing_actions",
        "trunk_hidden", "trunk_layers",
        "head_hidden", "noisy_sigma0",
        "dueling", "n_atoms", "v_min", "v_max",
    ) if k in cfg}


def train_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in (
        "gamma", "target_tau", "batch_size", "learning_rate",
        "replay_alpha", "grad_clip", "n_step"
    ) if k in cfg}


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

_ACTION_NAMES = ["hit", "stand", "double", "split"]


def actions_to_mask(actions_allowed: list[int]) -> np.ndarray:
    """Convert a list of allowed action indices to a (4,) bool mask."""
    mask = np.zeros(4, dtype=bool)
    for a in actions_allowed:
        mask[a] = True
    return mask


# ---------------------------------------------------------------------------
# Beta annealing
# ---------------------------------------------------------------------------

def beta_schedule(step: int, total_steps: int,
                  beta_start: float, beta_end: float) -> float:
    frac = min(step / max(total_steps, 1), 1.0)
    return beta_start + frac * (beta_end - beta_start)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def quick_eval(
    agent: DQNAgent,
    cfg: dict,
    n_hands: int,
    phase: dict,
    device: torch.device,
    num_envs: int = 64,
    seed: int = 99999,
) -> dict:
    """Evaluate under the constraints of the current curriculum phase.

    Applies curriculum mask and count zeroing so the reported EV is
    comparable to the phase's min_ev threshold.  Bet is always flat 1x.
    """
    curriculum_mask = actions_to_mask(phase["actions_allowed"])
    zero_count = phase["zero_count"]

    agent.online_net.set_deterministic(True)
    vec_env = VecBlackjackEnv(
        num_envs, cfg, seeds=[seed + i for i in range(num_envs)]
    )
    obs, masks, infos = vec_env.reset()

    rewards: list[float] = []
    action_counts = np.zeros(4, dtype=np.int64)

    while len(rewards) < n_hands:
        in_bet = vec_env.in_bet_phase
        actions = np.zeros(num_envs, dtype=np.int32)  # bet envs: flat 1x (action=0)

        play_idx = np.where(~in_bet)[0]
        if len(play_idx) > 0:
            batch_obs = obs[play_idx].copy()
            if zero_count:
                batch_obs[:, 25] = 0.0
            batch_masks = masks[play_idx] & curriculum_mask
            obs_t = torch.tensor(batch_obs, dtype=torch.float32, device=device)
            mask_t = torch.tensor(batch_masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                play_q = agent.online_net(obs_t).clone()
                play_q[~mask_t] = -1e9
            play_acts = play_q.argmax(dim=1).cpu().numpy().astype(np.int32)
            actions[play_idx] = play_acts
            for a in play_acts:
                action_counts[a] += 1

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

    agent.online_net.set_deterministic(False)
    ev = float(np.mean(rewards[:n_hands]))

    return {
        "ev": ev,
        "action_counts": action_counts,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> DQNAgent:
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )

    ckpt = None
    if args.checkpoint:
        chkpt_path = f"./outputs/checkpoints/{args.exp_name}/{args.checkpoint}"
        print(f"Loading checkpoint: {chkpt_path}")
        ckpt = torch.load(chkpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        print("  Using config from checkpoint.")
    else:
        cfg = load_config(args.config)

    phases: list[dict] = cfg.get("phases", [])
    if not phases:
        raise ValueError("No curriculum phases found in config. "
                         "Check 'curriculum.phases' in default.yaml.")

    total_hands = args.total_hands or cfg.get("total_hands", 200_000_000)
    exp_name = args.exp_name

    print(f"Device:       {device}")
    print(f"Experiment:   {exp_name}")
    print(f"Total hands:  {total_hands:,}")
    print(f"Phases:       {[p['name'] for p in phases]}")

    ckpt_dir = Path("outputs/checkpoints") / exp_name
    tb_dir = Path("outputs/runs") / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if _TB_AVAILABLE:
        writer = SummaryWriter(str(tb_dir))

    ncfg = net_config(cfg)
    tcfg = train_config(cfg)
    agent = DQNAgent(
        net_config=ncfg,
        train_config=tcfg,
        replay_capacity=cfg.get("replay_buffer_size", 1_000_000),
        device=device,
    )

    if ckpt is not None:
        agent.load_state_dict(ckpt["agent"])
        print("  Loaded agent weights from checkpoint.")

    num_envs = cfg.get("num_envs", 64)
    vec_env = VecBlackjackEnv(
        num_envs=num_envs,
        config=cfg,
        seeds=list(range(num_envs)),
    )

    train_every = cfg.get("train_every_n_steps", 4)
    beta_start = cfg.get("replay_beta_start", 0.4)
    eval_every = cfg.get("eval_every_n_hands", 1_000_000)
    eval_hands = cfg.get("eval_hands", 1_000_000)
    ckpt_every = cfg.get("eval_every_n_hands", 1_000_000)
    warmup_transitions = cfg.get("warmup_transitions", 100_000)

    # Curriculum state (resumed from checkpoint if present)
    if ckpt is not None:
        phase_idx = ckpt["phase_idx"]
        phase_hands = ckpt["phase_hands"]
        hands_played = ckpt["hands_played"]
        print(f"  Resumed at phase={phases[phase_idx]['name']}, "
              f"phase_hands={phase_hands:,}, hands_played={hands_played:,}")
    else:
        phase_idx = 0
        phase_hands = 0
        hands_played = 0

    last_eval_ev = -np.inf

    def current_phase() -> dict:
        return phases[phase_idx]

    def announce_phase(idx: int, hp: int) -> None:
        p = phases[idx]
        print(
            f"\n>>> Phase {idx}: {p['name']}  "
            f"(actions={p['actions_allowed']}, zero_count={p['zero_count']})"
        )
        if writer:
            writer.add_scalar("train/curriculum_phase", idx, hp)

    announce_phase(phase_idx, hands_played)

    obs_arr, masks_arr, infos = vec_env.reset()
    global_step = 0
    last_eval = hands_played
    last_ckpt = hands_played
    t_start = time.time()

    # Ctrl+C → finish current step then save an interrupt checkpoint.
    # A second Ctrl+C restores the default handler and hard-kills.
    _interrupted = [False]

    def _handle_sigint(signum, frame):
        _interrupted[0] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[Interrupted] Finishing current step, then saving…")

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"\nStarting curriculum training...  "
          f"(batch_size={agent.batch_size}, buffer={agent.replay.capacity:,})")

    while hands_played < total_hands:
        phase = current_phase()
        curriculum_mask = actions_to_mask(phase["actions_allowed"])
        zero_count = phase["zero_count"]

        in_bet = vec_env.in_bet_phase  # (N,) bool

        # --- Action selection ---
        actions = np.zeros(num_envs, dtype=np.int32)  # bet envs: flat 1x (action=0)

        play_envs = np.where(~in_bet)[0]
        if len(play_envs) > 0:
            batch_obs = obs_arr[play_envs].copy()
            if zero_count:
                batch_obs[:, 25] = 0.0
            batch_masks = masks_arr[play_envs] & curriculum_mask
            actions[play_envs] = agent.select_play_actions_batch(
                batch_obs, batch_masks
            )

        # --- Step all envs ---
        next_obs, next_masks, rewards, dones, infos = vec_env.step(actions)

        # --- Store play transitions ---
        for i in play_envs:
            ob = obs_arr[i].copy()
            next_ob = next_obs[i].copy()
            if zero_count:
                ob[25] = 0.0
                next_ob[25] = 0.0
            effective_mask = masks_arr[i] & curriculum_mask
            effective_next_mask = next_masks[i] & curriculum_mask
            agent.replay.add(
                obs=ob,
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_obs=next_ob,
                done=bool(dones[i]),
                mask=effective_mask,
                next_mask=effective_next_mask,
            )

        # --- Handle done envs ---
        for i in range(num_envs):
            if dones[i]:
                hands_played += 1
                phase_hands += 1
                reset_obs, reset_mask, _ = vec_env.reset_at(i)
                next_obs[i] = reset_obs
                next_masks[i] = reset_mask

        obs_arr = next_obs
        masks_arr = next_masks
        global_step += 1

        # --- Gradient update ---
        if global_step % train_every == 0 and len(agent.replay) >= warmup_transitions:
            beta = beta_schedule(hands_played, total_hands, beta_start, 1.0)
            loss = agent.train_step(beta)

            if writer and loss is not None and global_step % 1000 == 0:
                writer.add_scalar("train/loss", loss, hands_played)
                writer.add_scalar("train/beta", beta, hands_played)
                writer.add_scalar("train/replay_size", len(agent.replay), hands_played)

        # --- Periodic eval + phase gate ---
        if hands_played - last_eval >= eval_every:
            stats = quick_eval(agent, cfg, eval_hands, phase, device)
            last_eval_ev = stats["ev"]
            elapsed = time.time() - t_start
            rate = hands_played / elapsed

            total_acts = stats["action_counts"].sum()
            action_pcts = stats["action_counts"] / max(total_acts, 1)
            action_str = "  ".join(
                f"{name}={action_pcts[i]*100:.1f}%"
                for i, name in enumerate(_ACTION_NAMES)
            )

            phase_budget = phase["hands"]
            phase_pct = phase_hands / phase_budget * 100
            print(
                f"  hands={hands_played:>10,}  "
                f"EV={stats['ev']*100:+.3f}%  "
                f"phase={phase['name']} ({phase_pct:.0f}%)  "
                f"rate={rate/1000:.1f}k/s\n"
                f"    actions: {action_str}"
            )

            if writer:
                writer.add_scalar("eval/ev", stats["ev"], hands_played)
                writer.add_scalar("eval/phase_hands", phase_hands, hands_played)
                for i, name in enumerate(_ACTION_NAMES):
                    writer.add_scalar(
                        f"eval/action_pct_{name}", action_pcts[i], hands_played
                    )

            min_ev = phase.get("min_ev")
            ev_ok = (min_ev is None) or (last_eval_ev >= min_ev)
            if phase_hands >= phase["hands"] and ev_ok:
                if phase_idx + 1 < len(phases):
                    phase_idx += 1
                    phase_hands = 0
                    announce_phase(phase_idx, hands_played)
                # else: final phase — keep training until total_hands
            elif phase_hands >= phase["hands"] and not ev_ok:
                print(
                    f"    [EV gate] phase {phase['name']}: "
                    f"EV={last_eval_ev*100:+.3f}% < min={min_ev*100:+.3f}%  "
                    f"continuing..."
                )

            last_eval = hands_played

        # --- Periodic checkpoint ---
        if hands_played - last_ckpt >= ckpt_every:
            ckpt_path = ckpt_dir / f"step_{hands_played:010d}.pt"
            torch.save({
                "agent": agent.state_dict(),
                "hands_played": hands_played,
                "phase_idx": phase_idx,
                "phase_hands": phase_hands,
                "config": cfg,
            }, ckpt_path)
            last_ckpt = hands_played

        # --- Save on interrupt ---
        if _interrupted[0]:
            interrupt_path = ckpt_dir / f"interrupt_{hands_played:010d}.pt"
            torch.save({
                "agent": agent.state_dict(),
                "hands_played": hands_played,
                "phase_idx": phase_idx,
                "phase_hands": phase_hands,
                "config": cfg,
            }, interrupt_path)
            print(f"[Interrupted] Saved checkpoint: {interrupt_path}")
            if writer:
                writer.close()
            return agent

    # --- Final checkpoint ---
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "agent": agent.state_dict(),
        "hands_played": hands_played,
        "phase_idx": phase_idx,
        "phase_hands": phase_hands,
        "config": cfg,
    }, final_path)
    print(f"\nTraining complete.  Final checkpoint: {final_path}")

    if writer:
        writer.close()

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curriculum training: hit/stand → doubles → splits → full game"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--total-hands", type=int, default=None,
                   help="Override total training hands (default: from config).")
    p.add_argument("--exp-name", default=None, required=True,
                   help="Experiment name for checkpoints/logs.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional: warm-start from an existing checkpoint.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
