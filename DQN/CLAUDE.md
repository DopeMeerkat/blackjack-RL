# CLAUDE.md — Development Conventions

## Authoritative sources

The simulator rules (§2), action indices (§4), state vector layout (§3), and reward
magnitudes (§5) in `blackjack_rl_design.md` are authoritative. Any change to these
requires explicit confirmation from the user before implementation.

## Hyperparameters

All hyperparameters live in `configs/default.yaml` (see §16 of the design doc).
Variant configs (e.g., `configs/eps_greedy.yaml`) override only the differing fields.
Do **not** hard-code values in training scripts.

## Testing

Run `pytest` before committing any change to `env/` or `agent/`.

## torch.compile

`torch.compile` is opt-in only, and only after correctness is established. It interacts
poorly with NoisyLinear's per-step `reset_noise()` if not wrapped carefully.

## Logging

- TensorBoard logs → `outputs/runs/<experiment_name>/`
- Checkpoints → `outputs/checkpoints/<experiment_name>/`
- The `outputs/` directory is gitignored.

## When the design doc is silent on a tradeoff

Prefer the simpler implementation and note the alternative in a code comment.

## When code and doc disagree

Fix the code, not the doc — unless the doc is wrong, in which case stop and confirm
with the user before changing either.

## Vec env auto-reset

`VecBlackjackEnv` does **not** auto-reset on `done=True`. The training loop is
responsible for calling `env.reset()` on completed environments before the next step.
