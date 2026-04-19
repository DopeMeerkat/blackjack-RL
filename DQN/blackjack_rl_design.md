# Design Document: Reinforcement Learning Agent for Card-Counting Blackjack

## 1. Project Overview

This project trains a reinforcement learning agent to play blackjack at a level that exploits card-counting information. The agent makes two decisions per hand: a bet size at the start of each hand, and a sequence of playing decisions (hit, stand, double, split) once the cards are dealt. The agent has access to the running true count as part of its observation, so it does not need to learn counting from raw card streams — it needs to learn how to *use* the count, both for play deviations and for bet sizing.

The target outcome is a positive expected value per shoe under realistic casino rules, demonstrably better than flat-betting basic strategy. The project is scoped to roughly one week of work on a single workstation with one GPU.

## 2. Environment Specification

The simulator models a six-deck shoe with approximately 75% penetration before reshuffle. House rules are fixed up front and held constant across training and evaluation: dealer stands on soft 17, double-after-split is allowed, resplitting is capped at three hands, blackjack pays 3:2, and surrender is not offered. Deck composition persists across hands within a shoe so that the running count carries real information.

The simulator runs vectorized across at least 64 parallel environments and exposes, at every decision point, the legal action mask along with the observation vector. Episodes are defined as a single hand (potentially with sub-hands from splits), with the bet-sizing decision treated as the first step of the episode.

## 3. State Representation

The observation is a fixed 28-dimensional vector with the following exact layout:

| Indices | Feature | Encoding |
|---------|---------|----------|
| 0 | Player hand sum | `(sum − 4) / 17`, so 4 → 0.0 and 21 → 1.0 |
| 1 | Usable ace flag | `{0, 1}` |
| 2–11 | Dealer up-card | One-hot over `[A, 2, 3, 4, 5, 6, 7, 8, 9, 10]`; J/Q/K map to index 11 |
| 12 | Pair flag | `{0, 1}`; set when first two cards are equal-value |
| 13–22 | Pair rank | One-hot over `[A, 2, 3, 4, 5, 6, 7, 8, 9, 10]`; all zeros when pair flag is 0 |
| 23 | Can-double flag | `{0, 1}` |
| 24 | Can-split flag | `{0, 1}` |
| 25 | True count | `clip(running_count / decks_remaining, −5, +5) / 5` |
| 26 | Decks remaining | `decks_remaining / 6` |
| 27 | Current bet (normalized) | `bet_multiplier / 12` |

At bet-sizing time, the playing-state features (indices 0, 1, 12–24, 27) are not yet defined and are set to zero. The bet-sizing head sees the trunk's encoding of this masked observation and outputs Q-values over bet multipliers. This keeps the architecture uniform: one trunk, two heads, single 28-dim input format.

## 4. Action Space

The playing-policy action space is four discrete actions with fixed indices:

| Index | Action | Legality conditions |
|-------|--------|---------------------|
| 0 | Hit | Always legal during a hand |
| 1 | Stand | Always legal during a hand |
| 2 | Double | First two cards of a hand only; allowed on hands created by splitting (DAS rule) |
| 3 | Split | First two cards must be equal-value; current sub-hand count below the resplit cap (3) |

Surrender is omitted per the project's specified rule set.

Illegal actions are handled with hard masking: the environment returns a boolean mask of shape `(4,)` at every step alongside the observation, and the Q-network's playing-head logits are set to `−1e9` at masked indices before argmax/softmax. The same masking is applied inside the Bellman target (see §15.3) so that the bootstrap never selects an illegal action even when the target network is stale. Reward-based illegal-action penalties are explicitly avoided because they create credit-assignment noise.

The bet-sizing action space is a discrete set of five bet multipliers with fixed indices:

| Index | Multiplier |
|-------|------------|
| 0 | 1× |
| 1 | 2× |
| 2 | 4× |
| 3 | 8× |
| 4 | 12× |

A discrete set is used rather than a continuous output to keep the algorithm uniform (DQN end-to-end) and to make Kelly-style sizing emerge naturally from the count signal.

## 5. Reward Structure

Rewards are sparse and terminal. A win pays +1 times the bet multiplier; a loss pays −1 times the bet multiplier; a push pays zero. A natural blackjack pays +1.5 times the bet multiplier. A doubled hand multiplies the terminal outcome by two. Split hands sum the outcomes across all sub-hands, each scaled by its own bet (which equals the original bet per sub-hand).

Discounting within a hand is set to gamma = 1.0, since horizons are short (a handful of decisions) and there is no reason to prefer earlier rewards over later ones inside a single hand.

The bet-sizing decision receives the same terminal reward as the playing policy, but scaled by the chosen bet — this is what allows the bet-sizing head to learn that high counts justify larger bets.

## 6. Network Architecture

A small multi-layer perceptron is sufficient. There is no spatial or sequential structure that would justify CNNs or RNNs, given that the count is provided as a feature.

The architecture has one shared trunk and two heads:

- **Trunk**: `Linear(28 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 256) → ReLU`. Standard `torch.nn.Linear`. Output is a 256-dim feature vector.
- **Playing head**: `NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → 4)`. Output is four Q-values, one per playing action; masked post-hoc.
- **Bet-sizing head**: `NoisyLinear(256 → 256) → ReLU → NoisyLinear(256 → 5)`. Output is five Q-values, one per bet multiplier.

A target network with identical architecture is maintained and updated via Polyak averaging (τ = 0.005).

`NoisyLinear` is **not** part of core PyTorch and must be implemented from scratch — exact formula in §15.1. The trunk uses standard `Linear` layers; only the heads are noisy. This follows the recipe from Fortunato et al. (2017) and keeps noise output-relevant for stable training.

Total parameter count is approximately 400K (NoisyLinear roughly doubles the parameters of an equivalent Linear layer because it learns both `μ` and `σ`). Well within budget for fast training on a single GPU.

## 7. Learning Algorithm

The algorithm is Double DQN with a target network and prioritized experience replay. This combination is the standard robust choice for small discrete-action problems with cheap simulators and short horizons, all of which describe this setting.

Hyperparameters are set to common defaults:

- Replay buffer size: 1,000,000 transitions
- Batch size: 512
- Optimizer: Adam, learning rate 3e-4
- Target network: Polyak updates with τ = 0.005
- Train step every 4 environment steps
- Prioritized replay alpha = 0.6, beta annealed from 0.4 to 1.0 over training

Both the playing head and the bet-sizing head are trained with the same Double DQN objective, sharing the trunk. Transitions for the bet-sizing decision and the playing decisions are stored in the same replay buffer, with a flag indicating which head's loss to compute. Gradients from both heads flow into the trunk.

An alternative training schedule — train the playing policy to convergence first, freeze it, then train the bet-sizing head — was considered. It decouples credit assignment cleanly but doubles training time. Joint training is preferred for the project timeline and is empirically stable when both heads share a trunk and use the same algorithm.

## 8. Bet Sizing

Bet sizing is the lever through which card counting actually translates to positive expected value. The playing policy alone, even with perfect deviations, only narrows the house edge — bet variation is what produces a player edge.

At the start of each hand, the bet-sizing head observes the current true count and decks remaining and selects a bet multiplier from {1, 2, 4, 8, 12}. The chosen multiplier scales all subsequent rewards from that hand. Because the multiplier is part of the observation passed to the playing head, the playing policy can adjust its risk tolerance for doubles based on the bet at stake, though in practice this effect is small.

A risk-of-ruin or bankroll constraint is not modeled in this version. The agent optimizes pure expected value per hand. A more realistic version would penalize variance or impose a bankroll cap, but this is left as an extension.

## 9. Exploration: Noisy Networks

This project uses NoisyNets (Fortunato et al., 2017) in place of ε-greedy exploration. The key idea is to inject learnable parametric Gaussian noise into the weights and biases of selected linear layers. Each forward pass samples fresh noise, so the policy is stochastic in a way that depends on the input state. The noise parameters (sigma) are learned by gradient descent like any other parameter, so the network can choose to be more deterministic in well-understood states and noisier in uncertain ones.

The factorized Gaussian variant is used, which decomposes the noise as an outer product of two vectors per layer. This is computationally cheaper than independent Gaussian noise per weight and is the standard recipe in the original paper.

Initial sigma_0 is set to 0.5 (the published default). Noise is resampled per forward pass during training and at the start of each episode during evaluation rollouts; for greedy evaluation, noise is set to its mean (i.e., disabled).

### 9.1 Why NoisyNets fit this problem

State distribution in blackjack is highly skewed: most hands are dealt at neutral or near-neutral counts, while the strategically interesting states (high |true count|, where deviations from basic strategy matter) are rare. With ε-greedy, the exploration rate is identical in common and rare states, so the agent under-explores precisely where exploration matters most. NoisyNets give state-dependent exploration: the agent can learn to be confident in well-trodden low-count states while still exploring action choices in rare high-count states where it has less data.

A second advantage is the absence of an exploration schedule. ε-greedy requires choosing a decay schedule (linear from 1.0 to 0.05 over N steps), and the right N is problem-dependent. NoisyNets eliminate this hyperparameter by letting the network itself decide how much noise to retain, which reduces tuning burden in a short-timeline project.

### 9.2 Evaluation plan

The effectiveness of NoisyNets versus ε-greedy will be measured directly with a controlled A/B comparison. Both variants share architecture, optimizer, replay configuration, environment seeds, and training budget; only the exploration mechanism differs. The ε-greedy baseline uses linear decay from 1.0 to 0.05 over the first 500K steps, then holds at 0.05.

The metrics are:

1. **Sample efficiency**: hands of training required to reach a fixed performance threshold (e.g., within 0.2% EV of basic strategy with no count). This isolates how quickly each method converges on the well-known optimum.
2. **Final expected value per hand**, evaluated over at least one million hands against a fixed basic-strategy baseline using paired shoe seeds for variance reduction.
3. **Deviation discovery rate**: fraction of well-known count-based deviations (insurance at TC ≥ 3, 16 vs 10 stand at TC ≥ 0, 12 vs 3 stand at TC ≥ 2, etc.) that the agent learns to play correctly. This metric specifically probes the rare-state regime where NoisyNets are predicted to help.
4. **Bet-sizing quality**: correlation between true count and chosen bet multiplier in evaluation, and EV per shoe versus a flat-betting variant of the same playing policy.
5. **Wall-clock training time** to reach a fixed performance threshold, since NoisyLinear layers add per-step overhead.

The hypothesis is that NoisyNets will match or exceed ε-greedy on metrics 1 and 2, will outperform on metric 3 (especially for deviations triggered at extreme counts), and will pay a small wall-clock cost on metric 5. If NoisyNets do not show a clear advantage on metric 3, the choice between methods becomes a near-wash and the simpler method (ε-greedy) would be preferred for the production agent.

## 10. Training Procedure

Training proceeds in a single phase with both heads learning jointly. At each environment step, the appropriate head produces logits (with masking and noise sampling), an action is selected, and the resulting transition is pushed to the replay buffer with a head flag. Every four environment steps, a batch of 512 transitions is sampled with prioritization, Q-targets are computed using the target network, and the loss is the sum of the playing-head and bet-sizing-head losses on their respective transitions.

64 parallel environments run on CPU with the network on GPU. At roughly 50–100 million hands per day on this setup, the training budget is 200–300 million hands across three days, which exceeds what is typically required for convergence on this problem.

## 11. Evaluation Protocol

Evaluation is performed periodically during training (every 1M hands) and once at the end. Each evaluation runs the deterministic version of the policy (NoisyNet noise disabled, no exploration) for at least 1M hands against a fresh shoe stream, and reports:

- Mean reward per hand (EV) and its standard error
- Mean reward per shoe, paired against a basic-strategy baseline on identical seeds
- Action distribution conditional on (player total, dealer up-card, true count bucket), compared to published basic-strategy and deviation tables
- Bet distribution conditional on true count bucket

The paired-difference design is critical for variance reduction. Blackjack outcomes are noisy enough that unpaired comparisons over 1M hands can fail to resolve a 0.5% EV difference, while paired comparisons resolve it cleanly.

## 12. Risks and Mitigations

The biggest risk is **insurance**, which is a side-bet decision available only when the dealer shows an ace. It is technically a separate action and is almost always wrong unless the count is high. For this project, insurance is hard-coded to take when the true count is at least +3 and is not part of the learned action space. This avoids the action-space asymmetry that comes with conditionally-available actions.

The second risk is **evaluation variance**. As noted above, paired evaluation over at least 1M hands is required for stable EV estimates. Underpowered evaluations have a real risk of producing misleading conclusions about the NoisyNets vs ε-greedy comparison.

The third risk is **rare-state under-training**. Even with 200M training hands, the count of training experience at extreme true counts (|TC| ≥ 4) may be sparse. If the deviation-discovery metric is poor for both exploration strategies, prioritized replay weights should be augmented with a count-magnitude bonus, oversampling high-|count| transitions during training.

The fourth risk is **bet-sizing instability** early in training. Before the playing policy is competent, large bets at high counts produce large negative rewards, which can push the bet-sizing head toward conservative bets it never recovers from. If observed, the mitigation is to clip bet multipliers to {1, 2} for the first several million training steps and gradually unlock the larger bets, or to fall back on the decoupled training schedule (freeze playing policy, then train bet head).

## 13. Implementation Milestones

Each milestone has explicit deliverables and an acceptance test. Do not advance to the next milestone until the current one's test passes.

### Milestone 1 — Simulator and environment interface (Day 1)

**Deliverables**: `env/blackjack.py` (single-environment game logic), `env/vec_env.py` (vectorized wrapper), `env/encoding.py` (state vector encoder per §3), `env/count.py` (Hi-Lo running count per §15.2), `tests/test_env.py`.

**Acceptance test**: `pytest tests/test_env.py` passes, including:
- Random-policy expected value over 100K hands lies in [−1.5%, −0.4%] (sanity check on house edge for this rule set).
- State vector dimension is exactly 28; mask shape is exactly `(4,)`.
- Hi-Lo count update on a hand-checked card sequence matches §15.2 exactly.
- Splits produce sub-hands whose summed reward is correctly returned at hand termination.

### Milestone 2 — Playing policy with NoisyNets, no count features (Day 2 morning)

**Deliverables**: `agent/noisy_linear.py` (per §15.1), `agent/network.py`, `agent/replay.py`, `agent/dqn.py`, `train/train_play.py`, `eval/compare_basic_strategy.py`.

**Acceptance test**: After training for 10M hands with the count feature zeroed and the bet head disabled (flat 1× bet), evaluated EV is within 0.2% of published basic-strategy EV for the rule set, and the agent's argmax action agrees with the basic-strategy chart on at least 95% of `(player_total, dealer_upcard)` cells.

### Milestone 3 — Playing policy with count features (Day 2 afternoon)

**Deliverables**: enable count feature in encoder; rerun training with the same configuration. Add `eval/check_deviations.py`.

**Acceptance test**: After training for 50M hands, the agent correctly plays at least 8 of the 10 most common Hi-Lo deviations from the Illustrious 18 list (excluding any that require surrender). Verified by `eval/check_deviations.py`.

### Milestone 4 — Bet-sizing head and joint training (Day 3)

**Deliverables**: enable bet head in `agent/network.py`; `train/train_joint.py`; bet-evaluation routines in `eval/`.

**Acceptance test**: After joint training for 100M hands, evaluated over 1M hands:
- Pearson correlation between true count and chosen bet multiplier is at least 0.6.
- Joint policy EV per shoe is positive.
- Joint policy EV per shoe exceeds the flat-betting variant of the same playing policy by a statistically significant margin (paired-difference t-test on identical shoe seeds, p < 0.01).

### Milestone 5 — NoisyNets vs ε-greedy A/B comparison (Day 4)

**Deliverables**: `train/train_eps_greedy.py` (identical architecture and budget, ε-greedy exploration), `eval/ab_report.py`.

**Acceptance test**: A report at `outputs/ab_report.md` containing all five metrics from §9.2 with confidence intervals, plus a final recommendation paragraph based on the results.

### Milestone 6 — Buffer and writeup (Day 5)

Reserved for risk mitigation (§12) and final writeup. No new acceptance criteria.

## 14. Implementation Stack and Repository Layout

### Tech stack

- Python 3.11+
- PyTorch 2.x (network, training loop, autograd)
- NumPy (simulator state, vectorized env)
- pytest (tests)
- TensorBoard (training metrics; no external service required)
- PyYAML (config files)

The simulator is built directly on NumPy rather than wrapping Gymnasium, but the env interface mirrors Gymnasium's `reset()` / `step()` convention for familiarity. Training loop, replay buffer, NoisyLinear layer, and DQN logic are written from scratch — using a high-level RL library (Stable-Baselines3, RLlib, CleanRL) is **not recommended** for this project because none ships NoisyNets together with a custom multi-head value architecture sharing a trunk. Reaching for a library will create more friction than it removes.

### Repository layout

```
blackjack-rl/
├── README.md
├── CLAUDE.md                  # see §17
├── pyproject.toml
├── env/
│   ├── __init__.py
│   ├── blackjack.py           # single-env game logic
│   ├── vec_env.py             # vectorized wrapper, 64 envs
│   ├── encoding.py            # state vector encoder per §3
│   └── count.py               # Hi-Lo running count per §15.2
├── agent/
│   ├── __init__.py
│   ├── noisy_linear.py        # custom NoisyLinear per §15.1
│   ├── network.py             # trunk + playing head + bet head
│   ├── replay.py              # prioritized experience replay
│   └── dqn.py                 # Double DQN training step
├── train/
│   ├── train_play.py          # playing-only (Milestones 2–3)
│   ├── train_joint.py         # joint playing + bet (Milestone 4)
│   └── train_eps_greedy.py    # ε-greedy variant (Milestone 5)
├── eval/
│   ├── compare_basic_strategy.py
│   ├── check_deviations.py
│   └── ab_report.py
├── tests/
│   ├── test_env.py
│   ├── test_encoding.py
│   ├── test_noisy_linear.py
│   └── test_dqn.py
├── configs/
│   └── default.yaml           # all hyperparameters per §16
└── outputs/                   # gitignored: reports, checkpoints, TB logs
```

All hyperparameters live in `configs/default.yaml`. Variant configs (e.g., `configs/eps_greedy.yaml`) override only the differing fields.

## 15. Algorithmic Details

### 15.1 NoisyLinear (factorized Gaussian)

A NoisyLinear layer with input dim `p` and output dim `q` has four learnable parameter tensors: `μ_W ∈ R^{q×p}`, `σ_W ∈ R^{q×p}`, `μ_b ∈ R^q`, `σ_b ∈ R^q`. On each forward pass:

1. Sample raw noise vectors `ε_in ∈ R^p` and `ε_out ∈ R^q`, each i.i.d. `N(0, 1)`.
2. Apply `f(x) = sign(x) · sqrt(|x|)` element-wise to each, producing `f(ε_in)` and `f(ε_out)`.
3. Form `ε_W = f(ε_out) ⊗ f(ε_in)` (outer product, shape `q × p`) and `ε_b = f(ε_out)`.
4. Effective parameters: `W = μ_W + σ_W ⊙ ε_W`, `b = μ_b + σ_b ⊙ ε_b`.
5. Forward: `y = Wx + b`.

Initialization: `μ` entries drawn uniformly from `[−1/√p, 1/√p]`; `σ` entries initialized to `σ_0 / √p` with `σ_0 = 0.5`.

For deterministic evaluation, set `ε_in = 0` and `ε_out = 0`, reducing the layer to `y = μ_W x + μ_b`.

Implement as a `torch.nn.Module` with a `reset_noise()` method called before each training forward pass. Provide a `deterministic=True` flag (or context manager) for evaluation.

### 15.2 Hi-Lo running count update

Update the running count whenever a card is revealed (dealt to player, dealer, or burned):

| Card rank | Δ count |
|-----------|---------|
| 2, 3, 4, 5, 6 | +1 |
| 7, 8, 9 | 0 |
| 10, J, Q, K, A | −1 |

True count = `running_count / max(decks_remaining, 0.5)`. The floor of 0.5 decks prevents division blow-ups near the cut card. `decks_remaining = cards_remaining_in_shoe / 52`.

### 15.3 Action masking in the Bellman target

Define `mask(s, a) = 0` if action `a` is legal in state `s`, else `−1e9`. The Double DQN target for a non-terminal next state `s'` is:

```
a*  = argmax_a [ Q_online(s', a) + mask(s', a) ]      # action selected by online net
y   = r + γ · ( Q_target(s', a*) + mask(s', a*) )      # bootstrap from target net
```

This guarantees the bootstrap target never selects an illegal action even when the target network is stale.

### 15.4 Bet-head transitions

A "bet decision" transition is stored at the start of each hand with: the bet-time observation (28-dim vector, playing-state features zeroed), the chosen bet index, the terminal hand reward summed across all sub-hands and scaled by the chosen multiplier, and `done = True`. Because the bet head's reward is the outcome of the entire downstream hand, it is a one-step bandit problem from the head's perspective; no bootstrapping into a next state is required, and γ does not apply.

Playing-decision transitions store the standard `(s, a, r, s', done, mask)` tuple. A `head_id` field on each transition tells the training loop which head's loss to compute.

### 15.5 Reward attribution within a hand

Within a playing decision sequence, intermediate steps yield reward 0; only the terminal step (player busts, dealer resolves, or all sub-hands resolve) yields the win/loss/push outcome. With γ = 1.0 and no intermediate reward, the un-discounted return for every non-terminal step in the hand equals the terminal reward. For split hands, each sub-hand resolves independently and the rewards are accumulated; the sum (scaled by the bet multiplier) is the reward attributed to the bet-head transition for that hand.

## 16. Hyperparameter Reference

All values consolidated for direct copy into `configs/default.yaml`.

### Environment

| Parameter | Value |
|-----------|-------|
| Number of decks | 6 |
| Penetration | 75% (reshuffle when 1.5 decks remain) |
| Dealer rule | Stand on soft 17 (S17) |
| Double after split | Allowed |
| Resplit cap | 3 hands max |
| Blackjack payout | 3:2 |
| Surrender | Disallowed |
| Insurance | Hard-coded: take when true count ≥ +3 |
| Parallel envs | 64 |

### State and action spaces

| Parameter | Value |
|-----------|-------|
| Observation dim | 28 |
| Playing actions | 4 (hit=0, stand=1, double=2, split=3) |
| Bet multipliers | [1, 2, 4, 8, 12] (indices 0–4) |
| True-count clip range | [−5, +5] |

### Network

| Parameter | Value |
|-----------|-------|
| Trunk | 3 × Linear(256), ReLU between |
| Heads | 2 × NoisyLinear(256→256→K), ReLU between; K=4 (play) or 5 (bet) |
| NoisyLinear σ₀ | 0.5 |
| Activation | ReLU |

### Training

| Parameter | Value |
|-----------|-------|
| Algorithm | Double DQN |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Batch size | 512 |
| Replay buffer size | 1,000,000 |
| Replay prioritization α | 0.6 |
| Replay importance β | linear 0.4 → 1.0 over training |
| Target update | Polyak, τ = 0.005 |
| Train step frequency | every 4 environment steps |
| Discount γ (within hand) | 1.0 |
| Bet head training | one-step bandit (no bootstrap) |
| Total joint training hands | 200M |

### ε-greedy baseline (A/B comparison only)

| Parameter | Value |
|-----------|-------|
| ε start | 1.0 |
| ε end | 0.05 |
| ε decay | linear over first 500K env steps |

### Evaluation

| Parameter | Value |
|-----------|-------|
| Eval frequency | every 1M training hands |
| Hands per eval | 1,000,000 |
| Paired baseline | flat-bet basic strategy on identical shoe seeds |

## 17. Recommended CLAUDE.md Conventions

Place a `CLAUDE.md` at the repository root with the following guidance for the implementing agent:

- The simulator rules (§2), action indices (§4), state vector layout (§3), and reward magnitudes (§5) in this design document are authoritative. Any change to these requires explicit confirmation from the user before implementation.
- Hyperparameters live in `configs/default.yaml` (see §16). Do not hard-code values in training scripts.
- Run `pytest` before committing any change to `env/` or `agent/`.
- `torch.compile` is opt-in only after correctness is established; it interacts poorly with NoisyLinear's per-step `reset_noise()` if not wrapped carefully.
- Logging goes to TensorBoard at `outputs/runs/<experiment_name>/`. Checkpoints go to `outputs/checkpoints/<experiment_name>/`.
- When the design document is silent on a tradeoff, prefer the simpler implementation and note the alternative in a code comment.
- When the design document and the code disagree, fix the code, not the document — unless the document is wrong, in which case stop and confirm with the user.
