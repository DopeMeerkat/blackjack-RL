import gymnasium as gym
import numpy as np
from collections import defaultdict

# Blackjack-v1 actions: 0 = stick, 1 = hit

def epsilon_greedy(Q, obs, rng, eps: float):
    if rng.random() < eps:
        return int(rng.integers(0, 2))
    q = Q[obs]
    # random tie-break among max actions
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))

def train_q_learning(
    env_id="Blackjack-v1",
    episodes=500_000,
    alpha=0.1,
    gamma=1.0,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_fraction=0.8,
    seed=0,
):
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)

    # Q maps obs -> np.array([Q(s,stick), Q(s,hit)])
    Q = defaultdict(lambda: np.zeros(2, dtype=np.float32))

    decay_episodes = max(1, int(episodes * eps_decay_fraction))

    def get_eps(t):
        # linear decay from start to end over decay_episodes; then fixed at eps_end
        if t >= decay_episodes:
            return eps_end
        frac = t / decay_episodes
        return eps_start + frac * (eps_end - eps_start)

    returns = np.empty(episodes, dtype=np.float32)

    for ep in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        ep_return = 0.0
        eps = get_eps(ep)

        while not done:
            a = epsilon_greedy(Q, obs, rng, eps)
            next_obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            ep_return += float(r)

            # Q-learning target: r + gamma * max_a' Q(s', a')   (0 if terminal)
            if done:
                target = float(r)
            else:
                target = float(r) + gamma * float(Q[next_obs].max())

            Q[obs][a] += alpha * (target - Q[obs][a])
            obs = next_obs

        returns[ep] = ep_return

    return Q, returns

def evaluate_policy(Q, episodes=50_000, seed=123):
    env = gym.make("Blackjack-v1")
    rng = np.random.default_rng(seed)

    rewards = np.empty(episodes, dtype=np.float32)

    for i in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        total = 0.0

        while not done:
            q = Q[obs]
            best = np.flatnonzero(q == q.max())
            a = int(rng.choice(best))
            obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total += float(r)

        rewards[i] = total

    return {
        "episodes": int(episodes),
        "avg_return": float(rewards.mean()),
        "win_rate": float((rewards == 1).mean()),
        "loss_rate": float((rewards == -1).mean()),
        "push_rate": float((rewards == 0).mean()),
    }

if __name__ == "__main__":
    Q, returns = train_q_learning(
        episodes=1000_000,
        alpha=0.1,
        gamma=1.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_fraction=0.8,
        seed=0,
    )

    stats = evaluate_policy(Q, episodes=50_000, seed=999)
    print("Evaluation (greedy w/ random tie-break):")
    for k, v in stats.items():
        print(f"{k:>10}: {v}")

    print("\nTraining return snapshot (mean over last 10k):", float(returns[-10_000:].mean()))