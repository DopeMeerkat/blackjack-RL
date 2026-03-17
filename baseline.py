import gymnasium as gym
import numpy as np

# Blackjack-v1 actions:
# 0 = stick, 1 = hit

def policy_random(obs, rng):
    return int(rng.integers(0, 2))

def policy_threshold(obs, rng=None, threshold=20):
    player_sum, dealer_card, usable_ace = obs
    return 0 if player_sum >= threshold else 1

def run_episode(env, policy_fn, rng):
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    total_reward = 0.0
    while not done:
        a = policy_fn(obs, rng)
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        total_reward += float(r)
    return total_reward

def evaluate(policy_fn, n_episodes=50_000, seed=0):
    env = gym.make("Blackjack-v1", natural = False, sab = False)  # headless by default
    rng = np.random.default_rng(seed)

    rewards = np.array([run_episode(env, policy_fn, rng) for _ in range(n_episodes)], dtype=np.float32)

    return {
        "episodes": int(n_episodes),
        "avg_return": float(rewards.mean()),
        "win_rate": float((rewards == 1).mean()),
        "loss_rate": float((rewards == -1).mean()),
        "push_rate": float((rewards == 0).mean()),
    }

if __name__ == "__main__":
    for name, pol in [
        ("random", policy_random),
        ("threshold>=20", lambda obs, rng: policy_threshold(obs, rng, threshold=20)),
        # ("qlearning", lambda obs, rng: policy_threshold(obs, rng, threshold=18)),
    ]:
        stats = evaluate(pol, n_episodes=50_000, seed=0)
        print(f"\nPolicy: {name}")
        for k, v in stats.items():
            print(f"{k:>10}: {v}")
