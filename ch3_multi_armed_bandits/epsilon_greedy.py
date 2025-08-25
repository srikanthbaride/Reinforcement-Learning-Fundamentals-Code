import numpy as np

def epsilon_greedy_bandit(n_arms=5, n_steps=200, epsilon=0.1, seed=42):
    rng = np.random.default_rng(seed)

    # true reward means for each arm ~ Normal(0,1)
    true_means = rng.normal(0, 1, n_arms)

    # estimated action values
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)  # counts

    rewards = []
    optimal_actions = 0

    for t in range(n_steps):
        # choose action
        if rng.random() < epsilon:
            a = rng.integers(n_arms)      # explore
        else:
            a = np.argmax(Q)              # exploit

        # reward ~ Normal(true_mean, 1)
        r = rng.normal(true_means[a], 1.0)

        # update
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards.append(r)
        if a == np.argmax(true_means):
            optimal_actions += 1

    avg_reward = np.mean(rewards)
    optimal_action_percent = 100.0 * optimal_actions / n_steps
    return avg_reward, optimal_action_percent

def run_demo():
    print("=== ε-Greedy Multi-Armed Bandit Demo ===")
    avg_reward, opt_pct = epsilon_greedy_bandit(epsilon=0.1)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Optimal action chosen: {opt_pct:.1f}% of the time")

if __name__ == "__main__":
    run_demo()
