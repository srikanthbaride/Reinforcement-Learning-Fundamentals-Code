import numpy as np

def ucb_bandit(n_arms=5, n_steps=200, c=2.0, seed=42):
    """
    UCB1-style demo for a Gaussian bandit with unit variance.
    true_means ~ N(0,1); reward ~ N(true_mean, 1).
    """
    rng = np.random.default_rng(seed)
    true_means = rng.normal(0, 1, n_arms)

    Q = np.zeros(n_arms)   # sample means
    N = np.zeros(n_arms)   # counts

    rewards = []
    optimal_actions = 0

    # Initialize: pull each arm once to avoid div-by-zero
    for a in range(n_arms):
        r = rng.normal(true_means[a], 1.0)
        Q[a] = r
        N[a] = 1.0
        rewards.append(r)
        if a == int(np.argmax(true_means)):
            optimal_actions += 1

    for t in range(n_arms, n_steps):
        # UCB score
        bonus = c * np.sqrt(2.0 * np.log(t + 1) / N)
        a = int(np.argmax(Q + bonus))

        r = rng.normal(true_means[a], 1.0)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards.append(r)
        if a == int(np.argmax(true_means)):
            optimal_actions += 1

    avg_reward = float(np.mean(rewards))
    optimal_action_percent = 100.0 * optimal_actions / n_steps
    return avg_reward, optimal_action_percent

def run_demo():
    print("=== UCB Bandit Demo (Gaussian rewards) ===")
    avg_reward, opt_pct = ucb_bandit(n_arms=5, n_steps=200, c=2.0)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Optimal action chosen: {opt_pct:.1f}% of the time")

if __name__ == "__main__":
    run_demo()
