import numpy as np

def thompson_gaussian_bandit(n_arms=5, n_steps=200, prior_var=1.0, seed=42):
    """
    Thompson Sampling for a Gaussian bandit with unit observation variance.
    Prior for each arm mean: Normal(0, prior_var).
    true_means ~ N(0,1); reward ~ N(true_mean, 1).
    Posterior (conjugate): Normal(mu_post, var_post), where
      var_post = 1 / (1/prior_var + n)
      mu_post  = var_post * (n * sample_mean)       (since prior mean = 0)
    We maintain sample_mean via Q and count via N.
    """
    rng = np.random.default_rng(seed)
    true_means = rng.normal(0, 1, n_arms)

    Q = np.zeros(n_arms)   # sample means of observed rewards
    N = np.zeros(n_arms)   # pull counts

    rewards = []
    optimal_actions = 0
    inv_prior_var = 1.0 / prior_var

    for t in range(n_steps):
        # Sample a posterior mean for each arm
        theta = np.empty(n_arms)
        for a in range(n_arms):
            n = N[a]
            if n == 0:
                # pure prior
                mu_post = 0.0
                var_post = 1.0 / inv_prior_var
            else:
                var_post = 1.0 / (inv_prior_var + n)
                mu_post = var_post * (n * Q[a])  # prior mean = 0
            theta[a] = rng.normal(mu_post, np.sqrt(var_post))

        a = int(np.argmax(theta))

        # Pull chosen arm
        r = rng.normal(true_means[a], 1.0)

        # Update running mean and count
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards.append(r)
        if a == int(np.argmax(true_means)):
            optimal_actions += 1

    avg_reward = float(np.mean(rewards))
    optimal_action_percent = 100.0 * optimal_actions / n_steps
    return avg_reward, optimal_action_percent

def run_demo():
    print("=== Thompson Sampling Bandit Demo (Gaussian rewards) ===")
    avg_reward, opt_pct = thompson_gaussian_bandit(n_arms=5, n_steps=200, prior_var=1.0)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Optimal action chosen: {opt_pct:.1f}% of the time")

if __name__ == "__main__":
    run_demo()
