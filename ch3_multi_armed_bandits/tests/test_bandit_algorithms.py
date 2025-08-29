import numpy as np
from ch3_multi_armed_bandits.bandit_env import MultiArmedBanditBernoulli
from ch3_multi_armed_bandits.algorithms import EpsilonGreedy, UCB1, ThompsonSamplingBeta, simulate

PROBS = [0.1, 0.2, 0.5, 0.4]
BEST = int(np.argmax(PROBS))
STEPS = 4000

def random_baseline_avg(probs, steps, seed=7):
    rng = np.random.default_rng(seed)
    k = len(probs)
    rewards = []
    for t in range(steps):
        a = int(rng.integers(k))
        r = int(rng.random() < probs[a])
        rewards.append(r)
    return float(np.mean(rewards))

def run_and_check(agent, steps=STEPS):
    env = MultiArmedBanditBernoulli(PROBS)
    stats = simulate(env, agent, steps, seed=42)
    return stats

def test_algorithms_beat_random_baseline():
    baseline = random_baseline_avg(PROBS, STEPS)
    agents = [
        EpsilonGreedy(n_arms=len(PROBS), epsilon=0.1),
        UCB1(n_arms=len(PROBS)),
        ThompsonSamplingBeta(n_arms=len(PROBS)),
    ]
    for agent in agents:
        stats = run_and_check(agent)
        # Should beat random baseline by a margin
        assert stats['avg_reward'] >= baseline + 0.05, (agent.__class__.__name__, stats['avg_reward'], baseline)

def test_learn_best_arm_frequently():
    agents = [
        EpsilonGreedy(n_arms=len(PROBS), epsilon=0.1),
        UCB1(n_arms=len(PROBS)),
        ThompsonSamplingBeta(n_arms=len(PROBS)),
    ]
    for agent in agents:
        stats = run_and_check(agent)
        pulls = stats['pulls']
        assert pulls[BEST] == pulls.max()  # best arm most selected
        # At least 50% of pulls go to best arm after learning
        assert pulls[BEST] >= STEPS * 0.5
