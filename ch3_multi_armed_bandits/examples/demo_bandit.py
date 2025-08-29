from ch3_multi_armed_bandits.bandit_env import MultiArmedBanditBernoulli
from ch3_multi_armed_bandits.algorithms import EpsilonGreedy, UCB1, ThompsonSamplingBeta, simulate

def main():
    probs = [0.1, 0.2, 0.5, 0.4]
    env = MultiArmedBanditBernoulli(probs)
    best = env.best_arm()
    steps = 5000

    for agent in [
        EpsilonGreedy(n_arms=len(probs), epsilon=0.1),
        UCB1(n_arms=len(probs)),
        ThompsonSamplingBeta(n_arms=len(probs)),
    ]:
        stats = simulate(env, agent, steps, seed=123)
        print(f"{agent.__class__.__name__}: avg_reward={stats['avg_reward']:.3f}, best_arm_pulled={stats['pulls'][best]} times" )

if __name__ == "__main__":
    main()
