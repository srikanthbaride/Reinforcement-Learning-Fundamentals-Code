from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from .bandits import BernoulliBandit
from .epsilon_greedy import EpsilonGreedy
from .ucb import UCB1
from .thompson import ThompsonSamplingBernoulli

def run_algorithm(env, algo, T: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rewards = np.zeros(T, dtype=float)
    regret = np.zeros(T, dtype=float)
    for t in range(T):
        a = algo.select_arm()
        r = env.pull(a, rng)
        algo.update(a, r)
        rewards[t] = r
        regret[t] = env.pseudo_regret(a)
    return {
        "rewards": rewards,
        "cum_rewards": np.cumsum(rewards),
        "regret": regret,
        "cum_regret": np.cumsum(regret),
    }

def average_over_runs(env, algo_ctor, T: int, n_runs: int, base_seed: int = 0) -> dict:
    cum_regrets = []
    for run in range(n_runs):
        algo = algo_ctor()
        result = run_algorithm(env, algo, T, seed=base_seed + run)
        cum_regrets.append(result["cum_regret"])
    cum_regrets = np.array(cum_regrets)
    mean = cum_regrets.mean(axis=0)
    se = cum_regrets.std(axis=0, ddof=1) / np.sqrt(n_runs)
    return {"mean": mean, "se": se}

def plot_regret(curves: dict, title: str, fname: str | None):
    fig, ax = plt.subplots()
    for label, stats in curves.items():
        ax.plot(stats["mean"], label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Average cumulative pseudo-regret")
    ax.set_title(title)
    ax.legend()
    if fname:
        out_dir = os.path.dirname(fname)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(fname, bbox_inches="tight")
    else:
        plt.show()

def main():
    probs = np.array([0.2, 0.25, 0.3, 0.35, 0.5])
    env = BernoulliBandit(probs=probs)
    T = 2000
    n_runs = 200
    curves = {}
    curves["ε-greedy(0.10)"] = average_over_runs(env, lambda: EpsilonGreedy(env.K, 0.10), T, n_runs, 123)
    curves["ε-greedy(0.01)"] = average_over_runs(env, lambda: EpsilonGreedy(env.K, 0.01), T, n_runs, 223)
    curves["UCB1(c=0.5)"] = average_over_runs(env, lambda: UCB1(env.K, c=0.5), T, n_runs, 323)
    curves["Thompson (Beta-Bernoulli)"] = average_over_runs(env, lambda: ThompsonSamplingBernoulli(env.K), T, n_runs, 423)
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "plots", "regret_bernoulli.png")
    plot_regret(curves, "Multi-Armed Bandits: Average Cumulative Pseudo-Regret", out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
