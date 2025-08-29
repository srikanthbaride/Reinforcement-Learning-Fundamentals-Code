import numpy as np
from ch3_multi_armed_bandits.bandits import BernoulliBandit
from ch3_multi_armed_bandits.epsilon_greedy import EpsilonGreedy
from ch3_multi_armed_bandits.ucb import UCB1
from ch3_multi_armed_bandits.thompson import ThompsonSamplingBernoulli
from ch3_multi_armed_bandits.experiments import run_algorithm

def _avg_final_regret(env, algo_ctor, T=1500, n_runs=80, base_seed=0):
    regs = []
    for r in range(n_runs):
        algo = algo_ctor()
        out = run_algorithm(env, algo, T=T, seed=base_seed + r)
        regs.append(out["cum_regret"][-1])
    return float(np.mean(regs))

def test_ordering_ucb_ts_vs_epsgreedy():
    env = BernoulliBandit(probs=np.array([0.2, 0.25, 0.3, 0.35, 0.5]))
    reg_eps = _avg_final_regret(env, lambda: EpsilonGreedy(K=env.K, epsilon=0.10))
    reg_ucb = _avg_final_regret(env, lambda: UCB1(K=env.K))  # default c=0.5
    reg_ts  = _avg_final_regret(env, lambda: ThompsonSamplingBernoulli(K=env.K))
    assert reg_ts  < reg_eps * 1.05
    assert reg_ucb < reg_eps * 1.05

def test_regret_increases_sublinearly_for_ucb_ts():
    env = BernoulliBandit(probs=np.array([0.1, 0.15, 0.2, 0.55]))
    for algo_ctor in [lambda: UCB1(K=env.K), lambda: ThompsonSamplingBernoulli(K=env.K)]:
        out_1k = _avg_final_regret(env, algo_ctor, T=1000, n_runs=60, base_seed=10)
        out_2k = _avg_final_regret(env, algo_ctor, T=2000, n_runs=60, base_seed=110)
        assert out_2k / 2000 < out_1k / 1000
