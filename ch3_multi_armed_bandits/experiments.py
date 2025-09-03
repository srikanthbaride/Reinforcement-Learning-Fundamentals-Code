import argparse, os, numpy as np, matplotlib.pyplot as plt
from .epsilon_greedy import run as run_eps
from .ucb import run as run_ucb
from .thompson import run as run_ts

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--T", type=int, default=5000)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="ch3_multi_armed_bandits/plots")
    return p.parse_args()

def make_true_means(K, rng): return rng.uniform(0.1, 0.9, size=K)

def run_all(true_means, T, trials, eps, c, seed):
    rng = np.random.default_rng(seed)
    avg_regret = {"eps": np.zeros(T), "ucb": np.zeros(T), "ts": np.zeros(T)}
    for _ in range(trials):
        s = int(rng.integers(0, 2**31-1))
        avg_regret["eps"] += run_eps(true_means, eps, T, s)["cum_regret"]
        avg_regret["ucb"] += run_ucb(true_means, c, T, s)["cum_regret"]
        avg_regret["ts"]  += run_ts(true_means, T, s)["cum_regret"]
    for k in avg_regret: avg_regret[k] /= trials
    return avg_regret

def plot(xs, series, ylabel, title, outpath):
    plt.figure()
    for label,y in series: plt.plot(xs,y,label=label)
    plt.xlabel("Time"); plt.ylabel(ylabel); plt.title(title); plt.legend()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300); plt.close()

def main():
    a = parse_args()
    true_means = make_true_means(a.K, np.random.default_rng(a.seed))
    xs = np.arange(1, a.T+1)
    reg = run_all(true_means,a.T,a.trials,a.eps,a.c,a.seed)
    plot(xs,[("ε-Greedy",reg["eps"]),("UCB1",reg["ucb"]),("Thompson",reg["ts"])],
         "Cumulative Regret","Regret vs Time",os.path.join(a.outdir,"regret.png"))
if __name__=="__main__": main()
