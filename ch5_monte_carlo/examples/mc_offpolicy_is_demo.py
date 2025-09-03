# ch5_monte_carlo/examples/mc_offpolicy_is_demo.py
"""
Toy off-policy Monte Carlo evaluation with ordinary vs weighted IS.

Setting:
- One-step episodic problem from state s.
- Behavior policy b:  b(a1|s)=0.8, b(a2|s)=0.2
- Target  policy pi: pi(a1|s)=0.2, pi(a2|s)=0.8
- Rewards: P(R=1|a1)=0.3, P(R=1|a2)=0.7   (Bernoulli)
True value under pi: v_pi(s) = 0.2*0.3 + 0.8*0.7 = 0.62
"""

from __future__ import annotations
import numpy as np

BEHAVIOR = {"a1": 0.8, "a2": 0.2}
TARGET   = {"a1": 0.2, "a2": 0.8}
P_REWARD = {"a1": 0.3, "a2": 0.7}
ACTIONS  = ("a1", "a2")
TRUE_V   = 0.62

def sample_behavior_episode(rng: np.random.Generator) -> tuple[str, float]:
    """Draw action from behavior b, sample Bernoulli reward; one-step episode."""
    a = rng.choice(ACTIONS, p=[BEHAVIOR["a1"], BEHAVIOR["a2"]])
    r = float(rng.random() < P_REWARD[a])
    return a, r

def is_weight(a: str) -> float:
    """Importance ratio pi(a|s)/b(a|s)."""
    return TARGET[a] / BEHAVIOR[a]

def offpolicy_mc(N: int = 1000, seed: int | None = 0) -> tuple[float, float]:
    """
    Return (OIS, WIS) estimates for v_pi(s) using N i.i.d. episodes from b.
    OIS: ordinary IS (unbiased, higher variance)
    WIS: weighted/normalized IS (biased finite-N, lower variance)
    """
    rng = np.random.default_rng(seed)
    ws, wr = [], []
    for _ in range(N):
        a, r = sample_behavior_episode(rng)
        w = is_weight(a)
        ws.append(w)
        wr.append(w * r)
    ws = np.asarray(ws, dtype=float)
    wr = np.asarray(wr, dtype=float)
    ois = wr.mean()
    wis = wr.sum() / ws.sum()
    return float(ois), float(wis)

def monte_carlo_stats(N: int = 300, trials: int = 200, seed: int | None = 0):
    """Run multiple trials to observe the variance of OIS vs WIS."""
    rng = np.random.default_rng(seed)
    ois, wis = [], []
    for _ in range(trials):
        s = int(rng.integers(0, 2**31 - 1))
        est_ois, est_wis = offpolicy_mc(N=N, seed=s)
        ois.append(est_ois)
        wis.append(est_wis)
    ois = np.asarray(ois)
    wis = np.asarray(wis)
    return {
        "N": N,
        "trials": trials,
        "true_v": TRUE_V,
        "ois_mean": float(ois.mean()),
        "ois_var": float(ois.var(ddof=1)),
        "wis_mean": float(wis.mean()),
        "wis_var": float(wis.var(ddof=1)),
    }

if __name__ == "__main__":
    # Demo: single run
    ois, wis = offpolicy_mc(N=1000, seed=0)
    print(f"Single-run estimates (N=1000): OIS={ois:.4f}, WIS={wis:.4f}, True={TRUE_V:.4f}")

    # Variance comparison across many trials
    stats = monte_carlo_stats(N=300, trials=500, seed=1)
    print(
        f"[Variance over {stats['trials']} trials, N={stats['N']}] "
        f"OIS mean={stats['ois_mean']:.4f}, var={stats['ois_var']:.5f} | "
        f"WIS mean={stats['wis_mean']:.4f}, var={stats['wis_var']:.5f} | True={stats['true_v']:.4f}"
    )

