from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from .bandits import BernoulliBandit, regret_from_choices, ensure_rng

def run(true_means, epsilon: float, steps: int, seed: Optional[int] = None) -> Dict[str, Any]:
    if not (0 <= float(epsilon) <= 1):
        raise ValueError("epsilon must be in [0,1].")
    K = len(true_means)
    env = BernoulliBandit(true_means, seed=seed)
    rng = ensure_rng(seed)
    Q, N = np.zeros(K), np.zeros(K, dtype=int)
    choices, rewards = np.zeros(steps, int), np.zeros(steps, float)
    for t in range(steps):
        if rng.random() < epsilon:
            a = rng.integers(0, K)
        else:
            a = int(np.argmax(Q))
        r = env.step(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        choices[t], rewards[t] = a, r
    return {
        "rewards": rewards, "choices": choices, "Q": Q, "N": N,
        "cum_regret": regret_from_choices(np.asarray(true_means, float), choices, rewards),
    }
