from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from .bandits import BernoulliBandit, regret_from_choices, ensure_rng

def run(true_means, c: float, steps: int, seed: Optional[int] = None) -> Dict[str, Any]:
    if c <= 0: raise ValueError("c must be > 0.")
    K = len(true_means)
    if steps < K: raise ValueError("steps must be >= K.")
    env = BernoulliBandit(true_means, seed=seed)
    Q, N = np.zeros(K), np.zeros(K, int)
    choices, rewards = np.zeros(steps, int), np.zeros(steps, float)
    for a in range(K):
        r = env.step(a); Q[a], N[a] = r, 1; choices[a], rewards[a] = a, r
    for t in range(K, steps):
        ucb = Q + c * np.sqrt(np.log(t + 1) / N)
        a = int(np.argmax(ucb)); r = env.step(a)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]
        choices[t], rewards[t] = a, r
    return {
        "rewards": rewards, "choices": choices, "Q": Q, "N": N,
        "cum_regret": regret_from_choices(np.asarray(true_means, float), choices, rewards),
    }
