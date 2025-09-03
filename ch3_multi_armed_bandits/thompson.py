from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from .bandits import BernoulliBandit, regret_from_choices, ensure_rng

def run(true_means, steps: int, seed: Optional[int] = None,
        alpha0: float = 1.0, beta0: float = 1.0) -> Dict[str, Any]:
    K = len(true_means)
    env = BernoulliBandit(true_means, seed=seed)
    rng = ensure_rng(seed)
    alpha, beta = np.full(K, alpha0), np.full(K, beta0)
    choices, rewards = np.zeros(steps, int), np.zeros(steps, float)
    for t in range(steps):
        theta = rng.beta(alpha, beta)
        a = int(np.argmax(theta)); r = env.step(a)
        alpha[a] += r; beta[a] += 1 - r
        choices[t], rewards[t] = a, r
    return {
        "rewards": rewards, "choices": choices,
        "alpha": alpha, "beta": beta,
        "cum_regret": regret_from_choices(np.asarray(true_means, float), choices, rewards),
    }
