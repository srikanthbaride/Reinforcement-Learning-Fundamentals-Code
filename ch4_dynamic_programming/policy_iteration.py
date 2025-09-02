from __future__ import annotations
import numpy as np
from .policy_evaluation import policy_evaluation
from .utils import greedy_policy_from_V

def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 1.0, theta: float = 1e-8,
                     max_outer_loops: int = 1_000) -> tuple[np.ndarray, np.ndarray, int]:
    '''Howard's Policy Iteration. Returns (pi_star, V_star, num_improvements).'''
    nS, nA, _ = P.shape
    pi = np.ones((nS, nA), dtype=float) / nA
    improvements = 0
    for _ in range(max_outer_loops):
        V = policy_evaluation(P, R, pi, gamma=gamma, theta=theta)
        pi_new = greedy_policy_from_V(P, R, V, gamma=gamma)
        if np.array_equal(pi_new, pi):
            return pi_new, V, improvements
        pi = pi_new
        improvements += 1
    return pi, V, improvements
