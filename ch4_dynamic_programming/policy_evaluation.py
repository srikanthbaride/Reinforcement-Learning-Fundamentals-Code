from __future__ import annotations
import numpy as np

def policy_evaluation(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                      gamma: float = 1.0, theta: float = 1e-8, max_sweeps: int = 10_000) -> np.ndarray:
    '''Iterative policy evaluation (Bellman expectation updates).'''
    nS, nA, _ = P.shape
    V = np.zeros(nS, dtype=float)
    for _ in range(max_sweeps):
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            q = 0.0
            for a in range(nA):
                if pi[s, a] == 0.0:
                    continue
                q += pi[s, a] * np.sum(P[s, a] * (R[s, a] + gamma * V))
            V[s] = q
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V
