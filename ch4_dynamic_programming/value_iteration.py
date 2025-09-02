from __future__ import annotations
import numpy as np

def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 1.0, theta: float = 1e-8,
                    max_sweeps: int = 10_000) -> tuple[np.ndarray, np.ndarray, int]:
    '''Value Iteration via the Bellman optimality operator. Returns (pi_star, V_star, num_sweeps).'''
    nS, nA, _ = P.shape
    V = np.zeros(nS, dtype=float)
    sweeps = 0
    for k in range(max_sweeps):
        sweeps = k + 1
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            q_vals = [np.sum(P[s, a] * (R[s, a] + gamma * V)) for a in range(nA)]
            V[s] = max(q_vals)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    # Greedy policy extraction
    pi = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        q_vals = [np.sum(P[s, a] * (R[s, a] + gamma * V)) for a in range(nA)]
        a_star = int(np.argmax(q_vals))
        pi[s, a_star] = 1.0
    return pi, V, sweeps
