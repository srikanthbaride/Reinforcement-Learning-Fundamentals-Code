# ch4_dynamic_programming/value_iteration.py
import numpy as np
from .utils import greedy_from_q

def value_iteration(env, theta: float = 1e-8, max_iter: int = 10000):
    """
    Value iteration with max backup. Returns optimal V and greedy π.
    """
    S, A = len(env.S), len(env.A)
    gamma = env.gamma
    V = np.zeros(S, dtype=float)

    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            q = np.zeros(A, dtype=float)
            for a in range(A):
                q[a] = (env.P[s, a] * (env.R[s, a] + gamma * V)).sum()
            v_new = np.max(q)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break

    # derive greedy policy
    Q = np.zeros((S, A), dtype=float)
    for s in range(S):
        for a in range(A):
            Q[s, a] = (env.P[s, a] * (env.R[s, a] + gamma * V)).sum()
    pi = greedy_from_q(Q)
    return V, pi
