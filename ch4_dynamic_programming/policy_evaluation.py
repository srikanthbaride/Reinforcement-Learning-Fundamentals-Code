# ch4_dynamic_programming/policy_evaluation.py
import numpy as np

def policy_evaluation(env, pi: np.ndarray, theta: float = 1e-8, max_iter: int = 10000):
    """
    Iterative policy evaluation for a given stationary (possibly stochastic) π.

    Args:
        env: GridWorld4x4 (must provide P, R, gamma, S, A)
        pi:  (S,A) array, rows sum to 1
    Returns:
        V: (S,) state-value under π
    """
    S, A = len(env.S), len(env.A)
    V = np.zeros(S, dtype=float)
    gamma = env.gamma

    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            # v(s) = Σ_a π(a|s) Σ_s' P(s,a,s') [ R + γ V(s') ]
            v_new = 0.0
            for a in range(A):
                pa = pi[s, a]
                if pa == 0.0:
                    continue
                v_new += pa * (env.P[s, a] * (env.R[s, a] + gamma * V)).sum()
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V
