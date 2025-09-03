# ch4_dynamic_programming/policy_iteration.py
import numpy as np
from .utils import greedy_from_q

def policy_iteration(env, theta: float = 1e-8, max_eval_iter: int = 10000):
    """
    Standard policy iteration: alternate policy evaluation and greedy improvement.
    Returns:
        V: (S,), pi: (S,A) deterministic greedy policy
    """
    S, A = len(env.S), len(env.A)
    gamma = env.gamma

    # start with uniform random policy
    pi = np.full((S, A), 1.0 / A, dtype=float)
    V = np.zeros(S, dtype=float)

    stable = False
    iters = 0
    while not stable:
        # --- policy evaluation ---
        for _ in range(max_eval_iter):
            delta = 0.0
            for s in range(S):
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

        # --- policy improvement ---
        Q = np.zeros((S, A), dtype=float)
        for s in range(S):
            for a in range(A):
                Q[s, a] = (env.P[s, a] * (env.R[s, a] + gamma * V)).sum()
        new_pi = greedy_from_q(Q)
        stable = np.array_equal(new_pi, pi)
        pi = new_pi
        iters += 1

    return V, pi
