from __future__ import annotations
import numpy as np
from .gridworld import GridWorld4x4

def policy_evaluation(env: GridWorld4x4,
                      pi_actions: np.ndarray,
                      gamma: float = 0.9,
                      theta: float = 1e-8,
                      max_iter: int = 10000) -> np.ndarray:
    S = env.num_states
    V = np.zeros(S, dtype=float)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            a = int(pi_actions[s])
            v_new = sum(tr.p * (tr.r + gamma * V[tr.sp]) for tr in env.P[s][a])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V

def policy_evaluation_stochastic(env: GridWorld4x4,
                                 pi_probs: np.ndarray,
                                 gamma: float = 0.9,
                                 theta: float = 1e-8,
                                 max_iter: int = 10000) -> np.ndarray:
    S, A = env.num_states, env.num_actions
    V = np.zeros(S, dtype=float)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            v_new = 0.0
            for a in range(A):
                pa = pi_probs[s, a]
                if pa == 0.0:
                    continue
                v_new += pa * sum(tr.p * (tr.r + gamma * V[tr.sp]) for tr in env.P[s][a])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V

def q_from_v(env: GridWorld4x4, V: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    S, A = env.num_states, env.num_actions
    Q = np.zeros((S, A), dtype=float)
    for s in range(S):
        for a in range(A):
            Q[s, a] = sum(tr.p * (tr.r + gamma * V[tr.sp]) for tr in env.P[s][a])
    return Q

def greedy_from_q(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1)
