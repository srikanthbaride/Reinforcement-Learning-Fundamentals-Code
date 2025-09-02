from __future__ import annotations
import numpy as np

def uniform_random_policy(nS: int, nA: int) -> np.ndarray:
    return np.ones((nS, nA), dtype=float) / nA

def greedy_policy_from_V(P: np.ndarray, R: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    nS, nA, _ = P.shape
    pi = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        q = [np.sum(P[s, a] * (R[s, a] + gamma * V)) for a in range(nA)]
        a_star = int(np.argmax(q))
        pi[s] = 0.0
        pi[s, a_star] = 1.0
    return pi

def best_action(P: np.ndarray, R: np.ndarray, V: np.ndarray, s: int, gamma: float) -> int:
    q = [np.sum(P[s, a] * (R[s, a] + gamma * V)) for a in range(P.shape[1])]
    return int(np.argmax(q))
