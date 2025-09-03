# ch4_dynamic_programming/utils.py
import numpy as np

def greedy_from_q(Q: np.ndarray) -> np.ndarray:
    """Return deterministic greedy policy π(s) as one-hot over actions."""
    S, A = Q.shape
    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return pi

def q_from_v(P: np.ndarray, R: np.ndarray, gamma: float, V: np.ndarray) -> np.ndarray:
    """Compute Q(s,a) = Σ_s' P(s,a,s') [ R(s,a,s') + γ V(s') ]."""
    # (S,A,S) * (S,) via broadcasting
    return (P * (R + gamma * V[None, None, :])).sum(axis=2)
