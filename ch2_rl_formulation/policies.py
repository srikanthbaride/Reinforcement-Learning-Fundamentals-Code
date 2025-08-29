from __future__ import annotations
import numpy as np

def deterministic(actions: np.ndarray) -> np.ndarray:
    return actions.astype(int).copy()

def epsilon_greedy_from_Q(Q: np.ndarray, epsilon: float) -> np.ndarray:
    S, A = Q.shape
    probs = np.full((S, A), epsilon / A, dtype=float)
    greedy = np.argmax(Q, axis=1)
    probs[np.arange(S), greedy] += 1.0 - epsilon
    return probs
