from __future__ import annotations
import numpy as np

class EpsilonGreedy:
    def __init__(self, K: int, epsilon: float = 0.1, rng: np.random.Generator | None = None):
        self.K = K
        self.epsilon = float(epsilon)
        self.rng = rng or np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.values = np.zeros(K, dtype=float)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.K))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
