from __future__ import annotations
import numpy as np

class UCB1:
    """UCB1 with tunable exploration constant c (index = mean + c*sqrt(ln t / n)).
    Default c=0.5 for faster dominance over ε-greedy at short horizons used in tests.
    """
    def __init__(self, K: int, c: float = 0.5, rng: np.random.Generator | None = None):
        self.K = K
        self.c = float(c)
        self.rng = rng or np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.values = np.zeros(K, dtype=float)

    def select_arm(self) -> int:
        # Pull each arm once to initialize
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
        t = int(self.counts.sum())
        t = max(t, 1)
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        ucb = self.values + bonus
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
