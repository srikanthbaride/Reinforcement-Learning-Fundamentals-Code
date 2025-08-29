from __future__ import annotations
import numpy as np
from typing import Sequence, Optional

class MultiArmedBanditBernoulli:
    """K-armed bandit with Bernoulli rewards.
    probs: list/array of success probabilities for each arm (0..1).
    reward is 0/1. RNG can be injected for reproducibility.
    """
    def __init__(self, probs: Sequence[float], rng: Optional[np.random.Generator] = None):
        self.probs = np.asarray(probs, dtype=float)
        assert np.all((0 <= self.probs) & (self.probs <= 1)), "probs must be in [0,1]"
        self.k = len(self.probs)
        self.rng = rng if rng is not None else np.random.default_rng()

    def pull(self, arm: int) -> int:
        p = self.probs[arm]
        return int(self.rng.random() < p)

    def best_arm(self) -> int:
        return int(np.argmax(self.probs))

    def optimal_mean(self) -> float:
        return float(np.max(self.probs))
