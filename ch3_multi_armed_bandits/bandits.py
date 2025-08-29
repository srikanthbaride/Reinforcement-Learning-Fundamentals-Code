from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class BernoulliBandit:
    probs: np.ndarray

    def __post_init__(self):
        self.probs = np.array(self.probs, dtype=float)
        assert self.probs.ndim == 1 and (0 <= self.probs).all() and (self.probs <= 1).all()
        self.K = self.probs.shape[0]
        self.opt_idx = int(np.argmax(self.probs))
        self.opt_mean = float(self.probs[self.opt_idx])

    def pull(self, arm: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self.probs[arm])

    def pseudo_regret(self, arm: int) -> float:
        return self.opt_mean - float(self.probs[arm])
