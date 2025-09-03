from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

@dataclass
class BernoulliBandit:
    p: Iterable[float]
    seed: Optional[int] = None

    def __post_init__(self):
        self.p = np.asarray(list(self.p), dtype=float)
        if np.any(self.p < 0) or np.any(self.p > 1):
            raise ValueError("All probabilities must be in [0,1].")
        self.K = int(self.p.size)
        self._rng = np.random.default_rng(self.seed)

    def step(self, arm: int) -> int:
        if not (0 <= arm < self.K):
            raise IndexError("Arm index out of range.")
        return int(self._rng.random() < self.p[arm])

    def reset(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

def regret_from_choices(true_means: np.ndarray, choices: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    mu_star = float(np.max(true_means))
    t = np.arange(1, rewards.size + 1, dtype=float)
    return mu_star * t - np.cumsum(rewards)

def ensure_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)
