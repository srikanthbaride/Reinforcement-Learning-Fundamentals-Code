from __future__ import annotations
import numpy as np

class ThompsonSamplingBernoulli:
    def __init__(self, K: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, rng: np.random.Generator | None = None):
        self.K = K
        self.rng = rng or np.random.default_rng()
        self.alpha = np.full(K, float(prior_alpha), dtype=float)
        self.beta = np.full(K, float(prior_beta), dtype=float)

    def select_arm(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        if reward >= 1.0 - 1e-12:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
