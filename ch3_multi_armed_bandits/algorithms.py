from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EpsilonGreedy:
    n_arms: int
    epsilon: float = 0.1
    init: float = 0.0
    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.full(self.n_arms, float(self.init), dtype=float)
        self.rng = np.random.default_rng()
    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

@dataclass
class UCB1:
    n_arms: int
    c: float = 2.0
    def __post_init__(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.total = 0
        self.rng = np.random.default_rng()
    def select_arm(self) -> int:
        # pull each arm once
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        ucb = self.values + self.c * np.sqrt(np.log(self.total) / self.counts)
        return int(np.argmax(ucb))
    def update(self, arm: int, reward: float):
        self.total += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

@dataclass
class ThompsonSamplingBeta:
    n_arms: int
    a0: float = 1.0
    b0: float = 1.0
    def __post_init__(self):
        self.a = np.full(self.n_arms, self.a0, dtype=float)
        self.b = np.full(self.n_arms, self.b0, dtype=float)
        self.rng = np.random.default_rng()
    def select_arm(self) -> int:
        samples = self.rng.beta(self.a, self.b)
        return int(np.argmax(samples))
    def update(self, arm: int, reward: float):
        self.a[arm] += reward
        self.b[arm] += 1 - reward

def simulate(env, agent, steps: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run interaction loop and return history stats."""
    if seed is not None:
        try:
            agent.rng = np.random.default_rng(seed)
        except AttributeError:
            pass
        if hasattr(env, 'rng'):
            env.rng = np.random.default_rng(seed+1 if seed is not None else None)
    rewards = np.zeros(steps, dtype=float)
    pulls = np.zeros(env.k, dtype=int)
    choices = np.zeros(steps, dtype=int)
    for t in range(steps):
        a = agent.select_arm()
        r = env.pull(a)
        agent.update(a, r)
        rewards[t] = r
        pulls[a] += 1
        choices[t] = a
    return {
        "avg_reward": float(rewards.mean()),
        "cum_reward": float(rewards.sum()),
        "pulls": pulls,
        "choices": choices,
        "rewards": rewards,
    }
