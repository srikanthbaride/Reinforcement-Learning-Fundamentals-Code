from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

ACTIONS: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

@dataclass(frozen=True)
class Transition:
    s: int
    a: int
    sp: int
    r: float
    p: float

class GridWorld4x4:
    """Deterministic 4x4 gridworld with an absorbing goal (reward 0)."""
    def __init__(self, step_reward: float = -1.0, goal: Tuple[int, int] = (0, 3)):
        self.n = 4
        self.step_reward = float(step_reward)
        self.goal = tuple(goal)

        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s2i = {s: k for k, s in enumerate(self.S)}  # (i,j) -> idx
        self.i2s = {k: s for k, s in enumerate(self.S)}  # idx   -> (i,j)
        self.A = list(range(len(ACTIONS)))

        self.terminal = self.s2i[self.goal]
        self.num_states = len(self.S)
        self.num_actions = len(self.A)

        self.P: Dict[int, Dict[int, List[Transition]]] = self._build_P()

    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    def _step_det(self, s_idx: int, a: int) -> Tuple[int, float]:
        if s_idx == self.terminal:
            return s_idx, 0.0  # absorbing
        i, j = self.i2s[s_idx]
        di, dj = ACTIONS[a]
        ni, nj = i + di, j + dj
        if not self._in_bounds(ni, nj):
            ni, nj = i, j  # bounce to self
        sp_idx = self.s2i[(ni, nj)]
        r = 0.0 if sp_idx == self.terminal else self.step_reward
        return sp_idx, r

    def _build_P(self) -> Dict[int, Dict[int, List[Transition]]]:
        P: Dict[int, Dict[int, List[Transition]]] = {}
        for s in range(self.num_states):
            P[s] = {}
            for a in self.A:
                sp, r = self._step_det(s, a)
                P[s][a] = [Transition(s=s, a=a, sp=sp, r=r, p=1.0)]
        return P

