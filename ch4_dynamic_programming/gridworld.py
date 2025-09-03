from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

@dataclass
class GridWorld4x4:
    '''Deterministic 4x4 gridworld with an absorbing goal at (0, 3) by default.'''
    step_reward: float = -1.0
    goal: Tuple[int, int] = (0, 3)
    gamma: float = 1.0

    def __post_init__(self):
        self.n = 4
        self.S: List[Tuple[int, int]] = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.A = list(range(len(ACTIONS)))
        self.s2i = {s: i for i, s in enumerate(self.S)}
        self.i2s = {i: s for i, s in enumerate(self.S)}
        self.goal_idx = self.s2i[self.goal]

        S, A = len(self.S), len(self.A)
        self.P = np.zeros((S, A, S), dtype=float)
        self.R = np.zeros((S, A, S), dtype=float)

        for s_idx, (r, c) in enumerate(self.S):
            if s_idx == self.goal_idx:
                # Absorbing terminal: stay in place with reward 0
                self.P[s_idx, :, s_idx] = 1.0
                self.R[s_idx, :, s_idx] = 0.0
                continue
            for a_idx, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    s2_idx = self.s2i[(nr, nc)]
                else:
                    s2_idx = s_idx  # bump into wall
                self.P[s_idx, a_idx, s2_idx] = 1.0
                self.R[s_idx, a_idx, s2_idx] = self.step_reward

    @property
    def states(self):
        return list(range(len(self.S)))

    @property
    def actions(self):
        return list(range(len(ACTIONS)))

    def manhattan_to_goal(self, s_idx: int) -> int:
        r, c = self.i2s[s_idx]
        gr, gc = self.goal
        return abs(r - gr) + abs(c - gc)

