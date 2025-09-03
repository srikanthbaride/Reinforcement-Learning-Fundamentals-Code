import numpy as np
from collections import namedtuple

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
TR = namedtuple("TR", ["p", "sp", "r", "done"])

class GridWorld4x4:
    def __init__(self, step_reward: float = -1.0, goal=(0, 3)):
        self.n = 4
        self.goal = tuple(goal)
        self.step_reward = float(step_reward)

        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s2i = {s: k for k, s in enumerate(self.S)}
        self.i2s = {k: s for s, k in self.s2i.items()}
        self.A = list(range(len(ACTIONS)))

        self._goal_idx = self.s2i[self.goal]

        # Transition kernel with integer state indices
        self.P = {s_idx: {a: [] for a in self.A} for s_idx in range(self.num_states)}
        self._build_P()

    # --- properties ---
    @property
    def num_states(self) -> int:
        return len(self.S)

    @property
    def num_actions(self) -> int:
        return len(self.A)

    @property
    def terminal(self) -> int:
        """Index of the terminal (goal) state."""
        return self._goal_idx

    # --- helpers ---
    def _in_bounds(self, i, j) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    def _next_state_tuple(self, s_tuple, a):
        di, dj = ACTIONS[a]
        i, j = s_tuple
        ni, nj = i + di, j + dj
        if not self._in_bounds(ni, nj):
            return (i, j)
        return (ni, nj)

    def _build_P(self):
        goal_idx = self._goal_idx
        for s_idx in range(self.num_states):
            s_tuple = self.i2s[s_idx]
            if s_idx == goal_idx:
                for a in self.A:
                    self.P[s_idx][a] = [TR(1.0, goal_idx, 0.0, True)]
                continue

            for a in self.A:
                next_tuple = self._next_state_tuple(s_tuple, a)
                sp_idx = self.s2i[next_tuple]
                done = (sp_idx == goal_idx)
                self.P[s_idx][a] = [TR(1.0, sp_idx, self.step_reward, done)]
