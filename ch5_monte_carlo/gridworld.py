# ch5_monte_carlo/gridworld.py
import numpy as np

# Internal direction deltas (used only inside this module)
_DELTAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

class GridWorld4x4:
    """
    4x4 deterministic GridWorld with one terminal goal.
    Exposes both model-based (P) and simulator (reset/step) APIs.
      - States are (row, col) tuples.
      - P[s_idx][a] -> list[(prob, sp_idx, reward)]  (deterministic here)
      - step(a) returns (next_state_tuple, reward, done)
      - Reward: step_reward each step, plus +1 when entering the goal.
    """
    def __init__(self, step_reward=0.0, goal=(0, 3)):
        self.n = 4
        self.goal = tuple(goal)
        self.step_reward = float(step_reward)

        # State indexing
        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s2i = {s: i for i, s in enumerate(self.S)}
        self.i2s = {i: s for s, i in self.s2i.items()}

        # Actions are indices 0..3
        self.A = list(range(len(_DELTAS)))

        # Episode bookkeeping for step/reset
        self._start = (self.n - 1, 0)  # bottom-left
        self._state = None

        # Tabular model P[s_idx][a] = [(1.0, sp_idx, r)]
        self.P = [[] for _ in range(len(self.S))]
        for s_idx, s in enumerate(self.S):
            self.P[s_idx] = [[] for _ in self.A]
            for a in self.A:
                sp, r = self._next_state_reward(s, a)
                sp_idx = self.s2i[sp]
                self.P[s_idx][a] = [(1.0, sp_idx, r)]

    # ---------- simulation API ----------

    def reset(self):
        self._state = self._start
        return self._state

    def step(self, a):
        """Return (next_state_tuple, reward, done)."""
        assert a in self.A
        s = self._state
        if s == self.goal:
            return s, 0.0, True
        sp, r = self._next_state_reward(s, a)
        done = (sp == self.goal)
        self._state = sp
        return sp, r, done

    # ---------- helpers ----------

    def _in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    def _next_state_reward(self, s, a):
        """Deterministic transition used for both P and step()."""
        di, dj = _DELTAS[a]
        ni, nj = s[0] + di, s[1] + dj
        if not self._in_bounds(ni, nj):
            ni, nj = s  # bounce off walls
        sp = (ni, nj)
        # +1 bonus when entering goal
        r = self.step_reward + (1.0 if sp == self.goal else 0.0)
        return sp, r
