import numpy as np
from collections import namedtuple

# Actions: Right, Left, Down, Up
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
TR = namedtuple("TR", ["p", "sp", "r", "done"])  # transition record

class GridWorld4x4:
    """4x4 GridWorld with step cost on every move, including terminal entry.

    Convention:
    - States are indexed 0..15 in row-major order; i2s maps index->(i,j).
    - The goal is absorbing: once there, any action keeps you there with reward 0.
    - Each attempted move costs `step_reward` (e.g., -1), including the final move into the goal.
    - Bumping into a wall leaves you in place and still incurs the step cost.
    - Transition kernel P is indexed by state index: P[s][a] -> [TR(p, sp, r, done)].
    """

    def __init__(self, step_reward: float = -1.0, goal=(0, 3)):
        self.n = 4
        self.goal = tuple(goal)
        self.step_reward = float(step_reward)

        # Enumerate states as (i,j) tuples; provide index mappings
        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s2i = {s: k for k, s in enumerate(self.S)}
        self.i2s = {k: s for s, k in self.s2i.items()}

        self.A = list(range(len(ACTIONS)))  # 0..3
        self._goal_idx = self.s2i[self.goal]

        # Transition kernel with integer state indices
        self.P = {s_idx: {a: [] for a in self.A} for s_idx in range(self.num_states)}
        self._build_P()

    # --- properties expected by tests ---
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
            return (i, j)  # bump into wall: stay
        return (ni, nj)

    def _build_P(self):
        goal_idx = self._goal_idx

        for s_idx in range(self.num_states):
            s_tuple = self.i2s[s_idx]

            if s_idx == goal_idx:
                # Absorbing terminal: reward 0 thereafter
                for a in self.A:
                    self.P[s_idx][a] = [TR(1.0, goal_idx, 0.0, True)]
                continue

            for a in self.A:
                next_tuple = self._next_state_tuple(s_tuple, a)
                sp_idx = self.s2i[next_tuple]
                done = (sp_idx == goal_idx)
                # Charge step cost even when entering terminal
                r = self.step_reward
                self.P[s_idx][a] = [TR(1.0, sp_idx, r, done)]

    # Optional index-based step (used by some examples/tests)
    def step_idx(self, s_idx: int, a: int):
        trans = self.P[s_idx][a]
        if len(trans) == 1:
            tr = trans[0]
        else:
            probs = [t.p for t in trans]
            idx = np.random.choice(len(trans), p=probs)
            tr = trans[idx]
        return tr.sp, tr.r, tr.done
