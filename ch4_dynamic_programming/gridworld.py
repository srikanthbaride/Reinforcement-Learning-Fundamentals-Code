# ch4_dynamic_programming/gridworld.py
import numpy as np

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U (deterministic)

class GridWorld4x4:
    """
    4x4 deterministic GridWorld with a single terminal 'goal' state.
    Rewards are -1 per step and 0 upon entering/being in the goal.
    Transition model is encoded as tabular P[s,a,s'] and R[s,a,s'].
    """
    def __init__(self, step_reward: float = -1.0, goal=(0, 3), gamma: float = 1.0):
        self.n = 4
        self.goal = tuple(goal)
        self.step_reward = float(step_reward)
        self.gamma = float(gamma)

        # Tabular state/action spaces
        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s2i = {s: k for k, s in enumerate(self.S)}
        self.i2s = {k: s for s, k in self.s2i.items()}
        self.A = list(range(len(ACTIONS)))  # 4 actions

        # Build P and R
        self.P, self.R = self._build_PR()

    # -------- helpers --------
    def _in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    def _build_PR(self):
        S, A = len(self.S), len(self.A)
        P = np.zeros((S, A, S), dtype=float)
        R = np.full((S, A, S), self.step_reward, dtype=float)

        g_idx = self.s2i[self.goal]
        for s_idx, (i, j) in enumerate(self.S):
            for a_idx, (di, dj) in enumerate(ACTIONS):
                if s_idx == g_idx:
                    # absorbing terminal
                    P[s_idx, a_idx, s_idx] = 1.0
                    R[s_idx, a_idx, s_idx] = 0.0
                    continue

                ni, nj = i + di, j + dj
                if not self._in_bounds(ni, nj):
                    ni, nj = i, j  # bump into wall -> stay

                sp_idx = self.s2i[(ni, nj)]
                P[s_idx, a_idx, sp_idx] = 1.0
                if (ni, nj) == self.goal:
                    R[s_idx, a_idx, sp_idx] = 0.0  # no penalty entering goal
        return P, R

    # -------- environment API (used by ch5 as well) --------
    def is_terminal(self, s):
        return tuple(s) == self.goal

    def step(self, s, a):
        """Given state (tuple or index) and action index -> (next_state_tuple, reward)."""
        s_idx = self.s2i[s] if isinstance(s, tuple) else int(s)
        probs = self.P[s_idx, a]
        sp_idx = int(np.argmax(probs))  # deterministic
        r = float(self.R[s_idx, a, sp_idx])
        return self.i2s[sp_idx], r
