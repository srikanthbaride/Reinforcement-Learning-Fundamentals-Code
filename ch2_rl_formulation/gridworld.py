import numpy as np

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

class GridWorld4x4:
    def __init__(self, step_reward: float = -1.0, goal=(0, 3)):
        self.n = 4
        self.goal = tuple(goal)
        self.step_reward = float(step_reward)

        self.S = [(i, j) for i in range(self.n) for j in range(self.n)]
        self.s2i = {s: k for k, s in enumerate(self.S)}
        self.i2s = {k: s for s, k in self.s2i.items()}
        self.A = list(range(len(ACTIONS)))

        # Transition kernel: P[s][a] = [(prob, s_next, reward, done)]
        self.P = {s: {a: [] for a in self.A} for s in self.S}
        self._build_PR()

    # --- compatibility helpers expected by tests ---
    @property
    def num_states(self) -> int:
        return len(self.S)

    @property
    def num_actions(self) -> int:
        return len(self.A)

    # ---------------- internal helpers ----------------
    def _in_bounds(self, i, j):
        return 0 <= i < self.n and 0 <= j < self.n

    def _next_state(self, s, a):
        di, dj = ACTIONS[a]
        i, j = s
        ni, nj = i + di, j + dj
        if not self._in_bounds(ni, nj):
            return (i, j)  # bump: stay in place
        return (ni, nj)

    def _build_PR(self):
        """
        Convention:
        - Every attempted move costs step_reward (e.g., -1), INCLUDING the final move into goal.
        - Goal is absorbing: once there, any action yields (goal, 0, done=True).
        """
        goal = self.goal
        for s in self.S:
            if s == goal:
                for a in self.A:
                    self.P[s][a] = [(1.0, goal, 0.0, True)]
                continue

            for a in self.A:
                s_next = self._next_state(s, a)
                done = (s_next == goal)
                reward = self.step_reward  # charge cost even on final transition
                self.P[s][a] = [(1.0, s_next, reward, done)]

    # Optional simulator
    def step(self, s, a):
        trans = self.P[s][a]
        probs = [p for p, _, _, _ in trans]
        idx = np.random.choice(len(trans), p=probs)
        _, s_next, r, done = trans[idx]
        return s_next, r, done
