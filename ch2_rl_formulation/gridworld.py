import numpy as np
from typing import List, Tuple

class GridWorld4x4:
    """
    4x4 deterministic GridWorld for Chapter 2.
    - Step reward: -1
    - Terminal goal: (0,3) top-right, value 0, no outgoing actions
    - Actions: up, right, down, left
    - Deterministic transitions; hitting a wall = self-transition
    """

    ACTIONS = ("up", "right", "down", "left")

    def __init__(
        self,
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        deterministic: bool = True,
        self_transition_on_wall: bool = True,
        goal: Tuple[int, int] = (0, 3),
    ):
        assert deterministic, "Only deterministic dynamics supported in this minimal build."
        self.h, self.w = 4, 4
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.self_transition_on_wall = bool(self_transition_on_wall)
        self.goal = tuple(goal)

        self._S = [(r, c) for r in range(self.h) for c in range(self.w)]
        self._A = list(self.ACTIONS)
        self._si = {s: i for i, s in enumerate(self._S)}
        self._ai = {a: i for i, a in enumerate(self._A)}

        # Precompute transition (P) and reward (R) tensors
        self._P = np.zeros((len(self._S), len(self._A), len(self._S)), dtype=float)
        self._R = np.zeros_like(self._P)
        self._build_PR()

    # --- Public API ---
    def states(self) -> List[Tuple[int, int]]:
        return list(self._S)

    def actions(self) -> List[str]:
        return list(self._A)

    def state_index(self, r: int, c: int) -> int:
        return self._si[(r, c)]

    def action_index(self, name: str) -> int:
        return self._ai[name]

    def P_tensor(self) -> np.ndarray:
        return self._P.copy()

    def R_tensor(self) -> np.ndarray:
        return self._R.copy()

    # --- Internal helpers ---
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def _next_state(self, s: Tuple[int, int], a: str) -> Tuple[int, int]:
        if s == self.goal:
            return s  # terminal
        r, c = s
        if a == "up":
            nr, nc = r - 1, c
        elif a == "right":
            nr, nc = r, c + 1
        elif a == "down":
            nr, nc = r + 1, c
        elif a == "left":
            nr, nc = r, c - 1
        else:
            raise ValueError(a)
        if self._in_bounds(nr, nc):
            return (nr, nc)
        return s if self.self_transition_on_wall else s

    def _build_PR(self):
        for si, s in enumerate(self._S):
            for ai, a in enumerate(self._A):
                s2 = self._next_state(s, a)
                s2i = self._si[s2]
                self._P[si, ai, s2i] = 1.0
                # Reward per step; terminal has value 0 so step reward applies on entry
                r = self.step_reward
                self._R[si, ai, s2i] = r
