from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass(frozen=True)
class Pos:
    r: int
    c: int

class GridWorld:
    """Deterministic 4x4 GridWorld with a single terminal goal at (0, 3)."""
    def __init__(self, n_rows: int = 4, n_cols: int = 4, goal: Tuple[int,int]=(0,3), step_cost: float=-1.0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.goal = Pos(*goal)
        self.step_cost = step_cost
        # actions encoded as (dr, dc)
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.S = [Pos(r, c) for r in range(n_rows) for c in range(n_cols)]
        self.A = list(self.actions.keys())

    def is_terminal(self, s: Pos) -> bool:
        return s == self.goal

    def step(self, s: Pos, a: int) -> Tuple[Pos, float]:
        if self.is_terminal(s):
            return s, 0.0
        dr, dc = self.actions[a]
        nr, nc = s.r + dr, s.c + dc
        # stay in bounds
        nr = min(max(nr, 0), self.n_rows - 1)
        nc = min(max(nc, 0), self.n_cols - 1)
        s_next = Pos(nr, nc)
        r = 0.0 if self.is_terminal(s_next) else self.step_cost
        return s_next, r

    def index(self, s: Pos) -> int:
        return s.r * self.n_cols + s.c

    def neighbors(self, s: Pos) -> Dict[int, Pos]:
        out = {}
        for a in self.A:
            ns, _ = self.step(s, a)
            out[a] = ns
        return out
