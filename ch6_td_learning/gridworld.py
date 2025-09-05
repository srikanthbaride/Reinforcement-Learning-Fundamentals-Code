# ch6_td_learning/gridworld.py
from typing import Tuple, Any

class ThreeStateChainEnv:
    """
    Deterministic chain used in Chapter 6:
      A -> B -> C -> T
    Rewards: 0, 0, +1 (on entering T). Episode terminates at T.
    """
    def __init__(self) -> None:
        self.states = ("A", "B", "C", "T")
        self._s = None

    def reset(self) -> str:
        self._s = "A"
        return self._s

    def step(self, action: Any = None) -> Tuple[str, float, bool, dict]:
        assert self._s is not None, "Call reset() before step()."
        s = self._s
        if s == "A":
            s_next, r, done = "B", 0.0, False
        elif s == "B":
            s_next, r, done = "C", 0.0, False
        elif s == "C":
            s_next, r, done = "T", 1.0, True
        else:
            s_next, r, done = "T", 0.0, True
        self._s = s_next
        return s_next, r, done, {}

# --- Alias to match tests / examples expecting this name ---
Chain3Env = ThreeStateChainEnv
