from __future__ import annotations
import numpy as np

ACTIONS = ['U','R','D','L']  # up, right, down, left
A_DELTA = {'U':(-1,0), 'R':(0,1), 'D':(1,0), 'L':(0,-1)}

def make_gridworld(n: int = 4, step_reward: float = -1.0, terminal: tuple[int,int] | None = None):
    """Deterministic gridworld (n×n). Terminal default is (0, n-1)."""
    if terminal is None:
        terminal = (0, n-1)
    S = n*n
    A = len(ACTIONS)
    P = np.zeros((S, A, S), dtype=float)
    R = np.full((S, A, S), 0.0, dtype=float)

    def idx(i,j): return i*n + j
    term_idx = idx(*terminal)

    for i in range(n):
        for j in range(n):
            s = idx(i,j)
            for a_id, a in enumerate(ACTIONS):
                if s == term_idx:
                    P[s, a_id, s] = 1.0
                    R[s, a_id, s] = 0.0
                    continue
                di, dj = A_DELTA[a]
                ni, nj = i+di, j+dj
                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    ns = s  # bump into wall
                else:
                    ns = idx(ni, nj)
                P[s, a_id, ns] = 1.0
                R[s, a_id, ns] = step_reward if ns != term_idx else 0.0
    states = list(range(S))
    actions = list(range(A))
    return states, actions, P, R, (n, terminal, term_idx)

def unravel_index(s: int, n: int):
    return (s // n, s % n)

def arrows_from_policy(pi):
    """Convert one-hot deterministic policy (S×A) to symbol grid of U/R/D/L."""
    idx = np.argmax(pi, axis=1)
    return np.array([['U','R','D','L'][k] for k in idx])
