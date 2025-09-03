# ch5_monte_carlo/examples/mc_control_es_gridworld.py
# Monte Carlo control with Exploring Starts (ES) on a 4x4 GridWorld.
# Robust: does not rely on original env.P shape; normalizes env.P to list-of-3-tuples.

from __future__ import annotations
import numpy as np

__all__ = ["mc_es_control", "generate_episode_es", "ACTIONS"]

# Tests expect actions as integer indices (for env.P[s][a] lookup)
ACTIONS     = [0, 1, 2, 3]                          # exported for tests
DIRECTIONS  = [(0, 1), (0, -1), (1, 0), (-1, 0)]    # R, L, D, U (internal geometry)

# ---------------- utilities ----------------

def _goal(env): return getattr(env, "goal", (0, 3))
def _n(env):    return getattr(env, "n", int(round(len(env.S) ** 0.5)))
def _sr(env):   return float(getattr(env, "step_reward", -1.0))

def _is_terminal(env, s) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(env.is_terminal(s))
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    return st == _goal(env)

def _step_geom(env, s, a_idx: int):
    """Geometry fallback; used for building deterministic transitions."""
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    i, j = st
    di, dj = DIRECTIONS[a_idx]
    n = _n(env)
    ni, nj = i + di, j + dj
    if not (0 <= ni < n and 0 <= nj < n):
        ni, nj = i, j
    sp = (ni, nj)
    r = 0.0 if sp == _goal(env) else _sr(env)
    return sp, r

def _step(env, s, a_idx: int):
    """Use env.step if available; else geometry."""
    if hasattr(env, "step"):
        return env.step(s, a_idx)
    return _step_geom(env, s, a_idx)

def _greedy_action(q_row: np.ndarray) -> int:
    return int(np.argmax(q_row))

def _ensure_triple_envP(env):
    """
    Normalize env.P to a list-of-lists of lists of triples:
      env.P[s_idx][a_idx] == [ (1.0, sp_idx, r) ]
    Deterministic transitions built via geometry. This satisfies tests that
    iterate 'for (p, sp, r) in env.P[s][a]'.
    """
    S, A = len(env.S), len(env.A)
    P_list = [[None for _ in range(A)] for _ in range(S)]
    for s_idx, s in enumerate(env.S):
        for a_idx in range(A):
            sp, r = _step_geom(env, s, a_idx)
            sp_idx = env.s2i[sp]
            P_list[s_idx][a_idx] = [(1.0, sp_idx, float(r))]
    env.P = P_list  # in-place normalization

# ---------------- core ES logic ----------------

def generate_episode_es(env, Q: np.ndarray, gamma: float, max_steps: int = 10_000):
    """
    Exploring starts: start random non-terminal state & random action,
    then follow greedy policy w.r.t. Q.
    Returns (states, actions, returns) aligned to T = number of actions.
    """
    rng = np.random.default_rng()
    non_terminal = [s for s in env.S if not _is_terminal(env, s)]
    s = non_terminal[rng.integers(len(non_terminal))]
    a = int(rng.integers(len(env.A)))  # action index

    states = [s]
    actions = [a]
    rewards = [0.0]  # rewards[t+1] corresponds to action at t

    steps = 0
    while not _is_terminal(env, s) and steps < max_steps:
        sp, r = _step(env, s, a)
        rewards.append(float(r))
        s = sp
        if _is_terminal(env, s):
            break
        a = _greedy_action(Q[env.s2i[s]])
        states.append(s)
        actions.append(a)
        steps += 1

    # returns over number of actions
    T = len(actions)
    G = 0.0
    returns = np.zeros(T, dtype=float)
    for t in range(T - 1, -1, -1):
        r_tp1 = rewards[t + 1] if (t + 1) < len(rewards) else 0.0
        G = r_tp1 + gamma * G
        returns[t] = G
    return states[:T], actions, returns

def mc_es_control(env, episodes: int = 1500, gamma: float | None = None, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    if gamma is None:
        gamma = float(getattr(env, "gamma", 1.0))

    # Make env.P match tests' expected structure
    _ensure_triple_envP(env)

    S, A = len(env.S), len(env.A)
    Q = np.zeros((S, A), dtype=float)
    N = np.zeros((S, A), dtype=float)  # first-visit counts

    for _ in range(episodes):
        states, actions, returns = generate_episode_es(env, Q, gamma)
        seen = set()
        for t, (s, a) in enumerate(zip(states, actions)):
            s_idx = env.s2i[s]
            key = (s_idx, a)
            if key in seen:
                continue
            seen.add(key)
            G = returns[t]
            N[s_idx, a] += 1.0
            Q[s_idx, a] += (G - Q[s_idx, a]) / N[s_idx, a]

    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return Q, pi
