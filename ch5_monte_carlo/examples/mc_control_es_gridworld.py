# ch5_monte_carlo/examples/mc_control_es_gridworld.py
# Monte Carlo control with Exploring Starts (ES) on a 4x4 GridWorld.
# Robust: no reliance on env.P shape nor env.is_terminal presence.

from __future__ import annotations
import numpy as np

__all__ = ["mc_es_control", "generate_episode_es", "ACTIONS"]

# Tests expect ACTIONS to be action *indices* usable as env.P[s_idx][a] keys.
ACTIONS   = [0, 1, 2, 3]                      # exported for tests
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U (internal geometry)

def _goal(env): return getattr(env, "goal", (0, 3))
def _n(env):    return getattr(env, "n", int(round(len(env.S) ** 0.5)))
def _step_reward(env): return float(getattr(env, "step_reward", -1.0))

def _is_terminal(env, s) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(env.is_terminal(s))
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    return st == _goal(env)

def _step(env, s, a_idx: int):
    """Use env.step if present; else geometric fallback using DIRECTIONS."""
    if hasattr(env, "step"):
        return env.step(s, a_idx)
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    i, j = st
    di, dj = DIRECTIONS[a_idx]
    n = _n(env)
    ni, nj = i + di, j + dj
    if not (0 <= ni < n and 0 <= nj < n):
        ni, nj = i, j
    sp = (ni, nj)
    r = 0.0 if sp == _goal(env) else _step_reward(env)
    return sp, r

def _greedy_action(q_row: np.ndarray) -> int:
    return int(np.argmax(q_row))

def generate_episode_es(env, Q: np.ndarray, gamma: float, max_steps: int = 10_000):
    """
    Exploring starts: start from random non-terminal state & random action,
    then follow greedy policy w.r.t. Q.
    Returns aligned (states, actions, returns) of length T = #actions.
    """
    rng = np.random.default_rng()
    non_terminal = [s for s in env.S if not _is_terminal(env, s)]
    s = non_terminal[rng.integers(len(non_terminal))]
    a = int(rng.integers(len(env.A)))  # int action index

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

    # Compute returns over T = len(actions); guard rewards indexing just in case.
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

    # deterministic greedy policy over action indices
    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return Q, pi
