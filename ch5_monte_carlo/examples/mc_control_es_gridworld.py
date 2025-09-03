# ch5_monte_carlo/examples/mc_control_es_gridworld.py
# Monte Carlo control with Exploring Starts (ES) on a 4x4 GridWorld.
# Robust to different GridWorld implementations: does not rely on env.P or env.is_terminal.

from __future__ import annotations
import numpy as np

__all__ = ["mc_es_control", "generate_episode_es", "ACTIONS"]

# Must match the environment's action ordering everywhere in the repo
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

# ------------ helpers that do not assume specific env attributes -------------

def _goal(env):
    return getattr(env, "goal", (0, 3))

def _n(env):
    # prefer env.n; otherwise infer from |S|
    return getattr(env, "n", int(round(len(env.S) ** 0.5)))

def _step_reward(env):
    return float(getattr(env, "step_reward", -1.0))

def _is_terminal(env, s) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(env.is_terminal(s))
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    return st == _goal(env)

def _step(env, s, a):
    """Robust step that uses env.step if available; else uses grid geometry."""
    if hasattr(env, "step"):
        return env.step(s, a)
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    i, j = st
    di, dj = ACTIONS[a]
    n = _n(env)
    ni, nj = i + di, j + dj
    if not (0 <= ni < n and 0 <= nj < n):
        ni, nj = i, j  # wall -> stay
    sp = (ni, nj)
    r = 0.0 if sp == _goal(env) else _step_reward(env)
    return sp, r

def _greedy_action(q_row: np.ndarray) -> int:
    return int(np.argmax(q_row))

# ------------------------------- core logic ----------------------------------

def generate_episode_es(env, Q: np.ndarray, gamma: float, max_steps: int = 10_000):
    """
    Exploring starts:
      - start from a random NON-terminal state
      - start with a random action
      - thereafter follow greedy policy w.r.t. Q
    Returns:
      states:  list of states (tuples), length T = number of actions
      actions: list of action indices, length T
      returns: list/array of returns G_t, length T
    """
    rng = np.random.default_rng()
    non_terminal = [s for s in env.S if not _is_terminal(env, s)]
    s = non_terminal[rng.integers(len(non_terminal))]
    a = int(rng.integers(len(env.A)))

    states = [s]
    actions = [a]
    rewards = [0.0]  # align indexing so rewards[t+1] corresponds to action at t

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

    # ---- returns over number of actions (T) ----
    T = len(actions)
    G = 0.0
    returns = np.zeros(T, dtype=float)
    for t in range(T - 1, -1, -1):
        G = rewards[t + 1] + gamma * G
        returns[t] = G
    # keep states length consistent with actions/returns
    return states[:T], actions, returns

def mc_es_control(env, episodes: int = 1500, gamma: float | None = None, seed: int | None = None):
    """
    On-policy Monte Carlo control with Exploring Starts (ES).
    Returns:
        Q:  (S,A) action-value table
        pi: (S,A) deterministic greedy policy derived from Q
    """
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
                continue  # first-visit MC
            seen.add(key)
            G = returns[t]
            N[s_idx, a] += 1.0
            alpha = 1.0 / N[s_idx, a]
            Q[s_idx, a] += alpha * (G - Q[s_idx, a])

    # greedy deterministic policy from Q
    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return Q, pi
