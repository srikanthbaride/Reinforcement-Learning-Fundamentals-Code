# ch5_monte_carlo/examples/mc_control_onpolicy_gridworld.py
# On-policy MC control with ε-greedy behavior/target; returns an ε-soft dict policy.

from __future__ import annotations
import numpy as np

__all__ = ["mc_control_onpolicy", "ACTIONS", "generate_episode_onpolicy"]

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

def _goal(env): return getattr(env, "goal", (0, 3))
def _n(env):    return getattr(env, "n", int(round(len(env.S) ** 0.5)))
def _step_reward(env): return float(getattr(env, "step_reward", -1.0))

def _is_terminal(env, s) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(env.is_terminal(s))
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    return st == _goal(env)

def _step(env, s, a):
    if hasattr(env, "step"):
        return env.step(s, a)
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    i, j = st
    di, dj = ACTIONS[a]
    n = _n(env)
    ni, nj = i + di, j + dj
    if not (0 <= ni < n and 0 <= nj < n):
        ni, nj = i, j
    sp = (ni, nj)
    r = 0.0 if sp == _goal(env) else _step_reward(env)
    return sp, r

def _epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    return int(rng.integers(len(q_row))) if rng.random() < epsilon else int(np.argmax(q_row))

def generate_episode_onpolicy(env, Q: np.ndarray, epsilon: float,
                              rng: np.random.Generator, max_steps: int = 10_000):
    non_terminal = [s for s in env.S if not _is_terminal(env, s)]
    s = non_terminal[rng.integers(len(non_terminal))]

    states, actions, rewards = [s], [], [0.0]
    steps = 0
    while not _is_terminal(env, s) and steps < max_steps:
        a = _epsilon_greedy(Q[env.s2i[s]], epsilon, rng)
        actions.append(a)
        sp, r = _step(env, s, a)
        rewards.append(float(r))
        s = sp
        states.append(s)
        steps += 1

    gamma = float(getattr(env, "gamma", 1.0))
    T = len(actions)
    G = 0.0
    returns = np.zeros(T, dtype=float)
    for t in range(T - 1, -1, -1):
        G = rewards[t + 1] + gamma * G
        returns[t] = G
    return states[:T], actions, returns

def mc_control_onpolicy(env, episodes: int = 5000,
                        epsilon: float = 0.1, gamma: float | None = None,
                        seed: int | None = None):
    """
    Returns:
      Q: (S,A)
      pi_soft: dict mapping (state_tuple, action_tuple) -> probability
               (ε-soft, so tests can do pi_soft[(s, a_tup)])
    """
    rng = np.random.default_rng(seed)
    S, A = len(env.S), len(env.A)
    if gamma is None:
        gamma = float(getattr(env, "gamma", 1.0))

    Q = np.zeros((S, A), dtype=float)
    N = np.zeros((S, A), dtype=float)

    for _ in range(episodes):
        states, actions, returns = generate_episode_onpolicy(env, Q, epsilon, rng)
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

    # Build ε-soft dict policy keyed by (state_tuple, action_tuple)
    pi_soft = {}
    for s_idx, s in enumerate(env.S):
        a_star = int(np.argmax(Q[s_idx]))
        for a_idx, a_tup in enumerate(ACTIONS):
            prob = (1.0 - epsilon) if a_idx == a_star else 0.0
            prob += epsilon / A
            pi_soft[(s, a_tup)] = prob

    return Q, pi_soft
