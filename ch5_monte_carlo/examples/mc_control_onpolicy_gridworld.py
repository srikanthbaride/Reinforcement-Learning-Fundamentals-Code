import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4

__all__ = ["mc_control_onpolicy", "ACTIONS", "generate_episode_onpolicy"]

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

def _is_terminal(env: GridWorld4x4, s) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(env.is_terminal(s))
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    return st == env.goal

def _step(env: GridWorld4x4, s, a):
    """Robust step that does NOT depend on env.P; uses geometry."""
    if hasattr(env, "step"):
        return env.step(s, a)
    st = s if isinstance(s, tuple) else env.i2s[int(s)]
    i, j = st
    di, dj = ACTIONS[a]
    n = getattr(env, "n", int(round(len(env.S) ** 0.5)))
    ni, nj = i + di, j + dj
    if not (0 <= ni < n and 0 <= nj < n):
        ni, nj = i, j
    sp = (ni, nj)
    step_reward = float(getattr(env, "step_reward", -1.0))
    r = 0.0 if sp == getattr(env, "goal", (0, 3)) else step_reward
    return sp, r

def _epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    return int(rng.integers(len(q_row))) if rng.random() < epsilon else int(np.argmax(q_row))

def generate_episode_onpolicy(env: GridWorld4x4, Q: np.ndarray, epsilon: float,
                              rng: np.random.Generator, max_steps: int = 10000):
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
    G = 0.0
    returns = np.zeros(len(actions), dtype=float)
    for t in range(len(actions) - 1, -1, -1):
        G = rewards[t + 1] + gamma * G
        returns[t] = G
    return states[:-1], actions, returns

def mc_control_onpolicy(env: GridWorld4x4, episodes: int = 5000,
                        epsilon: float = 0.1, gamma: float | None = None,
                        seed: int | None = None):
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
            if (s_idx, a) in seen:
                continue
            seen.add((s_idx, a))
            G = returns[t]
            N[s_idx, a] += 1.0
            alpha = 1.0 / N[s_idx, a]
            Q[s_idx, a] += alpha * (G - Q[s_idx, a])

    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return Q, pi
