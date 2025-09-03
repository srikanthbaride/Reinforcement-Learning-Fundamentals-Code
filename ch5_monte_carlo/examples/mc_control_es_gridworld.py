# ch5_monte_carlo/examples/mc_control_es_gridworld.py
import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4

__all__ = ["mc_es_control", "generate_episode_es"]

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R,L,D,U (must match ch4 env)

def _greedy_action(q_row: np.ndarray) -> int:
    return int(np.argmax(q_row))

def generate_episode_es(env: GridWorld4x4, Q: np.ndarray, gamma: float, max_steps: int = 10000):
    """
    Exploring starts: start from a random non-terminal state and random action,
    then follow greedy policy w.r.t. Q thereafter. Returns (states, actions, returns).
    """
    rng = np.random.default_rng()
    non_terminal = [s for s in env.S if not env.is_terminal(s)]
    s = non_terminal[rng.integers(len(non_terminal))]
    a = int(rng.integers(len(env.A)))

    states = [s]
    actions = [a]
    rewards = [0.0]  # so rewards[t+1] aligns with action taken at t

    steps = 0
    while not env.is_terminal(s) and steps < max_steps:
        sp, r = env.step(s, a)
        rewards.append(float(r))
        s = sp
        if env.is_terminal(s):
            break
        s_idx = env.s2i[s]
        a = _greedy_action(Q[s_idx])
        states.append(s)
        actions.append(a)
        steps += 1

    # first-visit returns
    G = 0.0
    returns = np.zeros(len(states), dtype=float)
    for t in range(len(states) - 1, -1, -1):
        G = rewards[t + 1] + gamma * G
        returns[t] = G
    return states, actions, returns

def mc_es_control(env: GridWorld4x4, episodes: int = 1500, gamma: float | None = None, seed: int | None = None):
    """
    On-policy Monte Carlo control with Exploring Starts (ES).
    Returns:
        Q:  (S,A) action-value table
        pi: (S,A) deterministic greedy policy derived from Q
    """
    if seed is not None:
        np.random.seed(seed)
    if gamma is None:
        gamma = float(env.gamma)

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

    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return Q, pi

# Optional: run this file directly for a quick check
if __name__ == "__main__":
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    Q, pi = mc_es_control(env, episodes=2000, seed=0)
    start = env.s2i[(0, 0)]
    print("Q(start):", Q[start])
    print("Greedy action at start:", int(np.argmax(pi[start])))
