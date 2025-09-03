# ch5_monte_carlo/examples/mc_control_onpolicy_gridworld.py
import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4

__all__ = ["mc_control_onpolicy", "ACTIONS", "generate_episode_onpolicy"]

# Must match the environment's action ordering
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U

def _epsilon_greedy(Q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(Q_row)))
    return int(np.argmax(Q_row))

def generate_episode_onpolicy(env: GridWorld4x4, Q: np.ndarray, epsilon: float,
                              rng: np.random.Generator, max_steps: int = 10_000):
    """Start from a random non-terminal state; follow ε-greedy w.r.t. Q throughout."""
    non_terminal = [s for s in env.S if not env.is_terminal(s)]
    s = non_terminal[rng.integers(len(non_terminal))]
    S, A = len(env.S), len(env.A)

    states, actions, rewards = [s], [], [0.0]
    steps = 0
    while not env.is_terminal(s) and steps < max_steps:
        a = _epsilon_greedy(Q[env.s2i[s]], epsilon, rng)
        actions.append(a)
        sp, r = env.step(s, a)
        rewards.append(float(r))
        s = sp
        states.append(s)
        steps += 1

    # first-visit returns
    gamma = env.gamma
    G = 0.0
    returns = np.zeros(len(actions), dtype=float)
    for t in range(len(actions) - 1, -1, -1):
        G = rewards[t + 1] + gamma * G
        returns[t] = G
    return states[:-1], actions, returns

def mc_control_onpolicy(env: GridWorld4x4, episodes: int = 5000,
                        epsilon: float = 0.1, gamma: float | None = None,
                        seed: int | None = None):
    """
    On-policy Monte Carlo control using ε-greedy behavior/target policy (no exploring starts).
    Returns:
        Q:  (S,A) table
        pi: (S,A) deterministic greedy policy derived from Q
    """
    rng = np.random.default_rng(seed)
    S, A = len(env.S), len(env.A)
    if gamma is None:
        gamma = float(env.gamma)

    Q = np.zeros((S, A), dtype=float)
    N = np.zeros((S, A), dtype=float)

    for _ in range(episodes):
        states, actions, returns = generate_episode_onpolicy(env, Q, epsilon, rng)
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

    # deterministic greedy policy
    pi = np.zeros((S, A), dtype=float)
    pi[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return Q, pi

if __name__ == "__main__":
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    Q, pi = mc_control_onpolicy(env, episodes=3000, epsilon=0.1, seed=0)
    s0 = env.s2i[(0, 0)]
    print("Q(start):", Q[s0])
    print("Greedy action at start:", int(np.argmax(pi[s0])))
