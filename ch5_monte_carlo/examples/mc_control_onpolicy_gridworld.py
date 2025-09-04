# ch5_monte_carlo/examples/mc_control_onpolicy_gridworld.py
import numpy as np

# Tests import ACTIONS from here and use it as a list of action *indices*.
ACTIONS = [0, 1, 2, 3]

def _epsilon_soft_from_Q(Q, epsilon):
    nS, nA = Q.shape
    pi = np.full((nS, nA), epsilon / nA, dtype=float)
    best = Q.argmax(axis=1)
    pi[np.arange(nS), best] += 1.0 - epsilon
    return pi  # ndarray (nS, nA)

def mc_control_onpolicy(env, episodes=2000, gamma=0.9, epsilon=0.1, seed=None):
    """
    First-visit on-policy MC control with ε-soft policies.

    Returns:
      Q : ndarray (nS, nA)
      pi_soft_dict : dict keyed by (state_tuple, action_index) -> π(a|s)
    """
    if seed is not None:
        np.random.seed(seed)

    nS, nA = len(env.S), len(env.A)
    Q = np.zeros((nS, nA), dtype=float)
    returns_sum = np.zeros_like(Q)
    returns_cnt = np.zeros_like(Q)
    pi = _epsilon_soft_from_Q(Q, epsilon)

    for _ in range(episodes):
        # Generate episode under current ε-soft policy
        s = env.reset()
        traj = []
        for _ in range(500):  # safety cap
            s_idx = env.s2i[s]
            a = np.random.choice(nA, p=pi[s_idx])
            sp, r, done = env.step(a)
            traj.append((s, a, r))
            s = sp
            if done:
                break

        # First-visit MC updates
        G = 0.0
        visited = set()
        for t in reversed(range(len(traj))):
            s_t, a_t, r_t = traj[t]
            s_idx = env.s2i[s_t]
            G = r_t + gamma * G
            key = (s_idx, a_t)
            if key not in visited:
                returns_sum[s_idx, a_t] += G
                returns_cnt[s_idx, a_t] += 1
                Q[s_idx, a_t] = returns_sum[s_idx, a_t] / returns_cnt[s_idx, a_t]
                visited.add(key)

        # Improve policy (ε-soft)
        pi = _epsilon_soft_from_Q(Q, epsilon)

    # Convert ε-soft matrix to dict keyed by (state_tuple, action_index)
    pi_soft_dict = {}
    for s_idx in range(nS):
        s_tuple = env.i2s[s_idx]
        for a_idx in range(nA):
            pi_soft_dict[(s_tuple, a_idx)] = float(pi[s_idx, a_idx])

    return Q, pi_soft_dict
