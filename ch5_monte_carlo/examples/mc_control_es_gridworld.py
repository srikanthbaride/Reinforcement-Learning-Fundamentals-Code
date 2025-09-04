import numpy as np

def _greedy_policy_from_Q(Q):
    return np.argmax(Q, axis=1)  # ndarray (nS,)

def _generate_episode_es(env, Q, gamma, max_steps=500):
    nS, nA = len(env.S), len(env.A)

    # random non-terminal start
    while True:
        s0 = env.S[np.random.randint(nS)]
        if s0 != env.goal:
            break
    env._state = s0  # simple env

    a0 = np.random.randint(nA)  # exploring start
    traj = []
    s = s0

    sp, r, done = env.step(a0)
    traj.append((s, a0, r))
    s = sp
    if done:
        return traj

    greedy = _greedy_policy_from_Q(Q)
    for _ in range(max_steps - 1):
        a = greedy[env.s2i[s]]
        sp, r, done = env.step(a)
        traj.append((s, a, r))
        s = sp
        if done:
            break
    return traj

def mc_es_control(env, episodes=1500, gamma=0.9, seed=None):
    """
    Monte Carlo Control with Exploring Starts (MC-ES).
    Returns (Q, pi_dict) where:
      - Q is (nS, nA)
      - pi_dict[s_tuple] = greedy action index (int)
    """
    if seed is not None:
        np.random.seed(seed)

    nS, nA = len(env.S), len(env.A)
    Q = np.zeros((nS, nA), dtype=float)
    returns_sum = np.zeros_like(Q)
    returns_cnt = np.zeros_like(Q)

    for _ in range(episodes):
        episode = _generate_episode_es(env, Q, gamma)
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            s_idx = env.s2i[s_t]
            G = r_t + gamma * G
            key = (s_idx, a_t)
            if key not in visited:
                returns_sum[s_idx, a_t] += G
                returns_cnt[s_idx, a_t] += 1
                Q[s_idx, a_t] = returns_sum[s_idx, a_t] / returns_cnt[s_idx, a_t]
                visited.add(key)

    greedy = _greedy_policy_from_Q(Q)  # (nS,)
    # Convert to dict keyed by state tuple
    pi_dict = {env.i2s[s_idx]: int(greedy[s_idx]) for s_idx in range(nS)}
    return Q, pi_dict
