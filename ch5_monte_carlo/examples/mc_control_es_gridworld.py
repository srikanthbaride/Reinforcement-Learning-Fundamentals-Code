# ch5_monte_carlo/examples/mc_control_es_gridworld.py
import numpy as np
from collections import defaultdict
from ch2_rl_formulation.gridworld import GridWorld4x4  # your existing env

ACTIONS = [0, 1, 2, 3]  # R, L, D, U (consistent with your env)

def random_start(env: GridWorld4x4):
    s = env.S[np.random.randint(len(env.S))]
    a = np.random.choice(ACTIONS)
    return s, a

def step(env: GridWorld4x4, s, a):
    # env exposes P[s_idx][a] -> list of (prob, s', r)
    s_idx = env.s2i[s]
    trans = env.P[s_idx][a]
    probs = [p for (p, _, _) in trans]
    i = np.random.choice(len(trans), p=probs)
    _, sp_idx, r = trans[i]
    return env.i2s[sp_idx], r

def generate_episode_es(env, pi, gamma=1.0):
    s0, a0 = random_start(env)  # Exploring start
    episode = [(s0, a0, None)]
    s = s0
    a = a0
    done = (s == env.goal)
    rewards = []
    while not done:
        sp, r = step(env, s, a)
        rewards.append(r)
        s = sp
        if s == env.goal:
            break
        # follow current policy after the start
        a = pi[s]
        episode.append((s, a, None))
    return episode, rewards  # rewards aligned with transitions

def mc_es_control(env, episodes=5000, gamma=0.9):
    Q = defaultdict(lambda: 0.0)
    N = defaultdict(int)
    # start with arbitrary deterministic policy
    pi = {s: np.random.choice(ACTIONS) for s in env.S}
    for _ in range(episodes):
        ep, rewards = generate_episode_es(env, pi, gamma)
        G = 0.0
        visited = set()
        # process backwards
        for t in range(len(ep) - 1, -1, -1):
            s, a, _ = ep[t]
            G = gamma * G + rewards[t] if t < len(rewards) else G
            if (s, a) not in visited:
                N[(s, a)] += 1
                Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]  # incremental mean
                visited.add((s, a))
                # greedy improvement
                best_a = max(ACTIONS, key=lambda act: Q[(s, act)])
                pi[s] = best_a
    return Q, pi

if __name__ == "__main__":
    np.random.seed(0)
    env = GridWorld4x4(step_reward=0.0, goal=(0, 3))
    Q, pi = mc_es_control(env, episodes=3000, gamma=0.9)
    # print a small slice of the learned greedy policy arrows
    arrows = {0: "â†’", 1: "â†", 2: "â†“", 3: "â†‘"}
    for i in range(env.n):
        row = []
        for j in range(env.n):
            s = (i, j)
            row.append(" G " if s == env.goal else f" {arrows[pi[s]]} ")
        print("".join(row))

