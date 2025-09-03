# ch5_monte_carlo/examples/mc_control_onpolicy_gridworld.py

import numpy as np
from collections import defaultdict
from ch2_rl_formulation.gridworld import GridWorld4x4  # reuse Chapter 2 env

ACTIONS = [0, 1, 2, 3]  # R, L, D, U

def step(env: GridWorld4x4, s, a):
    """Sample one step given state and action from env's transition model."""
    s_idx = env.s2i[s]
    trans = env.P[s_idx][a]
    probs = [p for (p, _, _) in trans]
    i = np.random.choice(len(trans), p=probs)
    _, sp_idx, r = trans[i]
    return env.i2s[sp_idx], r

def generate_episode(env, pi, epsilon=0.1, gamma=0.9):
    """Generate one episode following epsilon-soft policy pi."""
    s = env.S[np.random.randint(len(env.S))]
    episode, rewards = [], []
    done = (s == env.goal)
    while not done:
        # choose action epsilon-greedily
        if np.random.rand() < epsilon:
            a = np.random.choice(ACTIONS)
        else:
            a = max(ACTIONS, key=lambda act: pi[(s, act)])
        sp, r = step(env, s, a)
        episode.append((s, a))
        rewards.append(r)
        s = sp
        done = (s == env.goal)
    return episode, rewards

def mc_control_onpolicy(env, episodes=5000, gamma=0.9, epsilon=0.1):
    """On-policy MC control with epsilon-soft policies."""
    Q = defaultdict(float)
    N = defaultdict(int)
    pi = {(s, a): 1.0/len(ACTIONS) for s in env.S for a in ACTIONS}  # uniform start

    for _ in range(episodes):
        ep, rewards = generate_episode(env, pi, epsilon, gamma)
        G, visited = 0.0, set()
        # backward return computation
        for t in range(len(ep) - 1, -1, -1):
            s, a = ep[t]
            G = gamma * G + rewards[t]
            if (s, a) not in visited:
                N[(s, a)] += 1
                Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]
                visited.add((s, a))
                # policy improvement: epsilon-greedy
                best_a = max(ACTIONS, key=lambda act: Q[(s, act)])
                for act in ACTIONS:
                    if act == best_a:
                        pi[(s, act)] = 1 - epsilon + epsilon/len(ACTIONS)
                    else:
                        pi[(s, act)] = epsilon/len(ACTIONS)
    return Q, pi

if __name__ == "__main__":
    np.random.seed(1)
    env = GridWorld4x4(step_reward=0.0, goal=(0, 3))
    Q, pi = mc_control_onpolicy(env, episodes=3000, gamma=0.9, epsilon=0.1)

    # Print learned greedy policy (arrows)
    arrows = {0: "â†’", 1: "â†", 2: "â†“", 3: "â†‘"}
    for i in range(env.n):
        row = []
        for j in range(env.n):
            s = (i, j)
            if s == env.goal:
                row.append(" G ")
            else:
                # choose most probable action
                best_a = max(ACTIONS, key=lambda a: pi[(s, a)])
                row.append(f" {arrows[best_a]} ")
        print("".join(row))

