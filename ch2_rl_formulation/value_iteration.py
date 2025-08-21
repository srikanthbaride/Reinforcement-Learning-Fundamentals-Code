import numpy as np
from utils.gridworld import GridWorld

def value_iteration(env: GridWorld, gamma: float = 1.0, theta: float = 1e-10):
    nS = env.n_rows * env.n_cols
    V = np.zeros(nS, dtype=float)

    while True:
        delta = 0.0
        V_new = V.copy()
        for s in env.S:
            i = env.index(s)
            if env.is_terminal(s):
                V_new[i] = 0.0
                continue
            q_vals = []
            for a in env.A:
                s2, r = env.step(s, a)
                j = env.index(s2)
                q_vals.append(r + gamma * V[j])
            V_new[i] = max(q_vals)
            delta = max(delta, abs(V_new[i] - V[i]))
        V = V_new
        if delta < theta:
            break

    # Greedy policy extraction
    pi = np.zeros(nS, dtype=int)
    for s in env.S:
        i = env.index(s)
        if env.is_terminal(s):
            pi[i] = -1
            continue
        q_vals = []
        for a in env.A:
            s2, r = env.step(s, a)
            j = env.index(s2)
            q_vals.append(r + gamma * V[j])
        pi[i] = int(np.argmax(q_vals))
    return V.reshape(env.n_rows, env.n_cols), pi.reshape(env.n_rows, env.n_cols)

if __name__ == "__main__":
    env = GridWorld()
    V, pi = value_iteration(env)
    print("Optimal V*:")
    print(V)
    print("\nGreedy policy (0=↑,1=↓,2=←,3=→, -1=terminal):")
    print(pi)
