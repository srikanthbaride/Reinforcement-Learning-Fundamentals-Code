from __future__ import annotations
import numpy as np
from .gridworld import GridWorld4x4

def value_iteration(env: GridWorld4x4,
                    gamma: float = 0.9,
                    theta: float = 1e-8,
                    max_iter: int = 10000):
    S, A = env.num_states, env.num_actions
    V = np.zeros(S, dtype=float)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(S):
            q_sa = np.array([
                sum(tr.p * (tr.r + gamma * V[tr.sp]) for tr in env.P[s][a])
                for a in range(A)
            ])
            v_new = q_sa.max()
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        q_sa = np.array([
            sum(tr.p * (tr.r + gamma * V[tr.sp]) for tr in env.P[s][a])
            for a in range(A)
        ])
        pi[s] = int(q_sa.argmax())
    return V, pi

if __name__ == "__main__":
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3))
    V, pi = value_iteration(env, gamma=0.9)
    print("Optimal V(s):\n", np.round(V.reshape(4, 4), 2))
    print("Greedy π(s):\n", pi.reshape(4, 4))
