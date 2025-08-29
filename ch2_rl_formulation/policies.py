import numpy as np

def greedy_toward_goal_policy(env):
    S, A = env.states(), env.actions()
    pi = np.zeros((len(S), len(A)))
    goal_r, goal_c = env.goal
    for si, (r, c) in enumerate(S):
        if (r, c) == env.goal:
            continue
        best_a = None; best_d = 1e9
        for a in A:
            s2 = env._next_state((r, c), a)
            d = abs(s2[0]-goal_r) + abs(s2[1]-goal_c)
            if d < best_d:
                best_d = d; best_a = a
        pi[si, env.action_index(best_a)] = 1.0
    return pi

def greedy_from_q(Q: np.ndarray) -> np.ndarray:
    nS, nA = Q.shape
    pi = np.zeros_like(Q)
    best = Q.argmax(axis=1)
    pi[np.arange(nS), best] = 1.0
    return pi

def epsilon_greedy(Q: np.ndarray, eps: float=0.1) -> np.ndarray:
    nS, nA = Q.shape
    pi = np.full((nS, nA), eps/nA, dtype=float)
    best = Q.argmax(axis=1)
    pi[np.arange(nS), best] += 1.0 - eps
    return pi
