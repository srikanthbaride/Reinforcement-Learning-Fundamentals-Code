from __future__ import annotations
import numpy as np

def policy_evaluation(states, actions, P, R, pi, gamma: float = 1.0, theta: float = 1e-6):
    """Iterative policy evaluation.
    states: list-like of states (indices 0..S-1)
    actions: list-like of actions (indices 0..A-1)
    P: shape [S, A, S] transition probabilities
    R: shape [S, A, S] expected rewards
    pi: shape [S, A] policy (row-stochastic)
    """
    S = len(states)
    V = np.zeros(S, dtype=float)
    while True:
        delta = 0.0
        for s in range(S):
            v_old = V[s]
            V[s] = sum(
                pi[s, a] * sum(P[s, a, s2] * (R[s, a, s2] + gamma * V[s2]) for s2 in range(S))
                for a in range(len(actions))
            )
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration(states, actions, P, R, gamma: float = 1.0, theta: float = 1e-6):
    """Howard's policy iteration."""
    S, A = len(states), len(actions)
    pi = np.ones((S, A)) / A
    V = np.zeros(S, dtype=float)
    stable = False

    while not stable:
        V = policy_evaluation(states, actions, P, R, pi, gamma, theta)
        stable = True
        for s in range(S):
            old_action = np.argmax(pi[s])
            q_values = [
                sum(P[s, a, s2] * (R[s, a, s2] + gamma * V[s2]) for s2 in range(S))
                for a in range(A)
            ]
            best = int(np.argmax(q_values))
            pi[s] = np.eye(A)[best]
            if best != old_action:
                stable = False
    return pi, V

def value_iteration(states, actions, P, R, gamma: float = 1.0, theta: float = 1e-6):
    """Bellman optimality updates until convergence."""
    S, A = len(states), len(actions)
    V = np.zeros(S, dtype=float)
    while True:
        delta = 0.0
        for s in range(S):
            v_old = V[s]
            q_values = [
                sum(P[s, a, s2] * (R[s, a, s2] + gamma * V[s2]) for s2 in range(S))
                for a in range(A)
            ]
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    # Derive greedy policy
    pi = np.zeros((S, A))
    for s in range(S):
        q_values = [
            sum(P[s, a, s2] * (R[s, a, s2] + gamma * V[s2]) for s2 in range(S))
            for a in range(A)
        ]
        best = int(np.argmax(q_values))
        pi[s] = np.eye(A)[best]
    return pi, V
