import numpy as np

def policy_evaluation(S, A, P, R, pi, gamma=1.0, theta=1e-10):
    """
    Tabular policy evaluation for general R(s,a,s').
    Inputs:
      - S: list of states
      - A: list of actions
      - P: [|S|, |A|, |S'|] transition probabilities
      - R: [|S|, |A|, |S'|] rewards
      - pi: [|S|, |A|] policy (row-stochastic; can be deterministic one-hot)
      - gamma: discount factor
      - theta: convergence threshold (max delta)
    Returns:
      - V: np.ndarray of shape [|S|]
    """
    nS, nA, nSp = P.shape
    assert nS == len(S) and nA == len(A) and nSp == nS
    assert pi.shape == (nS, nA)

    V = np.zeros(nS, dtype=float)
    while True:
        delta = 0.0
        V_new = np.zeros_like(V)
        for s in range(nS):
            val = 0.0
            for a in range(nA):
                p_sa = pi[s, a]
                if p_sa == 0.0:
                    continue
                val += p_sa * np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
            V_new[s] = val
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta:
            break
    return V

def q_from_v(S, A, P, R, V, gamma=1.0):
    nS, nA, _ = P.shape
    Q = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        for a in range(nA):
            Q[s, a] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
    return Q
