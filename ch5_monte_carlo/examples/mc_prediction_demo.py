# ch5_monte_carlo/examples/mc_prediction_demo.py

import numpy as np
from collections import defaultdict

def generate_episode(p=0.5):
    """Generate one episode in the two-state MDP.
    Returns a list of (state, reward)."""
    episode = []
    state = "A"
    while state == "A":
        if np.random.rand() < p:
            episode.append(("A", 0))   # self-loop in A
            state = "A"
        else:
            episode.append(("A", 0))   # A -> B
            state = "B"
    # B -> Terminal with +1 reward
    episode.append(("B", 1))
    return episode

def mc_prediction(episodes=5000, p=0.5, gamma=0.9, first_visit=True):
    """Monte Carlo prediction for the two-state MDP."""
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(episodes):
        episode = generate_episode(p)
        G, visited = 0, set()
        # process backward
        for t in reversed(range(len(episode))):
            s, r = episode[t]
            G = gamma * G + r
            if first_visit:
                if s not in visited:
                    returns[s].append(G)
                    V[s] = np.mean(returns[s])
                    visited.add(s)
            else:  # every-visit
                returns[s].append(G)
                V[s] = np.mean(returns[s])
    return V

if __name__ == "__main__":
    np.random.seed(42)
    V_fv = mc_prediction(episodes=5000, first_visit=True)
    V_ev = mc_prediction(episodes=5000, first_visit=False)

    # Analytic values
    gamma, p = 0.9, 0.5
    vA_true = (gamma**2 * (1 - p)) / (1 - p * gamma)
    vB_true = gamma

    print("First-visit MC:", dict(V_fv))
    print("Every-visit MC:", dict(V_ev))
    print(f"True values:   A={vA_true:.5f}, B={vB_true:.5f}")

