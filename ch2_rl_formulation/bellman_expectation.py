import numpy as np
from utils.gridworld import GridWorld, Pos

def policy_evaluation(env: GridWorld, gamma: float = 1.0, theta: float = 1e-10):
    """Evaluate a fixed deterministic greedy-to-goal policy on a tabular GridWorld.
    Policy: move right if possible else up (monotone toward goal at (0,3)).
    Rewards: step cost -1, terminal 0.
    """
    nS = env.n_rows * env.n_cols
    V = np.zeros(nS, dtype=float)

    def greedy_action(s: Pos) -> int:
        # Prefer right (3), then up (0), else whatever is closer to goal
        if s.c < env.goal.c:
            return 3
        if s.r > env.goal.r:
            return 0
        return 0  # already in final column: move up

    while True:
        delta = 0.0
        for s in env.S:
            i = env.index(s)
            if env.is_terminal(s):
                continue
            a = greedy_action(s)
            s2, r = env.step(s, a)
            j = env.index(s2)
            v_new = r + gamma * V[j]
            delta = max(delta, abs(v_new - V[i]))
            V[i] = v_new
        if delta < theta:
            break
    return V.reshape(env.n_rows, env.n_cols)

if __name__ == "__main__":
    env = GridWorld()
    V = policy_evaluation(env)
    print(V)
