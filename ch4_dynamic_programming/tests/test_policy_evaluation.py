# ch4_dynamic_programming/tests/test_policy_evaluation.py
import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4
from ch4_dynamic_programming.policy_evaluation import policy_evaluation

def test_policy_evaluation_converges_uniform():
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    S, A = len(env.S), len(env.A)
    pi = np.full((S, A), 1.0 / A, dtype=float)  # uniform random policy
    V = policy_evaluation(env, pi, theta=1e-9, max_iter=100000)
    # basic sanity: terminal state's value must be 0
    g_idx = env.s2i[env.goal]
    assert np.isclose(V[g_idx], 0.0, atol=1e-10)
    # values should be finite and <= 0 (since all step costs are <= 0)
    assert np.all(np.isfinite(V))
    assert np.all(V <= 1e-12)
