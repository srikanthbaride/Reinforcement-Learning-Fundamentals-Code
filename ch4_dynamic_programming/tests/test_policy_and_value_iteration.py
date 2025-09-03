# ch4_dynamic_programming/tests/test_policy_and_value_iteration.py
import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4
from ch4_dynamic_programming.policy_iteration import policy_iteration
from ch4_dynamic_programming.value_iteration import value_iteration

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def test_policy_iteration_reaches_goal_short_paths():
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    V_pi, pi = policy_iteration(env)
    # With gamma=1 and step_reward=-1, optimal value equals -Manhattan distance to goal
    for s_idx, s in enumerate(env.S):
        d = manhattan(s, env.goal)
        assert np.isclose(V_pi[s_idx], -float(d), atol=1e-6)

def test_value_iteration_matches_policy_iteration_value():
    env = GridWorld4x4(step_reward=-1.0, goal=(0,3), gamma=1.0)
    V_pi, _ = policy_iteration(env)
    V_vi, _ = value_iteration(env)
    assert np.allclose(V_pi, V_vi, atol=1e-8)
