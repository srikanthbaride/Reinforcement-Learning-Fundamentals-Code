import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4
from ch4_dynamic_programming.policy_iteration import policy_iteration
from ch4_dynamic_programming.value_iteration import value_iteration

def deterministic_next(env: GridWorld4x4, s: int, a: int) -> int:
    # P[s,a,:] is one-hot under determinism
    return int(np.argmax(env.P[s, a]))

def simulate(env: GridWorld4x4, pi: np.ndarray, start=(3,0), max_steps=100):
    s = env.s2i[start]
    steps = 0
    while s != env.goal_idx and steps < max_steps:
        a = int(np.argmax(pi[s]))
        s = deterministic_next(env, s, a)
        steps += 1
    return steps, (s == env.goal_idx)

def test_policy_iteration_reaches_goal_short_paths():
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    pi_star, V_star, k = policy_iteration(env.P, env.R, gamma=1.0, theta=1e-10)

    # From bottom-left (3,0) Manhattan distance to (0,3) is 6.
    steps, ok = simulate(env, pi_star, start=(3,0))
    assert ok and steps <= 6

def test_value_iteration_matches_policy_iteration_value():
    env = GridWorld4x4(step_reward=-1.0, goal=(0,3), gamma=1.0)
    pi_vi, V_vi, sweeps = value_iteration(env.P, env.R, gamma=1.0, theta=1e-10)
    pi_pi, V_pi, k = policy_iteration(env.P, env.R, gamma=1.0, theta=1e-10)

    assert np.allclose(V_vi, V_pi, atol=1e-8)
    assert np.array_equal(np.argmax(pi_vi, axis=1), np.argmax(pi_pi, axis=1))
