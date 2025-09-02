import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4
from ch4_dynamic_programming.policy_evaluation import policy_evaluation
from ch4_dynamic_programming.utils import uniform_random_policy

def test_policy_evaluation_converges_uniform():
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    nS, nA = len(env.states), len(env.actions)
    pi = uniform_random_policy(nS, nA)

    V = policy_evaluation(env.P, env.R, pi, gamma=1.0, theta=1e-10)
    assert V.shape == (nS,)
    # Terminal state should be 0 (absorbing, zero reward)
    assert abs(V[env.goal_idx]) < 1e-12

    # Values should be less negative closer to the goal (monotone in Manhattan distance)
    distances = np.array([env.manhattan_to_goal(s) for s in range(nS)])
    for d in range(distances.max()):
        near = V[distances == d].max()
        far  = V[distances == d + 1].min()
        assert near >= far  # higher (less negative) when closer to goal
