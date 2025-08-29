import numpy as np
from ch2_rl_formulation.gridworld import GridWorld4x4
from ch2_rl_formulation.evaluation import policy_evaluation_stochastic

def test_expectation_equation_under_uniform_policy():
    env = GridWorld4x4(step_reward=-1.0, goal=(0,3))
    S, A = env.num_states, env.num_actions
    pi = np.full((S,A), 1.0/A, dtype=float)  # equiprobable
    V = policy_evaluation_stochastic(env, pi, gamma=0.9, theta=1e-10)
    assert abs(V[env.terminal]) < 1e-12
    top_right_neighbor = env.s2i[(0,2)]
    bottom_left = env.s2i[(3,0)]
    assert V[top_right_neighbor] > V[bottom_left]
