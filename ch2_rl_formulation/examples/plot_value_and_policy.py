import numpy as np
from ..gridworld import GridWorld4x4
from ..value_iteration import value_iteration
from ..visualize import plot_value_grid, plot_greedy_policy

if __name__ == "__main__":
    env = GridWorld4x4(step_reward=-1.0, goal=(0,3))
    V, pi = value_iteration(env, gamma=0.9, theta=1e-10)
    plot_value_grid(V, title="Optimal V* (gamma=0.9)")
    plot_greedy_policy(pi, title="Greedy Policy π*")
