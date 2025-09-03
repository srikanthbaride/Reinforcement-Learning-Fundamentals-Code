import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4
from ch4_dynamic_programming.value_iteration import value_iteration

def main():
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    pi_star, V_star, sweeps = value_iteration(env.P, env.R, gamma=env.gamma, theta=1e-10)
    print("Value Iteration sweeps:", sweeps)
    print("V* reshaped (4x4):\n", V_star.reshape(4, 4))
    print("Greedy actions (0=R,1=L,2=D,3=U):\n", np.argmax(pi_star, axis=1).reshape(4, 4))

if __name__ == "__main__":
    main()

