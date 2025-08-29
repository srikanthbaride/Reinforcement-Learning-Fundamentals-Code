import numpy as np
from ..gridworld import GridWorld4x4
from ..value_iteration import value_iteration

if __name__ == "__main__":
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3))
    V_star, pi_star = value_iteration(env, gamma=0.9, theta=1e-10)
    print("Optimal V* (gamma=0.9):")
    print(np.round(V_star.reshape(4, 4), 2))
    print("\nGreedy π* (0:R,1:L,2:D,3:U):")
    print(pi_star.reshape(4, 4))
